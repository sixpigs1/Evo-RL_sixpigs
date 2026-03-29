#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team / Evo-RL Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PI05 with RL Token (PI05-RLT) model implementation.

Architecture:
  1. VLA backbone = PI0.5 (PaliGemma + Gemma action expert, flow matching)
  2. RL Token Encoder  g_phi  -- Transformer that takes image embeddings z_{1:M} and
     a learned query embedding e_rl to produce a single compressed RL token z_rl.
  3. RL Token Decoder  d_phi + h_phi  -- Transformer + linear projection that
     autoregressively reconstructs z_{1:M} from [z_rl, z_{1:i-1}] (stop-gradient on z).
     Used only during training to minimize the reconstruction (compression) loss L_ro.

Usage:
  - During PI05-RLT joint training (lerobot_train_rlt.py), the model computes both
    L_vla (flow matching) and L_ro (reconstruction), combined as L_ro + alpha * L_vla.
  - The `extract_rl_token` method is called during online RL to obtain z_rl for
    the Actor-Critic network.
"""

import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    pass

from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    get_gemma_config,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad_torch,
)
from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


# ---------------------------------------------------------------------------
# RL Token Encoder  g_phi
# ---------------------------------------------------------------------------

class RLTokenEncoder(nn.Module):
    """
    Transformer encoder g_phi that compresses VLM image embeddings z_{1:M}
    into a single RL token z_rl.

    Input:
      - image_embeddings: (B, M, vlm_dim)  -- last-layer VLM embeddings of image tokens
      - The module prepends a learned query embedding e_rl.
    Output:
      - rl_token: (B, rl_token_dim)  -- position [M] output of the Transformer, projected

    Architecture (per paper):
      - 4 Transformer encoder layers, hidden_dim=512, heads=8
      - input projection: vlm_dim -> hidden_dim
      - output projection: hidden_dim -> rl_token_dim
    """

    def __init__(
        self,
        vlm_dim: int,
        rl_token_dim: int,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rl_token_dim = rl_token_dim

        # Project VLM embeddings to encoder hidden dim
        self.input_proj = nn.Linear(vlm_dim, hidden_dim)

        # Learned RL token query embedding (prepended)
        self.rl_query_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional encoding (learnable)
        # We support up to 1024 positions (more than enough for typical image tokens)
        self.pos_embedding = nn.Embedding(1024, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project to rl_token_dim
        self.output_proj = nn.Linear(hidden_dim, rl_token_dim)

    def forward(self, image_embeddings: Tensor) -> Tensor:
        """
        Args:
            image_embeddings: (B, M, vlm_dim)  stop-gradient is applied outside.
        Returns:
            rl_token: (B, rl_token_dim)
        """
        B, M, _ = image_embeddings.shape
        device = image_embeddings.device

        # Project to hidden dim
        x = self.input_proj(image_embeddings)          # (B, M, H)

        # Prepend learned RL query
        rl_query = self.rl_query_embedding.expand(B, -1, -1)  # (B, 1, H)
        x = torch.cat([rl_query, x], dim=1)           # (B, M+1, H)

        # Add positional embeddings
        positions = torch.arange(M + 1, device=device)
        x = x + self.pos_embedding(positions).unsqueeze(0)  # broadcast

        # Encode
        x = self.transformer(x)                        # (B, M+1, H)

        # Take the RL query position output (index 0)
        rl_hidden = x[:, 0, :]                         # (B, H)

        # Project to rl_token_dim
        rl_token = self.output_proj(rl_hidden)         # (B, rl_token_dim)
        return rl_token


# ---------------------------------------------------------------------------
# RL Token Decoder  d_phi + h_phi
# ---------------------------------------------------------------------------

class RLTokenDecoder(nn.Module):
    """
    Autoregressive Transformer decoder d_phi that reconstructs image embeddings
    z_{1:M} from the RL token z_rl.

    At position i the decoder sees [z_rl, stop_grad(z_1), ..., stop_grad(z_{i-1})]
    and is asked to predict z_i via a linear head h_phi.

    Architecture:
      - 2 Transformer decoder layers, hidden_dim=512, heads=8
      - The rl_token and previous embeddings are projected to hidden_dim.
      - A causal mask prevents position i from seeing position j > i.
      - h_phi is a linear layer: hidden_dim -> vlm_dim.
    """

    def __init__(
        self,
        vlm_dim: int,
        rl_token_dim: int,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vlm_dim = vlm_dim
        self.hidden_dim = hidden_dim

        # Project rl_token to hidden_dim
        self.rl_token_proj = nn.Linear(rl_token_dim, hidden_dim)

        # Project image embeddings to hidden_dim (for teacher-forced inputs)
        self.emb_proj = nn.Linear(vlm_dim, hidden_dim)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(1024, hidden_dim)

        # Transformer decoder (using encoder-only with causal mask for simplicity)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear output head h_phi: hidden_dim -> vlm_dim
        self.output_head = nn.Linear(hidden_dim, vlm_dim)

    def forward(self, rl_token: Tensor, image_embeddings_sg: Tensor) -> Tensor:
        """
        Autoregressive reconstruction of image embeddings.

        Args:
            rl_token:          (B, rl_token_dim)  -- the compressed RL token
            image_embeddings_sg: (B, M, vlm_dim)  -- stop-gradient targets z_bar_{1:M}

        Returns:
            predictions: (B, M, vlm_dim)  -- predicted z_i for each position i

        The sequence fed to the decoder is:
          [proj(z_rl), proj(z_bar_1), ..., proj(z_bar_{M-1})]  (length M)
        with a causal mask so position i only attends to positions <= i.
        The prediction at position i is h_phi(decoder_out[i]).
        """
        B, M, _ = image_embeddings_sg.shape
        device = rl_token.device

        # Projected RL token: (B, 1, H)
        rl_proj = self.rl_token_proj(rl_token).unsqueeze(1)

        # Projected previous embeddings (teacher forcing, shifted by 1):
        # positions 0..M-1 in output correspond to inputs [z_rl, z_1, ..., z_{M-1}]
        if M > 1:
            prev_embs = self.emb_proj(image_embeddings_sg[:, :-1, :])  # (B, M-1, H)
            x = torch.cat([rl_proj, prev_embs], dim=1)                  # (B, M, H)
        else:
            x = rl_proj  # (B, 1, H)

        # Positional encoding
        positions = torch.arange(M, device=device)
        x = x + self.pos_embedding(positions).unsqueeze(0)

        # Causal mask: position i cannot attend to j > i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(M, device=device)

        # Decode
        out = self.transformer(x, mask=causal_mask, is_causal=True)  # (B, M, H)

        # Linear head to reconstruct vlm_dim
        predictions = self.output_head(out)  # (B, M, vlm_dim)
        return predictions


# ---------------------------------------------------------------------------
# PI05-RLT core model (extends PI05Pytorch)
# ---------------------------------------------------------------------------

class PI05RLTPytorch(PI05Pytorch):
    """
    PI0.5 with RL Token module.

    Adds encoder g_phi and decoder d_phi on top of PI05Pytorch.
    New forward returns (flow_matching_loss, reconstruction_loss) for joint training.
    The method `extract_rl_token` is used during inference / online RL.
    """

    def __init__(self, config: PI05RLTConfig):
        super().__init__(config)
        self.rlt_config = config

        # Determine the VLM hidden dimension from the paligemma config
        vlm_gemma_cfg = get_gemma_config(config.paligemma_variant)
        vlm_dim = vlm_gemma_cfg.width   # e.g. 2048 for gemma_2b

        # RL Token Encoder
        self.rlt_encoder = RLTokenEncoder(
            vlm_dim=vlm_dim,
            rl_token_dim=config.rl_token_dim,
            hidden_dim=config.rlt_encoder_hidden_dim,
            nheads=config.rlt_encoder_nheads,
            num_layers=config.rlt_encoder_layers,
            dropout=config.rlt_encoder_dropout,
        )

        # RL Token Decoder
        self.rlt_decoder = RLTokenDecoder(
            vlm_dim=vlm_dim,
            rl_token_dim=config.rl_token_dim,
            hidden_dim=config.rlt_decoder_hidden_dim,
            nheads=config.rlt_decoder_nheads,
            num_layers=config.rlt_decoder_layers,
            dropout=config.rlt_decoder_dropout,
        )

    def _embed_prefix_and_get_image_embeddings(
        self, images, img_masks, tokens, masks
    ):
        """
        Run the prefix (image + language) through PaliGemma's embedding layer
        and return:
          - prefix_embs: concatenated embeddings (B, N_prefix, vlm_dim)
          - prefix_pad_masks: (B, N_prefix)
          - prefix_att_masks: (B, N_prefix)
          - image_embeddings: (B, M, vlm_dim)  -- only the image token embeddings
          - M: number of image tokens per image (single image for now)
        """
        embs, pad_masks, att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        # Figure out how many image tokens come from the first (real) image
        # embed_image returns (B, num_img_embs, vlm_dim); we need to know M
        with torch.no_grad():
            _sample_img_emb = self.paligemma_with_expert.embed_image(images[0][:1])
            M_per_img = _sample_img_emb.shape[1]

        # Collect image embeddings for all real cameras
        num_real = sum(1 for m in img_masks if m[0].item())
        total_M = M_per_img * num_real

        # image token positions are the first `total_M` positions in prefix_embs
        image_embeddings = embs[:, :total_M, :]   # (B, total_M, vlm_dim)

        return embs, pad_masks, att_masks, image_embeddings, total_M

    def forward_rlt(
        self, images, img_masks, tokens, masks, actions, noise=None, time=None
    ):
        """
        Full forward pass for joint training.

        Returns:
            flow_loss:   (B,) per-sample flow matching loss (for L_vla)
            recon_loss:  scalar reconstruction loss (for L_ro)
            rl_token:    (B, rl_token_dim)  detached RL token (for logging/downstream use)
        """
        # 1. Run standard VLA forward pass ----------------------------------------
        flow_losses = self.forward(images, img_masks, tokens, masks, actions, noise=noise, time=time)
        # flow_losses: (B, C, action_dim)

        # 2. Obtain image embeddings from VLM -------------------------------------
        # We re-run the prefix embedding to get image embeddings.
        # NOTE: We run this inside torch.no_grad() for the VLM part when freezing,
        #       but here we allow gradients to flow into theta_vla (alpha > 0).
        prefix_embs, prefix_pad_masks, prefix_att_masks, image_embeddings, M = \
            self._embed_prefix_and_get_image_embeddings(images, img_masks, tokens, masks)

        # Determine the model dtype from the first q_proj weight (bfloat16 / float16 / float32)
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype

        # Cast embeddings and attention mask to match model dtype (required by SDPA)
        prefix_embs = prefix_embs.to(dtype=model_dtype)

        # Run image embeddings through the VLM's transformer layers to get final-layer embeddings
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=model_dtype)

        # Forward through PaliGemma language model (prefix only)
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        # prefix_out: (B, N_prefix, vlm_dim)
        # Take the image token positions (first M positions)
        # Cast back to float32 for RLT encoder/decoder (which are float32 modules)
        z_image = prefix_out[:, :M, :].float()   # (B, M, vlm_dim)

        # 3. RL Token Encoder -----------------------------------------------------
        rl_token = self.rlt_encoder(z_image)   # (B, rl_token_dim)

        # 4. RL Token Decoder (reconstruction loss) --------------------------------
        # stop-gradient on targets z_bar
        z_image_sg = z_image.detach()           # (B, M, vlm_dim)  stop-gradient
        predictions = self.rlt_decoder(rl_token, z_image_sg)  # (B, M, vlm_dim)

        # L_ro = mean over B and M of squared reconstruction error
        recon_loss = F.mse_loss(predictions, z_image_sg)

        return flow_losses, recon_loss, rl_token.detach()

    @torch.no_grad()
    def extract_rl_token(self, images, img_masks, tokens, masks) -> Tensor:
        """
        Extract RL token z_rl from current observations without gradients.
        Used during online RL rollout and inference.

        Returns:
            rl_token: (B, rl_token_dim)
        """
        self.eval()

        # Get VLM prefix embeddings
        prefix_embs, prefix_pad_masks, prefix_att_masks, image_embeddings, M = \
            self._embed_prefix_and_get_image_embeddings(images, img_masks, tokens, masks)

        # Cast to model dtype (bfloat16 / float16 / float32) to avoid SDPA dtype mismatch
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=model_dtype)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=model_dtype)

        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        # Cast back to float32 for RLT encoder (which is a float32 module)
        z_image = prefix_out[:, :M, :].float()    # (B, M, vlm_dim)

        rl_token = self.rlt_encoder(z_image)   # (B, rl_token_dim)
        return rl_token


# ---------------------------------------------------------------------------
# PI05RLTPolicy (PreTrainedPolicy wrapper)
# ---------------------------------------------------------------------------

class PI05RLTPolicy(PI05Policy):
    """
    PI05 with RL Token policy.

    Inherits from PI05Policy and replaces the inner model with PI05RLTPytorch.
    Overrides `forward` to optionally compute the combined training loss:
        L_total = L_ro + alpha * L_vla

    This policy is registered under the name "pi05_rlt" and can be used with
    the standard lerobot training pipeline.
    """

    config_class = PI05RLTConfig
    name = "pi05_rlt"

    def __init__(self, config: PI05RLTConfig, **kwargs):
        # Call nn.Module.__init__ and set config directly to avoid double init
        nn.Module.__init__(self)
        config.validate_features()
        self.config = config

        # Build the extended model
        self.init_rtc_processor()
        self.model = PI05RLTPytorch(config)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def forward(self, batch: dict, reduction: str = "mean"):
        """
        Joint training forward pass.
        Computes L_total = L_ro + alpha * L_vla.

        Args:
            batch:      standard lerobot training batch
            reduction:  "mean" (default) or "none" (per-sample)

        Returns:
            (loss, loss_dict)
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)

        flow_losses, recon_loss, rl_token = self.model.forward_rlt(
            images, img_masks, tokens, masks, actions
        )

        # Truncate flow_losses to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :original_action_dim]

        alpha = self.config.rlt_alpha

        if reduction == "none":
            per_sample_vla = flow_losses.mean(dim=(1, 2))  # (B,)
            loss = recon_loss + alpha * per_sample_vla.mean()
            loss_dict = {
                "loss": loss.item(),
                "loss_vla": per_sample_vla.mean().item(),
                "loss_recon": recon_loss.item(),
            }
            return per_sample_vla, loss_dict
        else:
            vla_loss = flow_losses.mean()
            loss = recon_loss + alpha * vla_loss
            loss_dict = {
                "loss": loss.item(),
                "loss_vla": vla_loss.item(),
                "loss_recon": recon_loss.item(),
                "loss_per_dim": flow_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            }
            return loss, loss_dict

    @torch.no_grad()
    def extract_rl_token(self, batch: dict) -> Tensor:
        """
        Extract RL token from a batch observation dict.
        Convenience wrapper around PI05RLTPytorch.extract_rl_token.

        Args:
            batch: observation dict (same format as select_action input)
        Returns:
            rl_token: (B, rl_token_dim)
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        return self.model.extract_rl_token(images, img_masks, tokens, masks)
