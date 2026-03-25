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

"""Configuration for PI05-RLT (PI0.5 with RL Token)."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05_rlt")
@dataclass
class PI05RLTConfig(PI05Config):
    """
    Configuration for PI05 with RL Token.

    Inherits all PI05Config settings and adds RL Token specific parameters.

    RL Token Architecture:
      - encoder: Transformer g_phi  (4 layers by default, hidden_dim=512)
      - decoder: Transformer d_phi  (2 layers by default, hidden_dim=512)
      - rl_token_dim: dimension of the compressed RL token (default 2048 per paper)

    Joint Training:
      - alpha: weight for VLA loss in joint training (L_ro + alpha * L_vla)

    Actor-Critic:
      - chunk_size_rl: action chunk length C for online RL (default 10)
      - actor_hidden_dim: hidden dimension for actor MLP layers
      - critic_hidden_dim: hidden dimension for critic MLP layers
      - actor_sigma: fixed std for Gaussian actor
      - beta: weight for VLA reference regularization in actor loss
      - ref_dropout: probability to zero out VLA reference action (reference dropout)
      - gamma: discount factor for RL
      - tau: soft update rate for target networks
    """

    # RL Token encoder/decoder
    rlt_encoder_layers: int = 4
    rlt_encoder_hidden_dim: int = 512
    rlt_encoder_nheads: int = 8
    rlt_encoder_dropout: float = 0.1

    rlt_decoder_layers: int = 2
    rlt_decoder_hidden_dim: int = 512
    rlt_decoder_nheads: int = 8
    rlt_decoder_dropout: float = 0.1

    # RL token output dimension (1 token of dim 2048 per paper)
    rl_token_dim: int = 2048

    # Joint training weight: L_total = L_ro + alpha * L_vla
    rlt_alpha: float = 0.5

    # Actor-Critic settings
    chunk_size_rl: int = 10            # C = 10 per paper
    actor_hidden_dim: int = 512        # 3-layer MLP with hidden_dim=512 per paper
    actor_num_layers: int = 3
    critic_hidden_dim: int = 512
    critic_num_layers: int = 3

    actor_sigma: float = 0.1           # fixed std for Gaussian actor
    beta: float = 0.5                  # reference regularization weight
    ref_dropout: float = 0.5           # probability to zero-out VLA reference action

    gamma: float = 0.99                # RL discount factor
    tau: float = 0.005                 # soft update for target networks

    # Mode switch MLP (optional)
    mode_switch_hidden_dim: int = 256
    mode_switch_num_layers: int = 2

    # Online RL settings
    warmup_steps: int = 500            # N_warmup: pure VLA warmup before RL kicks in
    buffer_capacity: int = 100_000
    rl_batch_size: int = 256
    updates_per_step: int = 5          # G: update-to-data ratio

    def __post_init__(self):
        super().__post_init__()
        # chunk_size_rl must be <= chunk_size
        if self.chunk_size_rl > self.chunk_size:
            raise ValueError(
                f"chunk_size_rl ({self.chunk_size_rl}) must be <= chunk_size ({self.chunk_size})"
            )
