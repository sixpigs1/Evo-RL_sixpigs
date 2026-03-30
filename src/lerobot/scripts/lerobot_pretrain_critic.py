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
Critic Pre-training Script for RL Token (RLT).

Trains the Double-Q Critic network *offline* from a standard LeRobot 3.0 dataset
(teleoperation data), as a warm-start before online RL.

Key design notes
----------------
- No actor training: teleop data has no a vs a_ref distinction, so we can only
  train the critic.
- RL Token extraction: requires a trained PI05RLT VLA checkpoint. The VLA is run
  in eval/no-grad mode to extract z_rl for every frame.
- Chunk-level transition construction: from a trajectory of length T and chunk
  size C, we build transitions
      (x_t, a_{t:t+C-1}, r_t, x_{t+C}, done)
  where r_t = 1.0 for the LAST chunk of a successful episode, 0.0 otherwise.
  Episode success is determined by the `success` field in the dataset (if
  available) or assumed to be 1 for all episodes (teleoperation data is assumed
  to be successful demonstrations).
- Target Q: uses Double-Q min-of-two-targets with the ACTOR's action as a'.
  Because we have no trained actor yet, we use a_{t+C:t+2C-1} from the dataset
  as a proxy for a'.  This is a behaviour-cloning approximation of the Bellman
  target and is reasonable for pre-training.

Transition format (mirrors RLTTransition in rlt_buffer.py):
    x_t       = (z_rl_t,  s^p_t)
    a_{t:t+C} = action chunk (from teleop, used as both a and a_ref)
    r_t       = 0 except last chunk of successful episode -> 1
    x_{t+C}   = (z_rl_{t+C}, s^p_{t+C})
    done      = True for last chunk

Usage
-----
lerobot-pretrain-critic \\
    --vla_checkpoint=/home/dell/yzw/RLT/output/rlt_joint/last_checkpoint \\
    --dataset.repo_id=sixpigs1/InsertTube \\
    --dataset.root=/home/dell/yzw/RLT/demoTubeData \\
    --output_dir=/home/dell/yzw/RLT/output/pretrain_critic \\
    --chunk_size=10 \\
    --steps=10000 \\
    --batch_size=256 \\
    --lr=3e-4 \\
    --gamma=0.99 \\
    --tau=0.005 \\
    --device=cuda \\
    --wandb.enable=true \\
    --wandb.project=RLT
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.default import WandBConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.rl.rlt_actor_critic import DoubleCritic
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CriticPretrainConfig:
    """Configuration for offline Critic pre-training from teleop data."""

    # --- VLA checkpoint (must be a trained PI05RLT model) ---
    vla_checkpoint: str = "outputs/rlt_joint/last_checkpoint"

    # --- Dataset ---
    dataset_repo_id: str = "sixpigs1/InsertTube"
    dataset_root: Optional[str] = None        # local root path (None = HuggingFace Hub)
    task: str = "perform the manipulation task"

    # --- Chunk-level RL settings ---
    chunk_size: int = 10          # C: action chunk length
    fps: int = 10                 # dataset FPS (used to compute delta_timestamps)
    gamma: float = 0.99           # discount factor
    # Whether to treat ALL episodes as successful (True = all r_T=1).
    # Set to False if the dataset has a 'success' field.
    assume_success: bool = True

    # --- Critic architecture (must match the online RL settings) ---
    critic_hidden_dim: int = 512
    critic_num_layers: int = 3

    # --- Training ---
    steps: int = 10_000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    tau: float = 0.005            # target network soft update rate
    grad_clip_norm: float = 1.0
    log_freq: int = 100
    save_freq: int = 2000

    # --- Output ---
    output_dir: str = "outputs/pretrain_critic"

    # --- System ---
    device: str = "cuda"
    num_workers: int = 4

    # --- WandB ---
    wandb: WandBConfig = field(default_factory=WandBConfig)


# ---------------------------------------------------------------------------
# Dataset-to-transition builder
# ---------------------------------------------------------------------------

class ChunkTransitionDataset(torch.utils.data.Dataset):
    """
    Converts a LeRobotDataset into chunk-level RL transitions offline.

    For each episode of length T, constructs floor(T/C) transitions:
        (z_rl_t, s^p_t, a_{t:t+C}, r_t, z_rl_{t+C}, s^p_{t+C}, done)

    RL token extraction is done lazily: raw observations (images + language)
    are stored and the VLA is used to extract z_rl at collation time.
    This avoids holding all z_rl in RAM at once.

    Because the VLA is large, we pre-extract z_rl for the whole dataset in one
    pass and cache them in CPU memory as float32 tensors.
    """

    def __init__(
        self,
        transitions: list[dict],
    ):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict:
        return self.transitions[idx]


def _extract_state(frame: dict, state_key: str = "observation.state") -> torch.Tensor:
    """Extract proprioceptive state from a dataset frame. Returns (state_dim,)."""
    s = frame.get(state_key)
    if s is None:
        # Fall back to searching for any key that contains 'state'
        for k, v in frame.items():
            if "state" in k and isinstance(v, torch.Tensor):
                s = v
                break
    if s is None:
        raise KeyError(f"Could not find state key in frame. Available: {list(frame.keys())}")
    if isinstance(s, torch.Tensor):
        return s.float().flatten()
    return torch.tensor(s, dtype=torch.float32).flatten()


def _extract_action(frame: dict, action_key: str = "action") -> torch.Tensor:
    """Extract action from a dataset frame. Returns (action_dim,)."""
    a = frame.get(action_key)
    if a is None:
        raise KeyError(f"'action' key not found in frame. Available: {list(frame.keys())}")
    if isinstance(a, torch.Tensor):
        return a.float().flatten()
    return torch.tensor(a, dtype=torch.float32).flatten()


@torch.no_grad()
def build_transitions_from_dataset(
    dataset: LeRobotDataset,
    vla_policy: PI05RLTPolicy,
    chunk_size: int,
    device: str,
    assume_success: bool = True,
    task: str = "perform the manipulation task",
) -> list[dict]:
    """
    Pre-process the entire dataset into a flat list of chunk-level transitions.

    Each transition dict contains:
        rl_token        (rl_token_dim,)   float32 CPU tensor
        state           (state_dim,)      float32 CPU tensor
        action          (C, action_dim)   float32 CPU tensor
        reward          float
        next_rl_token   (rl_token_dim,)   float32 CPU tensor
        next_state      (state_dim,)      float32 CPU tensor
        done            bool

    Returns sorted list, ready to be wrapped in ChunkTransitionDataset.
    """
    vla_policy.eval()
    vla_policy.to(device)

    transitions = []
    num_episodes = dataset.meta.total_episodes
    logging.info(f"[CriticPretrain] Building transitions from {num_episodes} episodes ...")

    for ep_idx in range(num_episodes):
        # Collect all frames for this episode
        # LeRobot 3.0: episodes[ep_idx] stores global frame indices as
        #   "dataset_from_index" (inclusive) and "dataset_to_index" (exclusive)
        ep_meta  = dataset.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx   = int(ep_meta["dataset_to_index"])

        ep_frames = [dataset[frame_idx] for frame_idx in range(from_idx, to_idx)]

        T = len(ep_frames)
        if T < chunk_size + 1:
            # Episode too short for even one transition
            logging.debug(f"  Episode {ep_idx}: skipped (T={T} < C+1={chunk_size+1})")
            continue

        # Determine episode success
        if assume_success:
            ep_success = 1.0
        else:
            # Try to read from last frame
            last = ep_frames[-1]
            ep_success = float(last.get("next.success", last.get("success", 1.0)))

        # Extract states and actions for the whole episode
        states  = [_extract_state(f)  for f in ep_frames]   # list of (state_dim,)
        actions = [_extract_action(f) for f in ep_frames]   # list of (action_dim,)
        state_dim  = states[0].shape[0]
        action_dim = actions[0].shape[0]

        # ------------------------------------------------------------------
        # Extract RL tokens for all frames in this episode via VLA.
        # We process the whole episode in one batch to be efficient.
        # ------------------------------------------------------------------
        rl_tokens = _extract_rl_tokens_for_episode(
            ep_frames, vla_policy, device, task
        )   # (T, rl_token_dim)  on CPU

        # ------------------------------------------------------------------
        # Build chunk-level transitions
        # Transition at step t (0-indexed):
        #   x_t       = (z_rl_t, s_t)
        #   a         = actions[t:t+C]  (C, action_dim)
        #   r         = 1.0 if this is the last chunk AND success, else 0.0
        #   x_{t+C}   = (z_rl_{t+C}, s_{t+C})  (last step padded with x_{T-1})
        #   done      = True if t + C >= T
        # ------------------------------------------------------------------
        num_chunks = T // chunk_size   # number of complete chunks

        for chunk_i in range(num_chunks):
            t_start = chunk_i * chunk_size
            t_end   = t_start + chunk_size   # exclusive

            # Action chunk: (C, action_dim)
            act_chunk = torch.stack(actions[t_start:t_end], dim=0)   # (C, action_dim)

            # Reward: sparse binary, only at last chunk boundary
            is_last_chunk = (t_end >= T)
            reward = ep_success if is_last_chunk else 0.0

            done = is_last_chunk

            # Next state index (clamped to T-1 for the terminal transition)
            t_next = min(t_end, T - 1)

            transitions.append({
                "rl_token":      rl_tokens[t_start].clone(),        # (rl_token_dim,)
                "state":         states[t_start].clone(),            # (state_dim,)
                "action":        act_chunk.clone(),                   # (C, action_dim)
                "reward":        float(reward),
                "next_rl_token": rl_tokens[t_next].clone(),         # (rl_token_dim,)
                "next_state":    states[t_next].clone(),             # (state_dim,)
                "done":          done,
            })

        if (ep_idx + 1) % 10 == 0 or ep_idx == num_episodes - 1:
            logging.info(
                f"  Episode {ep_idx + 1}/{num_episodes} | "
                f"T={T} | chunks={num_chunks} | total_transitions={len(transitions)}"
            )

    logging.info(f"[CriticPretrain] Built {len(transitions)} transitions total.")
    return transitions


@torch.no_grad()
def _extract_rl_tokens_for_episode(
    ep_frames: list[dict],
    vla_policy: PI05RLTPolicy,
    device: str,
    task: str,
) -> torch.Tensor:
    """
    Extract z_rl for every frame in an episode using the VLA policy.

    Returns: (T, rl_token_dim) float32 CPU tensor.

    We process frames one by one to avoid OOM on long episodes.
    Each frame's observation is wrapped into a pseudo-batch and passed through
    PI05RLTPolicy.extract_rl_token().
    """
    from lerobot.utils.constants import (
        OBS_IMAGES,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    T = len(ep_frames)
    rl_token_dim = vla_policy.config.rl_token_dim
    tokenizer_max_len = getattr(vla_policy.config, "tokenizer_max_length", 48)

    rl_tokens = []

    for frame in ep_frames:
        # Build a minimal batch dict with the keys that extract_rl_token needs
        batch: dict = {}

        # Images: find all image keys
        for key, val in frame.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 3:
                # Likely an image: (C, H, W) or (H, W, C) -> add batch dim
                batch[key] = val.unsqueeze(0).to(device)

        # State
        if OBS_STATE in frame:
            s = frame[OBS_STATE]
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.float32)
            batch[OBS_STATE] = s.unsqueeze(0).to(device)

        # Language tokens (use zeros as dummy; we only extract image-based z_rl)
        batch[OBS_LANGUAGE_TOKENS] = torch.zeros(
            1, tokenizer_max_len, dtype=torch.long, device=device
        )
        batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros(
            1, tokenizer_max_len, dtype=torch.bool, device=device
        )

        try:
            z_rl = vla_policy.extract_rl_token(batch)  # (1, rl_token_dim)
            rl_tokens.append(z_rl.squeeze(0).cpu().float())
        except Exception as e:
            logging.warning(f"  [extract_rl_token] Failed for one frame: {e}. Using zeros.")
            rl_tokens.append(torch.zeros(rl_token_dim))

    return torch.stack(rl_tokens, dim=0)   # (T, rl_token_dim)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _collate_transitions(batch: list[dict]) -> dict:
    """Collate a list of transition dicts into batched tensors."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k == "done":
            result[k] = torch.tensor([float(b[k]) for b in batch], dtype=torch.float32)
        elif k == "reward":
            result[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
        else:
            result[k] = torch.stack([b[k] for b in batch], dim=0)
    return result


# ---------------------------------------------------------------------------
# Critic update step
# ---------------------------------------------------------------------------

def update_critic(
    critic: DoubleCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    gamma: float,
    grad_clip_norm: float,
    device: str,
) -> dict:
    """
    Single critic update step.

    Target Q (offline BC approximation):
        Q_hat = r + gamma^C * (1-done) * min(Q1'(x', a'), Q2'(x', a'))

    where a' is taken from the dataset (next action chunk from teleop data).
    This is a behaviour-cloning Bellman backup: we use the demonstrator's
    next action as a proxy for the actor's action, which is a valid warm-start
    for the critic before the actor is trained.

    Returns dict with scalar metrics.
    """
    critic.train()

    rl_token      = batch["rl_token"].to(device)        # (B, rl_token_dim)
    state         = batch["state"].to(device)            # (B, state_dim)
    action        = batch["action"].to(device)           # (B, C, action_dim)
    reward        = batch["reward"].to(device)           # (B,)
    next_rl_token = batch["next_rl_token"].to(device)   # (B, rl_token_dim)
    next_state    = batch["next_state"].to(device)       # (B, state_dim)
    done          = batch["done"].to(device)             # (B,)
    # next_action is the dataset's action at next step (BC proxy for actor)
    next_action   = batch.get("next_action")
    if next_action is not None:
        next_action = next_action.to(device)
    else:
        # If next_action not stored, use zero (conservative lower bound)
        next_action = torch.zeros_like(action)

    chunk_size = action.shape[1]

    # Compute target Q (no grad)
    with torch.no_grad():
        target_q = critic.target_q_min(next_rl_token, next_state, next_action)  # (B, 1)
        target_q = target_q.squeeze(-1)  # (B,)
        # Chunk-level discount: gamma^C
        q_hat = reward + (gamma ** chunk_size) * (1.0 - done) * target_q   # (B,)

    # Online Q values
    q1, q2 = critic(rl_token, state, action)   # (B, 1) each
    q1 = q1.squeeze(-1)
    q2 = q2.squeeze(-1)

    loss_q1 = F.mse_loss(q1, q_hat)
    loss_q2 = F.mse_loss(q2, q_hat)
    loss = loss_q1 + loss_q2

    optimizer.zero_grad()
    loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm)
    optimizer.step()

    return {
        "critic_loss":    loss.item(),
        "critic_loss_q1": loss_q1.item(),
        "critic_loss_q2": loss_q2.item(),
        "q1_mean":        q1.mean().item(),
        "q2_mean":        q2.mean().item(),
        "q_target_mean":  q_hat.mean().item(),
    }


# ---------------------------------------------------------------------------
# Dataset with next_action
# ---------------------------------------------------------------------------

def build_transitions_with_next_action(
    dataset: LeRobotDataset,
    vla_policy: PI05RLTPolicy,
    chunk_size: int,
    device: str,
    assume_success: bool = True,
    task: str = "perform the manipulation task",
) -> list[dict]:
    """
    Same as build_transitions_from_dataset but also stores next_action (a')
    for the Bellman backup.

    next_action = action chunk starting at t+C (clamped to last frame).
    """
    vla_policy.eval()
    vla_policy.to(device)

    transitions = []
    num_episodes = dataset.meta.total_episodes
    logging.info(f"[CriticPretrain] Building transitions with next_action from {num_episodes} episodes ...")

    for ep_idx in range(num_episodes):
        ep_meta  = dataset.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx   = int(ep_meta["dataset_to_index"])
        ep_frames = [dataset[i] for i in range(from_idx, to_idx)]

        T = len(ep_frames)
        if T < chunk_size + 1:
            continue

        if assume_success:
            ep_success = 1.0
        else:
            last = ep_frames[-1]
            ep_success = float(last.get("next.success", last.get("success", 1.0)))

        states  = [_extract_state(f)  for f in ep_frames]
        actions = [_extract_action(f) for f in ep_frames]

        rl_tokens = _extract_rl_tokens_for_episode(ep_frames, vla_policy, device, task)

        num_chunks = T // chunk_size

        for chunk_i in range(num_chunks):
            t_start = chunk_i * chunk_size
            t_end   = t_start + chunk_size

            act_chunk = torch.stack(actions[t_start:t_end], dim=0)

            # Next action chunk (BC proxy for actor's next action)
            t_next_start = t_end
            t_next_end   = min(t_next_start + chunk_size, T)
            if t_next_end - t_next_start == chunk_size:
                next_act_chunk = torch.stack(actions[t_next_start:t_next_end], dim=0)
            else:
                # Pad with last action if episode ends
                next_acts = actions[t_next_start:t_next_end]
                while len(next_acts) < chunk_size:
                    next_acts.append(actions[-1])
                next_act_chunk = torch.stack(next_acts, dim=0)

            is_last_chunk = (t_end >= T)
            reward = ep_success if is_last_chunk else 0.0
            done   = is_last_chunk
            t_next = min(t_end, T - 1)

            transitions.append({
                "rl_token":      rl_tokens[t_start].clone(),
                "state":         states[t_start].clone(),
                "action":        act_chunk.clone(),
                "reward":        float(reward),
                "next_rl_token": rl_tokens[t_next].clone(),
                "next_state":    states[t_next].clone(),
                "next_action":   next_act_chunk.clone(),
                "done":          done,
            })

        if (ep_idx + 1) % 10 == 0 or ep_idx == num_episodes - 1:
            logging.info(
                f"  Episode {ep_idx + 1}/{num_episodes} | T={T} | "
                f"chunks={num_chunks} | total={len(transitions)}"
            )

    logging.info(f"[CriticPretrain] Built {len(transitions)} transitions.")
    return transitions


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

@parser.wrap()
def pretrain_critic(cfg: CriticPretrainConfig):
    """Offline critic pre-training from teleop demonstration data."""
    init_logging()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"[CriticPretrain] Device: {device}")

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    wandb_run = None
    if cfg.wandb.enable and cfg.wandb.project:
        import os
        import wandb
        if cfg.wandb.api_key:
            os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"pretrain_critic_{Path(cfg.output_dir).name}",
            config={
                "chunk_size":       cfg.chunk_size,
                "steps":            cfg.steps,
                "batch_size":       cfg.batch_size,
                "lr":               cfg.lr,
                "gamma":            cfg.gamma,
                "tau":              cfg.tau,
                "critic_hidden":    cfg.critic_hidden_dim,
                "critic_layers":    cfg.critic_num_layers,
                "vla_checkpoint":   cfg.vla_checkpoint,
                "dataset_repo_id":  cfg.dataset_repo_id,
            },
        )
        logging.info(f"[CriticPretrain] WandB run: {wandb_run.url}")

    # ------------------------------------------------------------------
    # Load VLA policy (eval mode, no grad)
    # ------------------------------------------------------------------
    logging.info(f"[CriticPretrain] Loading VLA: {cfg.vla_checkpoint}")
    vla_policy = PI05RLTPolicy.from_pretrained(cfg.vla_checkpoint)
    vla_policy.to(device)
    vla_policy.eval()
    for p in vla_policy.parameters():
        p.requires_grad = False
    logging.info("[CriticPretrain] VLA loaded (frozen).")

    rl_token_dim = vla_policy.config.rl_token_dim

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    logging.info(f"[CriticPretrain] Loading dataset: {cfg.dataset_repo_id}")
    dataset_kwargs = {"repo_id": cfg.dataset_repo_id}
    if cfg.dataset_root:
        dataset_kwargs["root"] = cfg.dataset_root

    dataset = LeRobotDataset(**dataset_kwargs)
    logging.info(
        f"[CriticPretrain] Dataset: {dataset.meta.total_episodes} episodes, "
        f"{dataset.meta.total_frames} frames, fps={dataset.fps}"
    )

    # Infer state_dim and action_dim from first frame
    first_frame = dataset[0]
    state_dim  = _extract_state(first_frame).shape[0]
    action_dim = _extract_action(first_frame).shape[0]
    logging.info(
        f"[CriticPretrain] Dims: rl_token={rl_token_dim}, "
        f"state={state_dim}, action={action_dim}, chunk_size={cfg.chunk_size}"
    )

    # ------------------------------------------------------------------
    # Build transitions (pre-extract all z_rl)
    # ------------------------------------------------------------------
    logging.info("[CriticPretrain] Pre-extracting RL tokens and building transitions ...")
    t0 = time.perf_counter()
    transitions = build_transitions_with_next_action(
        dataset=dataset,
        vla_policy=vla_policy,
        chunk_size=cfg.chunk_size,
        device=device,
        assume_success=cfg.assume_success,
        task=cfg.task,
    )
    elapsed = time.perf_counter() - t0
    logging.info(
        f"[CriticPretrain] Transition building done in {elapsed:.1f}s | "
        f"{len(transitions)} transitions"
    )

    if len(transitions) == 0:
        raise ValueError(
            "No transitions built. Check that your dataset episodes are longer "
            f"than chunk_size={cfg.chunk_size} frames."
        )

    # Log class balance (reward distribution)
    n_pos = sum(1 for t in transitions if t["reward"] > 0)
    logging.info(
        f"[CriticPretrain] Reward balance: "
        f"{n_pos}/{len(transitions)} positive ({100*n_pos/len(transitions):.1f}%)"
    )

    # ------------------------------------------------------------------
    # Create DataLoader
    # ------------------------------------------------------------------
    trans_dataset = ChunkTransitionDataset(transitions)
    dataloader = DataLoader(
        trans_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_transitions,
        pin_memory=(device == "cuda"),
    )

    # ------------------------------------------------------------------
    # Build Critic
    # ------------------------------------------------------------------
    critic = DoubleCritic(
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=cfg.chunk_size,
        hidden_dim=cfg.critic_hidden_dim,
        num_layers=cfg.critic_num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in critic.parameters())
    n_online = sum(p.numel() for p in critic.get_online_parameters())
    logging.info(
        f"[CriticPretrain] Critic params: total={n_params:,} | "
        f"online (trainable)={n_online:,}"
    )

    optimizer = torch.optim.AdamW(
        critic.get_online_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.01
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logging.info(f"[CriticPretrain] Starting training for {cfg.steps} steps ...")
    logging.info(
        f"[CriticPretrain] Batch size={cfg.batch_size} | "
        f"lr={cfg.lr} | gamma={cfg.gamma} | tau={cfg.tau}"
    )

    step = 0
    data_iter = iter(dataloader)
    best_loss = float("inf")

    while step < cfg.steps:
        # Cycle through the dataloader
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        metrics = update_critic(
            critic=critic,
            optimizer=optimizer,
            batch=batch,
            gamma=cfg.gamma,
            grad_clip_norm=cfg.grad_clip_norm,
            device=device,
        )

        # Soft-update target networks after each step
        critic.soft_update(tau=cfg.tau)

        scheduler.step()
        step += 1

        # ------ Logging ------
        if step % cfg.log_freq == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logging.info(
                f"  Step {step:6d}/{cfg.steps} | "
                f"loss={metrics['critic_loss']:.4f} "
                f"(Q1={metrics['critic_loss_q1']:.4f}, Q2={metrics['critic_loss_q2']:.4f}) | "
                f"Q1_mean={metrics['q1_mean']:.3f} | "
                f"Q_target={metrics['q_target_mean']:.3f} | "
                f"lr={lr_now:.2e}"
            )

            if wandb_run:
                wandb_run.log({
                    "train/critic_loss":    metrics["critic_loss"],
                    "train/critic_loss_q1": metrics["critic_loss_q1"],
                    "train/critic_loss_q2": metrics["critic_loss_q2"],
                    "train/q1_mean":        metrics["q1_mean"],
                    "train/q2_mean":        metrics["q2_mean"],
                    "train/q_target_mean":  metrics["q_target_mean"],
                    "train/lr":             lr_now,
                }, step=step)

        # ------ Checkpoint ------
        if step % cfg.save_freq == 0 or step == cfg.steps:
            loss_val = metrics["critic_loss"]
            ckpt = {
                "step":     step,
                "critic":   critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "rl_token_dim":  rl_token_dim,
                    "state_dim":     state_dim,
                    "action_dim":    action_dim,
                    "chunk_size":    cfg.chunk_size,
                    "hidden_dim":    cfg.critic_hidden_dim,
                    "num_layers":    cfg.critic_num_layers,
                },
            }
            ckpt_path = output_dir / f"critic_step_{step:06d}.pth"
            torch.save(ckpt, ckpt_path)
            logging.info(f"[CriticPretrain] Checkpoint saved: {ckpt_path}")

            if loss_val < best_loss:
                best_loss = loss_val
                best_path = output_dir / "critic_best.pth"
                torch.save(ckpt, best_path)
                logging.info(f"[CriticPretrain] New best critic saved (loss={best_loss:.4f})")

    # ------ Final save ------
    final_path = output_dir / "critic_final.pth"
    torch.save({
        "step":     step,
        "critic":   critic.state_dict(),
        "config": {
            "rl_token_dim":  rl_token_dim,
            "state_dim":     state_dim,
            "action_dim":    action_dim,
            "chunk_size":    cfg.chunk_size,
            "hidden_dim":    cfg.critic_hidden_dim,
            "num_layers":    cfg.critic_num_layers,
        },
    }, final_path)
    logging.info(f"[CriticPretrain] Training complete. Final model: {final_path}")
    logging.info(f"[CriticPretrain] Best loss: {best_loss:.4f}")

    if wandb_run:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    register_third_party_plugins()
    pretrain_critic()


if __name__ == "__main__":
    main()
