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
Offline Actor-Critic Pre-training Script for RL Token (RLT).

Pre-trains BOTH the Actor (GaussianActor) and Critic (DoubleCritic) offline
from a standard LeRobot 3.0 teleoperation dataset, using the same
RLTActorCritic architecture as online RL.

Key insight
-----------
Teleop data provides:
  - dataset["action"]  : the action the human operator actually executed  →  used as the
                         supervised target (what "good behaviour" looks like)
  - vla.predict_action_chunk(obs) : the VLA's current prediction for the same obs  →  used as
                         ref_action fed to the GaussianActor

This gives us both roles of the online transition:
    ref_action = VLA output  (what the policy would have done)
    action     = teleop demo (what was actually executed / "better" action)

So the Actor can be trained with the same loss as online RL:
    L_actor = -Q(x, a_actor) + beta * ||a_actor - ref_action||^2
And the Critic with the BC Bellman backup:
    Q_hat = r + gamma^C * (1-done) * min(Q1', Q2')(x', a')
    L_critic = MSE(Q1(x,a), Q_hat) + MSE(Q2(x,a), Q_hat)

Memory design
-------------
- Streaming frame-by-frame extraction: images are del'd immediately after z_rl
  and ref_action are extracted.
- All transition fields are stored in numpy memmap files on disk.
- DataLoader reads from memmap (OS page cache, safe for multi-worker).

Transition fields stored per-step
----------------------------------
  rl_token      (D,)       z_rl from VLA encoder
  state         (S,)       proprioceptive state
  action        (C, A)     teleop demo action  (the executed action)
  ref_action    (C, A)     VLA predicted action (reference for Actor)
  next_rl_token (D,)
  next_state    (S,)
  next_action   (C, A)     teleop demo action at t+C  (BC proxy for a')
  reward        ()         1 if last chunk of successful ep, else 0
  done          ()

Usage
-----
lerobot-pretrain-critic \\
    --vla_checkpoint=/path/to/rlt_checkpoint \\
    --dataset_repo_id=sixpigs1/InsertTube \\
    --dataset_root=/path/to/local/data \\
    --output_dir=outputs/pretrain_actor_critic \\
    --chunk_size=10 \\
    --steps=5500 \\
    --batch_size=64 \\
    --actor_update_freq=2
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.default import WandBConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.rl.rlt_actor_critic import RLTActorCritic
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CriticPretrainConfig:
    """Configuration for offline Actor-Critic pre-training from teleop data."""

    # --- VLA checkpoint ---
    vla_checkpoint: str = "outputs/rlt_joint/last_checkpoint"

    # --- Dataset ---
    dataset_repo_id: str = "sixpigs1/InsertTube"
    dataset_root: Optional[str] = None
    task: str = "perform the manipulation task"

    # --- Chunk-level RL ---
    chunk_size: int = 10
    gamma: float = 0.99
    assume_success: bool = True   # treat all demos as successful

    # --- Actor-Critic architecture (must match online RL) ---
    actor_hidden_dim: int = 512
    actor_num_layers: int = 3
    critic_hidden_dim: int = 512
    critic_num_layers: int = 3
    sigma: float = 0.1            # Gaussian actor std
    ref_dropout: float = 0.5      # probability of zeroing ref_action during actor training

    # --- Training ---
    steps: int = 5_500
    batch_size: int = 64
    critic_lr: float = 3e-4
    actor_lr: float = 1e-4        # actor typically uses smaller lr
    weight_decay: float = 1e-4
    tau: float = 0.005            # target network soft update
    grad_clip_norm: float = 1.0
    beta: float = 0.1             # actor ref-action regularisation weight
    # Update actor every N critic updates (delayed actor update, like TD3)
    actor_update_freq: int = 2
    log_freq: int = 20
    save_freq: int = 1000

    # --- Output ---
    output_dir: str = "outputs/pretrain_actor_critic"

    # --- Caching ---
    # If True and the mmap store already exists (from a previous run with the
    # same output_dir), skip the VLA extraction pass entirely and reuse the
    # cached transitions on disk.  Set to False to force re-extraction.
    cache_mmap_store: bool = True

    # --- Memory control ---
    num_workers: int = 0          # 0 = main process; safe for mmap

    # --- System ---
    device: str = "cuda"

    # --- WandB ---
    wandb: WandBConfig = field(default_factory=WandBConfig)


# ---------------------------------------------------------------------------
# Frame-level helpers
# ---------------------------------------------------------------------------

def _extract_state(frame: dict, state_key: str = "observation.state") -> torch.Tensor:
    """Return (state_dim,) float32 tensor."""
    s = frame.get(state_key)
    if s is None:
        for k, v in frame.items():
            if "state" in k and isinstance(v, torch.Tensor):
                s = v
                break
    if s is None:
        raise KeyError(f"No state key in frame. Keys: {list(frame.keys())}")
    return (s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)).float().flatten()


def _extract_action(frame: dict, action_key: str = "action") -> torch.Tensor:
    """Return (action_dim,) float32 tensor (single-step action from parquet)."""
    a = frame.get(action_key)
    if a is None:
        raise KeyError(f"'action' not found. Keys: {list(frame.keys())}")
    return (a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)).float().flatten()


def _build_vla_batch(frame: dict, device: str, tokenizer_max_len: int) -> dict:
    """
    Build the minimal batch dict that PI05RLTPolicy expects.
    Returns a dict with image tensors on `device`, state on `device`,
    and dummy language tokens.  Does NOT retain a reference to `frame`.
    """
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )
    batch: dict = {}
    for key, val in frame.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 3:
            batch[key] = val.unsqueeze(0).to(device)
    if OBS_STATE in frame:
        s = frame[OBS_STATE]
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        batch[OBS_STATE] = s.unsqueeze(0).to(device)
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(
        1, tokenizer_max_len, dtype=torch.long, device=device
    )
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros(
        1, tokenizer_max_len, dtype=torch.bool, device=device
    )
    return batch


@torch.no_grad()
def _extract_rl_token_and_ref_action(
    frame: dict,
    vla_policy: PI05RLTPolicy,
    device: str,
    tokenizer_max_len: int,
    action_dim: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run VLA on one frame.

    Returns:
        z_rl       (rl_token_dim,)   float32 CPU
        ref_action (chunk_size, action_dim)  float32 CPU  — VLA predicted action chunk
    """
    vla_batch = _build_vla_batch(frame, device, tokenizer_max_len)
    try:
        z_rl = vla_policy.extract_rl_token(vla_batch).squeeze(0).cpu().float()
        # predict_action_chunk returns (1, C, action_dim)
        ref_action = vla_policy.predict_action_chunk(vla_batch).squeeze(0).cpu().float()
        # Clamp to actual action_dim (VLA may pad)
        ref_action = ref_action[:chunk_size, :action_dim]
    except Exception as e:
        logging.warning(f"  [VLA extract] failed: {e}. Using zeros.")
        z_rl       = torch.zeros(vla_policy.config.rl_token_dim)
        ref_action = torch.zeros(chunk_size, action_dim)
    return z_rl, ref_action


# ---------------------------------------------------------------------------
# MMap-backed transition storage
# ---------------------------------------------------------------------------

class MmapTransitionStore:
    """
    All transition arrays stored as numpy memmap files on disk.

    Fields:
        rl_token      (N, D)
        state         (N, S)
        action        (N, C, A)    teleop executed action
        ref_action    (N, C, A)    VLA predicted action  ← NEW
        next_rl_token (N, D)
        next_state    (N, S)
        next_action   (N, C, A)    teleop at t+C (BC proxy for a')
        reward        (N,)
        done          (N,)
    """

    FIELDS = [
        "rl_token", "state", "action", "ref_action",
        "next_rl_token", "next_state", "next_action",
        "reward", "done",
    ]

    def __init__(
        self,
        store_dir: Path,
        capacity: int,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
    ):
        self.store_dir    = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.capacity     = capacity
        self.rl_token_dim = rl_token_dim
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.chunk_size   = chunk_size
        self._size        = 0

        shapes = self._field_shapes()
        self._mmaps: dict[str, np.memmap] = {}
        for f, shape in shapes.items():
            path = self.store_dir / f"{f}.bin"
            self._mmaps[f] = np.memmap(
                path, dtype=np.float32, mode="w+", shape=(capacity, *shape)
            )

    def _field_shapes(self) -> dict[str, tuple]:
        D, S, A, C = self.rl_token_dim, self.state_dim, self.action_dim, self.chunk_size
        return {
            "rl_token":      (D,),
            "state":         (S,),
            "action":        (C, A),
            "ref_action":    (C, A),
            "next_rl_token": (D,),
            "next_state":    (S,),
            "next_action":   (C, A),
            "reward":        (),
            "done":          (),
        }

    def append(
        self,
        rl_token:      np.ndarray,
        state:         np.ndarray,
        action:        np.ndarray,
        ref_action:    np.ndarray,
        next_rl_token: np.ndarray,
        next_state:    np.ndarray,
        next_action:   np.ndarray,
        reward:        float,
        done:          float,
    ) -> None:
        i = self._size
        self._mmaps["rl_token"][i]      = rl_token
        self._mmaps["state"][i]         = state
        self._mmaps["action"][i]        = action
        self._mmaps["ref_action"][i]    = ref_action
        self._mmaps["next_rl_token"][i] = next_rl_token
        self._mmaps["next_state"][i]    = next_state
        self._mmaps["next_action"][i]   = next_action
        self._mmaps["reward"][i]        = reward
        self._mmaps["done"][i]          = done
        self._size += 1

    def flush_and_close(self) -> int:
        n = self._size
        for m in self._mmaps.values():
            m.flush()
            del m
        self._mmaps.clear()
        return n

    @classmethod
    def open_readonly(
        cls,
        store_dir: Path,
        size: int,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
    ) -> "MmapTransitionStore":
        obj = object.__new__(cls)
        obj.store_dir     = Path(store_dir)
        obj.capacity      = size
        obj._size         = size
        obj.rl_token_dim  = rl_token_dim
        obj.state_dim     = state_dim
        obj.action_dim    = action_dim
        obj.chunk_size    = chunk_size
        shapes = obj._field_shapes()
        obj._mmaps = {}
        for f, shape in shapes.items():
            path = obj.store_dir / f"{f}.bin"
            obj._mmaps[f] = np.memmap(
                path, dtype=np.float32, mode="r", shape=(size, *shape)
            )
        return obj

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict:
        return {f: torch.from_numpy(np.array(self._mmaps[f][idx])) for f in self.FIELDS}


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MmapCriticDataset(Dataset):
    def __init__(self, store: MmapTransitionStore, size: int):
        self._store = store
        self._size  = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict:
        return self._store[idx]


# ---------------------------------------------------------------------------
# Streaming extraction → mmap
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_mmap_store(
    dataset: LeRobotDataset,
    vla_policy: Optional[PI05RLTPolicy],
    cfg: CriticPretrainConfig,
    store_dir: Path,
    rl_token_dim: int,
    state_dim: int,
    action_dim: int,
) -> int:
    """
    Two-pass streaming extraction.  Only **chunk-boundary frames** are fed to
    the VLA — specifically the first frame of each chunk (used as s_t) and the
    first frame of the *next* chunk (used as s_{t+1}).  All intermediate frames
    within a chunk only need their `state` and `action` values read from the
    parquet columns; they are never pushed through the VLA encoder.

    Concretely, for an episode of length T with chunk_size C we produce
    ``T // C`` transitions.  Each transition needs VLA inference for at most
    **two** distinct frames (t_start and t_end), and consecutive transitions
    share their boundary frame, so the total number of VLA calls per episode
    is at most ``T // C + 1``, not ``T``.

    Cache behaviour
    ---------------
    After a successful extraction a ``meta.npz`` file is written to
    ``store_dir``.  On subsequent runs, if ``cfg.cache_mmap_store=True``
    and ``meta.npz`` is present, the whole extraction is skipped and the
    cached size / dims are returned directly.
    """
    meta_path = store_dir / "meta.npz"

    # ------------------------------------------------------------------
    # Cache hit: reuse existing mmap store
    # ------------------------------------------------------------------
    if cfg.cache_mmap_store and meta_path.exists():
        meta = np.load(meta_path)
        cached_size          = int(meta["size"])
        cached_rl_token_dim  = int(meta["rl_token_dim"])
        cached_state_dim     = int(meta["state_dim"])
        cached_action_dim    = int(meta["action_dim"])
        cached_chunk_size    = int(meta["chunk_size"])
        if (
            cached_rl_token_dim == rl_token_dim
            and cached_state_dim == state_dim
            and cached_action_dim == action_dim
            and cached_chunk_size == cfg.chunk_size
        ):
            logging.info(
                f"[Pretrain] Cache HIT at {store_dir} — "
                f"reusing {cached_size} transitions (skipping VLA extraction). "
                f"Set cache_mmap_store=False to force re-extraction."
            )
            return cached_size
        else:
            logging.warning(
                "[Pretrain] Cache MISMATCH (dims changed) — re-extracting."
            )

    vla_policy.eval()
    tokenizer_max_len = getattr(vla_policy.config, "tokenizer_max_length", 48)
    chunk_size        = cfg.chunk_size
    num_episodes      = dataset.meta.total_episodes

    # ------------------------------------------------------------------
    # Pass 1: count valid transitions (metadata only, no I/O)
    # ------------------------------------------------------------------
    logging.info("[Pretrain] Pass 1/2: counting valid transitions ...")
    ep_info: list[tuple[int, int, int]] = []
    total = 0
    for ep_idx in range(num_episodes):
        ep_meta  = dataset.meta.episodes[ep_idx]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx   = int(ep_meta["dataset_to_index"])
        T = to_idx - from_idx
        if T >= chunk_size + 1:
            ep_info.append((ep_idx, from_idx, to_idx))
            total += T // chunk_size

    logging.info(
        f"[Pretrain] {len(ep_info)} valid episodes, {total} transitions. "
        f"Allocating mmap at {store_dir} ..."
    )
    if total == 0:
        return 0

    store = MmapTransitionStore(
        store_dir=store_dir,
        capacity=total,
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
    )

    # ------------------------------------------------------------------
    # Pass 2: extraction — only chunk-boundary frames go through VLA
    # ------------------------------------------------------------------
    logging.info(
        "[Pretrain] Pass 2/2: extracting transitions "
        "(VLA called only on chunk-boundary frames) ..."
    )
    t0 = time.perf_counter()

    # For progress: count boundary frames (not all frames)
    total_boundary_frames = sum(
        (to_idx - from_idx) // chunk_size + 1  # at most this many per episode
        for _, from_idx, to_idx in ep_info
    )
    ep_bar = tqdm(ep_info, desc="extract episodes", unit="ep", dynamic_ncols=True)
    vla_bar = tqdm(
        total=total_boundary_frames,
        desc="  VLA calls",
        unit="frame",
        dynamic_ncols=True,
        leave=False,
    )

    for ep_idx, from_idx, to_idx in ep_bar:
        T = to_idx - from_idx
        num_chunks = T // chunk_size

        # ---- Step 1: read states and actions for ALL frames in the episode.
        # These are cheap parquet column reads — no images, no VLA.
        states_ep  = np.empty((T, state_dim),  dtype=np.float32)
        actions_ep = np.empty((T, action_dim), dtype=np.float32)
        for t, abs_idx in enumerate(range(from_idx, to_idx)):
            frame = dataset[abs_idx]
            states_ep[t]  = _extract_state(frame).numpy()
            actions_ep[t] = _extract_action(frame).numpy()
            del frame

        # ---- Step 2: VLA inference only on chunk-boundary frames.
        # Boundary set = {t_start for each chunk} ∪ {t_end for the last chunk}.
        # t_end of chunk i == t_start of chunk i+1, so we deduplicate naturally
        # by building a sorted list of unique boundary indices.
        boundary_abs: list[int] = []
        for chunk_i in range(num_chunks):
            b = from_idx + chunk_i * chunk_size
            if not boundary_abs or boundary_abs[-1] != b:
                boundary_abs.append(b)
        # Also add the final "next" frame (t_end of last chunk, capped at T-1)
        last_next = from_idx + min(num_chunks * chunk_size, T - 1)
        if boundary_abs[-1] != last_next:
            boundary_abs.append(last_next)

        # Map absolute index → (z_rl, ref_action)
        vla_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for abs_idx in boundary_abs:
            frame = dataset[abs_idx]
            z_rl, ref_act = _extract_rl_token_and_ref_action(
                frame, vla_policy, cfg.device, tokenizer_max_len, action_dim, chunk_size
            )
            vla_cache[abs_idx] = (z_rl.numpy(), ref_act.numpy())
            del frame
            vla_bar.update(1)

        # ---- Step 3: assemble transitions from cached boundary lookups
        if cfg.assume_success:
            ep_success = 1.0
        else:
            last_frame = dataset[to_idx - 1]
            ep_success = float(
                last_frame.get("next.success", last_frame.get("success", 1.0))
            )
            del last_frame

        for chunk_i in range(num_chunks):
            t_start    = chunk_i * chunk_size
            t_end      = t_start + chunk_size
            is_last    = (t_end >= T)
            t_next_loc = min(t_end, T - 1)          # local index within episode

            abs_start = from_idx + t_start
            abs_next  = from_idx + t_next_loc

            z_cur,  ref_act_cur  = vla_cache[abs_start]
            z_next, _            = vla_cache[abs_next]

            act_np = actions_ep[t_start:t_end]       # (C, A)  teleop demo chunk

            # Next action chunk (BC proxy for a')
            t_ns, t_ne = t_end, min(t_end + chunk_size, T)
            if t_ne - t_ns == chunk_size:
                next_act_np = actions_ep[t_ns:t_ne]
            else:
                pad = np.repeat(actions_ep[[-1]], chunk_size - (t_ne - t_ns), axis=0)
                next_act_np = np.concatenate([actions_ep[t_ns:t_ne], pad], axis=0)

            store.append(
                rl_token      = z_cur,
                state         = states_ep[t_start],
                action        = act_np,
                ref_action    = ref_act_cur,
                next_rl_token = z_next,
                next_state    = states_ep[t_next_loc],
                next_action   = next_act_np,
                reward        = ep_success if is_last else 0.0,
                done          = float(is_last),
            )

        del states_ep, actions_ep, vla_cache

        elapsed = time.perf_counter() - t0
        ep_bar.set_postfix(
            ep=ep_idx, T=T,
            vla_calls=len(boundary_abs),
            written=store._size,
            elapsed=f"{elapsed:.0f}s",
        )

    vla_bar.close()
    ep_bar.close()

    actual = store.flush_and_close()
    logging.info(
        f"[Pretrain] Extraction done: {actual} transitions in "
        f"{time.perf_counter() - t0:.1f}s"
    )

    # ------------------------------------------------------------------
    # Write cache metadata so next run can skip extraction
    # ------------------------------------------------------------------
    np.savez(
        meta_path,
        size=actual,
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
    )
    logging.info(f"[Pretrain] Cache metadata written to {meta_path}")
    return actual


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Update functions (mirrors online_rl.py exactly)
# ---------------------------------------------------------------------------

def _update_critic(
    batch:     dict,
    ac:        RLTActorCritic,
    critic_opt: torch.optim.Optimizer,
    gamma:     float,
    chunk_size: int,
    device:    str,
    grad_clip: float,
) -> dict:
    """
    Bellman update for DoubleCritic.

    Target Q uses the ACTOR's next-action (a' ~ pi(x')) rather than the
    BC proxy — this is consistent with how online_rl.py computes target Q.
    """
    ac.critic.train()

    rl_token      = batch["rl_token"].to(device)
    state         = batch["state"].to(device)
    action        = batch["action"].to(device)       # teleop demo action
    reward        = batch["reward"].to(device)
    next_rl_token = batch["next_rl_token"].to(device)
    next_state    = batch["next_state"].to(device)
    done          = batch["done"].to(device)

    with torch.no_grad():
        B = next_rl_token.shape[0]
        # Use zero ref_action → ref_dropout will handle; consistent with online_rl
        zero_ref = torch.zeros(B, chunk_size, ac.action_dim, device=device)
        next_action, _, _ = ac.actor.sample(next_rl_token, next_state, zero_ref)
        q_next   = ac.critic.target_q_min(next_rl_token, next_state, next_action).squeeze(-1)
        q_target = reward * (gamma ** (chunk_size - 1)) + (1.0 - done) * (gamma ** chunk_size) * q_next

    q1, q2  = ac.critic(rl_token, state, action)
    q1, q2  = q1.squeeze(-1), q2.squeeze(-1)
    loss_q1 = F.mse_loss(q1, q_target)
    loss_q2 = F.mse_loss(q2, q_target)
    loss    = loss_q1 + loss_q2

    critic_opt.zero_grad()
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(ac.critic.get_online_parameters(), grad_clip)
    critic_opt.step()

    return {
        "critic_loss":    loss.item(),
        "critic_loss_q1": loss_q1.item(),
        "critic_loss_q2": loss_q2.item(),
        "q1_mean":        q1.mean().item(),
        "q_target_mean":  q_target.mean().item(),
    }


def _update_actor(
    batch:     dict,
    ac:        RLTActorCritic,
    actor_opt: torch.optim.Optimizer,
    beta:      float,
    device:    str,
    grad_clip: float,
) -> dict:
    """
    Actor update:
        L = -Q_min(x, a_actor) + beta * ||a_actor - ref_action||^2

    ref_action = VLA's prediction for this obs (stored in the mmap).
    This is identical to update_actor() in online_rl.py.
    """
    ac.actor.train()

    rl_token   = batch["rl_token"].to(device)
    state      = batch["state"].to(device)
    ref_action = batch["ref_action"].to(device)   # VLA predicted action

    action, mu, _ = ac.actor.sample(rl_token, state, ref_action)

    q_val       = ac.critic.q_min(rl_token, state, action).squeeze(-1)
    ref_reg     = F.mse_loss(action, ref_action)
    loss_q_term = -q_val.mean()
    loss_ref    = beta * ref_reg
    actor_loss  = loss_q_term + loss_ref

    actor_opt.zero_grad()
    actor_loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(ac.actor.parameters(), grad_clip)
    actor_opt.step()

    return {
        "actor_loss":     actor_loss.item(),
        "actor_loss_q":   loss_q_term.item(),
        "actor_loss_ref": loss_ref.item(),
        "actor_q_mean":   q_val.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@parser.wrap()
def pretrain_critic(cfg: CriticPretrainConfig):
    init_logging()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_dir = output_dir / "mmap_store"

    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"[Pretrain] device={device}")

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    wandb_run = None
    if cfg.wandb.enable and cfg.wandb.project:
        import os
        import wandb
        if getattr(cfg.wandb, "api_key", None):
            os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"pretrain_ac_{output_dir.name}",
            config={k: getattr(cfg, k) for k in (
                "chunk_size", "steps", "batch_size", "critic_lr", "actor_lr",
                "gamma", "tau", "beta", "actor_update_freq",
                "actor_hidden_dim", "critic_hidden_dim",
                "vla_checkpoint", "dataset_repo_id",
            )},
        )
        logging.info(f"[Pretrain] WandB: {wandb_run.url}")

    # ------------------------------------------------------------------
    # Determine if cache already exists so we can skip VLA loading
    # ------------------------------------------------------------------
    meta_path = store_dir / "meta.npz"
    cache_valid = False
    if cfg.cache_mmap_store and meta_path.exists():
        try:
            meta = np.load(meta_path)
            cache_valid = True
            rl_token_dim = int(meta["rl_token_dim"])
            state_dim    = int(meta["state_dim"])
            action_dim   = int(meta["action_dim"])
            logging.info(
                f"[Pretrain] Found existing mmap cache at {store_dir}. "
                f"VLA loading will be skipped."
            )
        except Exception as e:
            logging.warning(f"[Pretrain] Could not read cache meta: {e}. Will re-extract.")
            cache_valid = False

    # ------------------------------------------------------------------
    # Load dataset (always needed for dims if cache is absent)
    # ------------------------------------------------------------------
    logging.info(f"[Pretrain] Loading dataset: {cfg.dataset_repo_id}")
    ds_kwargs: dict = {"repo_id": cfg.dataset_repo_id}
    if cfg.dataset_root:
        ds_kwargs["root"] = cfg.dataset_root
    dataset = LeRobotDataset(**ds_kwargs)
    logging.info(
        f"[Pretrain] Dataset: {dataset.meta.total_episodes} eps, "
        f"{dataset.meta.total_frames} frames, fps={dataset.fps}"
    )

    if not cache_valid:
        # Need actual dims from dataset and VLA
        first_frame = dataset[0]
        state_dim   = _extract_state(first_frame).shape[0]
        action_dim  = _extract_action(first_frame).shape[0]
        del first_frame

        # ------------------------------------------------------------------
        # Load VLA (frozen, eval) — only needed when extraction is required
        # ------------------------------------------------------------------
        logging.info(f"[Pretrain] Loading VLA: {cfg.vla_checkpoint}")
        vla_policy = PI05RLTPolicy.from_pretrained(cfg.vla_checkpoint)
        vla_policy.to(device).eval()
        for p in vla_policy.parameters():
            p.requires_grad = False
        rl_token_dim = vla_policy.config.rl_token_dim
        logging.info(f"[Pretrain] VLA loaded (frozen). rl_token_dim={rl_token_dim}")
    else:
        vla_policy = None  # not needed; extraction will be skipped

    logging.info(
        f"[Pretrain] state_dim={state_dim}, action_dim={action_dim}, "
        f"chunk_size={cfg.chunk_size}, rl_token_dim={rl_token_dim}"
    )

    # ------------------------------------------------------------------
    # Build mmap store (or reuse cache)
    actual_size = build_mmap_store(
        dataset=dataset,
        vla_policy=vla_policy,
        cfg=cfg,
        store_dir=store_dir,
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    if actual_size == 0:
        raise ValueError(
            f"No transitions. All episodes shorter than chunk_size+1={cfg.chunk_size+1}."
        )

    # ------------------------------------------------------------------
    # Re-open read-only for training
    # ------------------------------------------------------------------
    store = MmapTransitionStore.open_readonly(
        store_dir=store_dir,
        size=actual_size,
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=cfg.chunk_size,
    )
    n_pos = int((store._mmaps["reward"][:actual_size] > 0).sum())
    logging.info(
        f"[Pretrain] {actual_size} transitions | "
        f"reward>0: {n_pos} ({100*n_pos/actual_size:.1f}%)"
    )

    dataloader = DataLoader(
        MmapCriticDataset(store, actual_size),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    # ------------------------------------------------------------------
    # Build RLTActorCritic (same as online RL)
    # ------------------------------------------------------------------
    ac = RLTActorCritic(
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=cfg.chunk_size,
        actor_hidden_dim=cfg.actor_hidden_dim,
        actor_num_layers=cfg.actor_num_layers,
        critic_hidden_dim=cfg.critic_hidden_dim,
        critic_num_layers=cfg.critic_num_layers,
        sigma=cfg.sigma,
        ref_dropout=cfg.ref_dropout,
    ).to(device)

    n_critic = sum(p.numel() for p in ac.critic.parameters())
    n_actor  = sum(p.numel() for p in ac.actor.parameters())
    logging.info(f"[Pretrain] Critic params={n_critic:,} | Actor params={n_actor:,}")

    critic_opt = torch.optim.AdamW(
        ac.critic.get_online_parameters(),
        lr=cfg.critic_lr, weight_decay=cfg.weight_decay,
    )
    actor_opt = torch.optim.AdamW(
        ac.actor.parameters(),
        lr=cfg.actor_lr, weight_decay=cfg.weight_decay,
    )
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_opt, T_max=cfg.steps, eta_min=cfg.critic_lr * 0.01
    )
    actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        actor_opt, T_max=cfg.steps, eta_min=cfg.actor_lr * 0.01
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logging.info(
        f"[Pretrain] Training {cfg.steps} steps | "
        f"batch={cfg.batch_size} | critic_lr={cfg.critic_lr} actor_lr={cfg.actor_lr} | "
        f"beta={cfg.beta} actor_update_freq={cfg.actor_update_freq}"
    )

    step       = 0
    best_loss  = float("inf")
    data_iter  = iter(dataloader)
    _ema_closs: float | None = None
    _ema_aloss: float | None = None
    _ema_alpha = 0.1

    pbar = tqdm(total=cfg.steps, desc="pretrain AC", unit="step", dynamic_ncols=True)

    while step < cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # --- Critic update (every step) ---
        c_metrics = _update_critic(
            batch=batch,
            ac=ac,
            critic_opt=critic_opt,
            gamma=cfg.gamma,
            chunk_size=cfg.chunk_size,
            device=device,
            grad_clip=cfg.grad_clip_norm,
        )

        # --- Actor update (every actor_update_freq steps, delayed like TD3) ---
        a_metrics: dict = {}
        if step % cfg.actor_update_freq == 0:
            a_metrics = _update_actor(
                batch=batch,
                ac=ac,
                actor_opt=actor_opt,
                beta=cfg.beta,
                device=device,
                grad_clip=cfg.grad_clip_norm,
            )
            actor_sched.step()

        # Soft-update target networks
        ac.soft_update_targets(tau=cfg.tau)
        critic_sched.step()
        step += 1

        # EMA for display
        _ema_closs = _ema_alpha * c_metrics["critic_loss"] + (1 - _ema_alpha) * (_ema_closs or c_metrics["critic_loss"])
        if a_metrics:
            _ema_aloss = _ema_alpha * a_metrics["actor_loss"] + (1 - _ema_alpha) * (_ema_aloss or a_metrics["actor_loss"])

        pbar.update(1)
        pbar.set_postfix(
            c_loss=f"{c_metrics['critic_loss']:.4f}",
            c_ema=f"{_ema_closs:.4f}",
            a_loss=f"{a_metrics.get('actor_loss', 0):.4f}" if a_metrics else "-",
            Qt=f"{c_metrics['q_target_mean']:.3f}",
        )

        # Logging
        if step % cfg.log_freq == 0:
            lr_c = critic_opt.param_groups[0]["lr"]
            lr_a = actor_opt.param_groups[0]["lr"]
            log_parts = [
                f"  Step {step:6d}/{cfg.steps}",
                f"critic={c_metrics['critic_loss']:.4f}(ema={_ema_closs:.4f})",
                f"Q1={c_metrics['critic_loss_q1']:.4f} Q2={c_metrics['critic_loss_q2']:.4f}",
                f"Qt={c_metrics['q_target_mean']:.3f}",
            ]
            if a_metrics:
                log_parts += [
                    f"actor={a_metrics['actor_loss']:.4f}(ema={_ema_aloss:.4f})",
                    f"a_q={a_metrics['actor_loss_q']:.4f} a_ref={a_metrics['actor_loss_ref']:.4f}",
                ]
            log_parts.append(f"lr_c={lr_c:.1e} lr_a={lr_a:.1e}")
            logging.info(" | ".join(log_parts))

            if wandb_run:
                log_dict = {f"train/{k}": v for k, v in c_metrics.items()}
                log_dict["train/critic_loss_ema"] = _ema_closs
                log_dict["train/lr_critic"] = lr_c
                if a_metrics:
                    log_dict.update({f"train/{k}": v for k, v in a_metrics.items()})
                    log_dict["train/actor_loss_ema"] = _ema_aloss
                    log_dict["train/lr_actor"] = lr_a
                wandb_run.log(log_dict, step=step)

        # Checkpoint
        if step % cfg.save_freq == 0 or step == cfg.steps:
            ckpt = {
                "step":       step,
                "actor":      ac.actor.state_dict(),
                "critic":     ac.critic.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "actor_opt":  actor_opt.state_dict(),
                "config": {
                    "rl_token_dim": rl_token_dim, "state_dim": state_dim,
                    "action_dim":   action_dim,   "chunk_size": cfg.chunk_size,
                    "actor_hidden": cfg.actor_hidden_dim,
                    "critic_hidden": cfg.critic_hidden_dim,
                },
            }
            p = output_dir / f"ac_step_{step:06d}.pth"
            torch.save(ckpt, p)
            logging.info(f"[Pretrain] Saved: {p}")
            if c_metrics["critic_loss"] < best_loss:
                best_loss = c_metrics["critic_loss"]
                torch.save(ckpt, output_dir / "ac_best.pth")
                logging.info(f"[Pretrain] New best critic_loss={best_loss:.4f}")

    pbar.close()

    # Final save
    torch.save(
        {
            "step":   step,
            "actor":  ac.actor.state_dict(),
            "critic": ac.critic.state_dict(),
            "config": {
                "rl_token_dim": rl_token_dim, "state_dim": state_dim,
                "action_dim":   action_dim,   "chunk_size": cfg.chunk_size,
                "actor_hidden": cfg.actor_hidden_dim,
                "critic_hidden": cfg.critic_hidden_dim,
            },
        },
        output_dir / "ac_final.pth",
    )
    logging.info(
        f"[Pretrain] Done. Final={output_dir / 'ac_final.pth'} | "
        f"best_critic_loss={best_loss:.4f}"
    )
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
