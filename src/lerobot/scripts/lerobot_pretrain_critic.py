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

内存优化设计
-----------
旧版本在 "build transitions" 阶段会把所有 episode 的原始帧 (包含图像) 以及
所有 transition 的 rl_token / state / action 一次性存入 Python list，导致
对于大数据集出现数十 GB 的内存占用并发生卡死。

新版本的核心改进：
1. **流式提取，按列存储到 numpy mmap 文件**
   - 每次只在内存中保留"当前帧"的图像，提取完 z_rl 后立刻丢弃图像
   - 提取结果直接写入预分配的 numpy memmap，不在 Python heap 上堆积 tensor
2. **ChunkTransitionDataset 从 mmap 中按需读取**
   - 训练时 DataLoader 的每个 batch 才触发真正的数据拷贝
   - 多进程 worker 可以安全地共享 mmap (read-only)
3. **不缓存 ep_frames 列表**
   - 逐帧调用 dataset[abs_idx]，处理完立即 del 释放图像

Key design notes
----------------
- No actor training: teleop data has no a vs a_ref distinction.
- Chunk-level Bellman backup: a' from dataset next chunk (BC proxy).
- Reward: 0 everywhere, 1 for last chunk of successful demo.

Usage
-----
lerobot-pretrain-critic \\
    --vla_checkpoint=/path/to/rlt_checkpoint \\
    --dataset_repo_id=sixpigs1/InsertTube \\
    --dataset_root=/path/to/local/data \\
    --output_dir=outputs/pretrain_critic \\
    --chunk_size=10 \\
    --steps=10000 \\
    --batch_size=256
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

    # --- VLA checkpoint ---
    vla_checkpoint: str = "outputs/rlt_joint/last_checkpoint"

    # --- Dataset ---
    dataset_repo_id: str = "sixpigs1/InsertTube"
    dataset_root: Optional[str] = None
    task: str = "perform the manipulation task"

    # --- Chunk-level RL ---
    chunk_size: int = 10
    gamma: float = 0.99
    assume_success: bool = True   # treat all episodes as successful demos

    # --- Critic architecture ---
    critic_hidden_dim: int = 512
    critic_num_layers: int = 3

    # --- Training ---
    steps: int = 10_000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    tau: float = 0.005
    grad_clip_norm: float = 1.0
    log_freq: int = 100
    save_freq: int = 2000

    # --- Output ---
    output_dir: str = "outputs/pretrain_critic"

    # --- Memory control ---
    # Number of DataLoader workers (0 = main process, safe for mmap)
    num_workers: int = 0

    # --- System ---
    device: str = "cuda"

    # --- WandB ---
    wandb: WandBConfig = field(default_factory=WandBConfig)


# ---------------------------------------------------------------------------
# Helper: extract state / action scalars from a dataset frame
# ---------------------------------------------------------------------------

def _extract_state(frame: dict, state_key: str = "observation.state") -> torch.Tensor:
    """Return (state_dim,) float32 tensor from a dataset frame."""
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
    """Return (action_dim,) float32 tensor from a dataset frame."""
    a = frame.get(action_key)
    if a is None:
        raise KeyError(f"'action' not found in frame. Keys: {list(frame.keys())}")
    return (a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)).float().flatten()


# ---------------------------------------------------------------------------
# Helper: extract z_rl for a SINGLE frame (images not kept in memory)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_rl_token_single(
    frame: dict,
    vla_policy: PI05RLTPolicy,
    device: str,
    tokenizer_max_len: int,
) -> torch.Tensor:
    """
    Extract z_rl for one frame.  Returns (rl_token_dim,) float32 CPU tensor.
    The caller should `del frame` immediately after to free image memory.
    """
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    batch: dict = {}

    # Images / videos: any 3-D+ tensor
    for key, val in frame.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 3:
            batch[key] = val.unsqueeze(0).to(device)

    # State
    if OBS_STATE in frame:
        s = frame[OBS_STATE]
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        batch[OBS_STATE] = s.unsqueeze(0).to(device)

    # Language (dummy zeros — we only need the visual z_rl)
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(
        1, tokenizer_max_len, dtype=torch.long, device=device
    )
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros(
        1, tokenizer_max_len, dtype=torch.bool, device=device
    )

    try:
        z_rl = vla_policy.extract_rl_token(batch)   # (1, D)
        return z_rl.squeeze(0).cpu().float()
    except Exception as e:
        logging.warning(f"  [extract_rl_token] failed: {e}. Returning zeros.")
        return torch.zeros(vla_policy.config.rl_token_dim)


# ---------------------------------------------------------------------------
# MMap-backed transition storage
# ---------------------------------------------------------------------------

class MmapTransitionStore:
    """
    Stores chunk-level transitions in memory-mapped numpy arrays on disk.

    Layout (N = total transitions):
        rl_token      (N, D)      float32
        state         (N, S)      float32
        action        (N, C, A)   float32
        next_rl_token (N, D)      float32
        next_state    (N, S)      float32
        next_action   (N, C, A)   float32
        reward        (N,)        float32
        done          (N,)        float32

    Write mode: allocated once with mode='w+', flushed and closed after
    extraction.  Re-opened with mode='r' for training (OS page cache shared
    across DataLoader workers — no Python object copies).
    """

    FIELDS = [
        "rl_token", "state", "action",
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
        self._mmaps["next_rl_token"][i] = next_rl_token
        self._mmaps["next_state"][i]    = next_state
        self._mmaps["next_action"][i]   = next_action
        self._mmaps["reward"][i]        = reward
        self._mmaps["done"][i]          = done
        self._size += 1

    def flush_and_close(self) -> int:
        """Flush all files, release mmap objects, return number of rows written."""
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
        """Re-open an already-written store read-only (safe for multi-worker DataLoader)."""
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
# PyTorch Dataset wrapping MmapTransitionStore
# ---------------------------------------------------------------------------

class MmapCriticDataset(Dataset):
    """Read-only Dataset over a MmapTransitionStore."""

    def __init__(self, store: MmapTransitionStore, size: int):
        self._store = store
        self._size  = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict:
        return self._store[idx]


# ---------------------------------------------------------------------------
# Core streaming extraction → mmap
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_mmap_store(
    dataset: LeRobotDataset,
    vla_policy: PI05RLTPolicy,
    cfg: CriticPretrainConfig,
    store_dir: Path,
    rl_token_dim: int,
    state_dim: int,
    action_dim: int,
) -> int:
    """
    Two-pass extraction:
      Pass 1: count valid transitions (no data loaded).
      Pass 2: stream frame-by-frame — extract z_rl, del frame, write to mmap.

    Memory profile during Pass 2:
      - At any point: ONE decoded video frame + current episode's z_rl strip
        (float32 vectors only, no raw images kept).
      - Mmap files live on disk; OS page cache handles hot pages automatically.

    Returns the number of transitions written (int).
    """
    vla_policy.eval()
    tokenizer_max_len = getattr(vla_policy.config, "tokenizer_max_length", 48)
    chunk_size = cfg.chunk_size
    num_episodes = dataset.meta.total_episodes

    # ------------------------------------------------------------------
    # Pass 1: count transitions
    # ------------------------------------------------------------------
    logging.info("[CriticPretrain] Pass 1/2: counting valid transitions ...")
    ep_info: list[tuple[int, int, int]] = []   # (ep_idx, from_abs, to_abs)
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
        f"[CriticPretrain] {len(ep_info)} valid episodes, "
        f"{total} transitions -> allocating mmap store at {store_dir} ..."
    )
    if total == 0:
        return 0

    # ------------------------------------------------------------------
    # Allocate mmap store
    # ------------------------------------------------------------------
    store = MmapTransitionStore(
        store_dir=store_dir,
        capacity=total,
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
    )

    # ------------------------------------------------------------------
    # Pass 2: streaming extraction
    # ------------------------------------------------------------------
    logging.info("[CriticPretrain] Pass 2/2: streaming extraction ...")
    t0 = time.perf_counter()

    for prog_i, (ep_idx, from_idx, to_idx) in enumerate(ep_info):
        T = to_idx - from_idx

        # Allocate per-episode float buffers (small — no images)
        rl_tokens_ep = np.empty((T, rl_token_dim), dtype=np.float32)
        states_ep    = np.empty((T, state_dim),     dtype=np.float32)
        actions_ep   = np.empty((T, action_dim),    dtype=np.float32)

        # Stream frame-by-frame: images live only for this iteration
        for t, abs_idx in enumerate(range(from_idx, to_idx)):
            frame = dataset[abs_idx]          # decode one frame (images in RAM)

            rl_tokens_ep[t] = _extract_rl_token_single(
                frame, vla_policy, cfg.device, tokenizer_max_len
            ).numpy()
            states_ep[t]    = _extract_state(frame).numpy()
            actions_ep[t]   = _extract_action(frame).numpy()

            del frame                         # free image tensors NOW

        # Episode success
        if cfg.assume_success:
            ep_success = 1.0
        else:
            last = dataset[to_idx - 1]
            ep_success = float(last.get("next.success", last.get("success", 1.0)))
            del last

        # Write chunk transitions into mmap
        num_chunks = T // chunk_size
        for chunk_i in range(num_chunks):
            t_start = chunk_i * chunk_size
            t_end   = t_start + chunk_size

            is_last = (t_end >= T)
            t_next  = min(t_end, T - 1)

            # Current action chunk (C, A)
            act_np = actions_ep[t_start:t_end]

            # Next action chunk (BC proxy for a')
            t_ns, t_ne = t_end, min(t_end + chunk_size, T)
            if t_ne - t_ns == chunk_size:
                next_act_np = actions_ep[t_ns:t_ne]
            else:
                # Pad last action to fill chunk
                pad = np.repeat(actions_ep[[-1]], chunk_size - (t_ne - t_ns), axis=0)
                next_act_np = np.concatenate([actions_ep[t_ns:t_ne], pad], axis=0)

            store.append(
                rl_token      = rl_tokens_ep[t_start],
                state         = states_ep[t_start],
                action        = act_np,
                next_rl_token = rl_tokens_ep[t_next],
                next_state    = states_ep[t_next],
                next_action   = next_act_np,
                reward        = ep_success if is_last else 0.0,
                done          = float(is_last),
            )

        # Free episode buffers
        del rl_tokens_ep, states_ep, actions_ep

        if (prog_i + 1) % 10 == 0 or prog_i == len(ep_info) - 1:
            elapsed = time.perf_counter() - t0
            spd = (prog_i + 1) / elapsed
            eta = (len(ep_info) - prog_i - 1) / spd if spd > 0 else 0
            logging.info(
                f"  ep {prog_i + 1}/{len(ep_info)} "
                f"(ep_idx={ep_idx}, T={T}) | "
                f"written={store._size} | "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
            )

    actual = store.flush_and_close()
    logging.info(
        f"[CriticPretrain] Extraction done: {actual} transitions, "
        f"{time.perf_counter() - t0:.1f}s total."
    )
    return actual


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


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
    chunk_size: int,
) -> dict:
    """TD3-style double-Q Bellman update with BC next-action proxy."""
    critic.train()

    rl_token      = batch["rl_token"].to(device)
    state         = batch["state"].to(device)
    action        = batch["action"].to(device)
    reward        = batch["reward"].to(device)
    next_rl_token = batch["next_rl_token"].to(device)
    next_state    = batch["next_state"].to(device)
    next_action   = batch["next_action"].to(device)
    done          = batch["done"].to(device)

    with torch.no_grad():
        target_q = critic.target_q_min(next_rl_token, next_state, next_action).squeeze(-1)
        q_hat    = reward + (gamma ** chunk_size) * (1.0 - done) * target_q

    q1, q2   = critic(rl_token, state, action)
    q1, q2   = q1.squeeze(-1), q2.squeeze(-1)
    loss_q1  = F.mse_loss(q1, q_hat)
    loss_q2  = F.mse_loss(q2, q_hat)
    loss     = loss_q1 + loss_q2

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
# Main
# ---------------------------------------------------------------------------

@parser.wrap()
def pretrain_critic(cfg: CriticPretrainConfig):
    init_logging()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_dir = output_dir / "mmap_store"

    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"[CriticPretrain] device={device}")

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
            name=f"pretrain_critic_{output_dir.name}",
            config={k: getattr(cfg, k) for k in (
                "chunk_size", "steps", "batch_size", "lr",
                "gamma", "tau", "critic_hidden_dim", "critic_num_layers",
                "vla_checkpoint", "dataset_repo_id",
            )},
        )
        logging.info(f"[CriticPretrain] WandB: {wandb_run.url}")

    # ------------------------------------------------------------------
    # Load VLA (frozen, eval)
    # ------------------------------------------------------------------
    logging.info(f"[CriticPretrain] Loading VLA: {cfg.vla_checkpoint}")
    vla_policy = PI05RLTPolicy.from_pretrained(cfg.vla_checkpoint)
    vla_policy.to(device).eval()
    for p in vla_policy.parameters():
        p.requires_grad = False
    rl_token_dim = vla_policy.config.rl_token_dim
    logging.info(f"[CriticPretrain] VLA loaded (frozen). rl_token_dim={rl_token_dim}")

    # ------------------------------------------------------------------
    # Load dataset (metadata only)
    # ------------------------------------------------------------------
    logging.info(f"[CriticPretrain] Loading dataset: {cfg.dataset_repo_id}")
    ds_kwargs: dict = {"repo_id": cfg.dataset_repo_id}
    if cfg.dataset_root:
        ds_kwargs["root"] = cfg.dataset_root
    dataset = LeRobotDataset(**ds_kwargs)
    logging.info(
        f"[CriticPretrain] Dataset: {dataset.meta.total_episodes} eps, "
        f"{dataset.meta.total_frames} frames, fps={dataset.fps}"
    )

    # Infer dims from first frame, then immediately free it
    first_frame = dataset[0]
    state_dim   = _extract_state(first_frame).shape[0]
    action_dim  = _extract_action(first_frame).shape[0]
    del first_frame
    logging.info(
        f"[CriticPretrain] state_dim={state_dim}, action_dim={action_dim}, "
        f"chunk_size={cfg.chunk_size}"
    )

    # ------------------------------------------------------------------
    # Build mmap store (streaming, low memory)
    # ------------------------------------------------------------------
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
            f"No transitions built. All episodes shorter than "
            f"chunk_size+1={cfg.chunk_size + 1} frames."
        )

    # ------------------------------------------------------------------
    # Re-open mmap read-only for training
    # ------------------------------------------------------------------
    store = MmapTransitionStore.open_readonly(
        store_dir=store_dir,
        size=actual_size,
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=cfg.chunk_size,
    )

    reward_arr = store._mmaps["reward"]
    n_pos = int((reward_arr[:actual_size] > 0).sum())
    logging.info(
        f"[CriticPretrain] Reward balance: {n_pos}/{actual_size} positive "
        f"({100 * n_pos / actual_size:.1f}%)"
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

    n_total  = sum(p.numel() for p in critic.parameters())
    n_online = sum(p.numel() for p in critic.get_online_parameters())
    logging.info(f"[CriticPretrain] Critic: total={n_total:,} params | online={n_online:,}")

    optimizer = torch.optim.AdamW(
        critic.get_online_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.01
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logging.info(
        f"[CriticPretrain] Training {cfg.steps} steps | "
        f"batch={cfg.batch_size} lr={cfg.lr} gamma={cfg.gamma} tau={cfg.tau}"
    )
    step      = 0
    best_loss = float("inf")
    data_iter = iter(dataloader)

    while step < cfg.steps:
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
            chunk_size=cfg.chunk_size,
        )
        critic.soft_update(tau=cfg.tau)
        scheduler.step()
        step += 1

        if step % cfg.log_freq == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logging.info(
                f"  Step {step:6d}/{cfg.steps} | "
                f"loss={metrics['critic_loss']:.4f} "
                f"(Q1={metrics['critic_loss_q1']:.4f} Q2={metrics['critic_loss_q2']:.4f}) | "
                f"Q1={metrics['q1_mean']:.3f} Qt={metrics['q_target_mean']:.3f} | "
                f"lr={lr_now:.2e}"
            )
            if wandb_run:
                wandb_run.log(
                    {f"train/{k}": v for k, v in metrics.items()} | {"train/lr": lr_now},
                    step=step,
                )

        if step % cfg.save_freq == 0 or step == cfg.steps:
            ckpt = {
                "step":      step,
                "critic":    critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "rl_token_dim": rl_token_dim, "state_dim": state_dim,
                    "action_dim": action_dim,     "chunk_size": cfg.chunk_size,
                    "hidden_dim": cfg.critic_hidden_dim,
                    "num_layers": cfg.critic_num_layers,
                },
            }
            p = output_dir / f"critic_step_{step:06d}.pth"
            torch.save(ckpt, p)
            logging.info(f"[CriticPretrain] Saved: {p}")
            if metrics["critic_loss"] < best_loss:
                best_loss = metrics["critic_loss"]
                torch.save(ckpt, output_dir / "critic_best.pth")
                logging.info(f"[CriticPretrain] New best (loss={best_loss:.4f})")

    torch.save(
        {
            "step":   step,
            "critic": critic.state_dict(),
            "config": {
                "rl_token_dim": rl_token_dim, "state_dim": state_dim,
                "action_dim": action_dim,     "chunk_size": cfg.chunk_size,
                "hidden_dim": cfg.critic_hidden_dim,
                "num_layers": cfg.critic_num_layers,
            },
        },
        output_dir / "critic_final.pth",
    )
    logging.info(
        f"[CriticPretrain] Done. Final={output_dir / 'critic_final.pth'} | "
        f"best_loss={best_loss:.4f}"
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
