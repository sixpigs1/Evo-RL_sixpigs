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
Mode Switch Network Training Script.

Trains a binary MLP classifier to predict whether the robot should use:
  - Mode 0: VLA action (a_ref)
  - Mode 1: Actor-corrected action (from pi_theta)

Training data is collected during online RL (``--save_mode_switch_data=true``).
The data lives under the unified data directory::

    data_path/
        mode_switch.pkl   ← loaded by this script
        meta.json
        play_buffer.pkl

Input to classifier:  (x_t, a_ref) = (z_rl, s^p, a_ref_{1:C})
Label:  1 = robot is in RL phase at this step, 0 = VLA / warmup phase

Usage:
------
lerobot-train-mode-switch \\
    --data_path=outputs/online_rl/data \\
    --output_dir=outputs/mode_switch \\
    --rl_token_dim=2048 \\
    --state_dim=14 \\
    --action_dim=7 \\
    --chunk_size_rl=10
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset, random_split

from lerobot.configs import parser
from lerobot.rl.rlt_actor_critic import ModeSwitchMLP
from lerobot.rl.rlt_buffer import RLTDataManager
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


# ---------------------------------------------------------------------------
# Dataset wrapper for Mode Switch data
# ---------------------------------------------------------------------------

class ModeSwitchDataset(Dataset):
    """
    Dataset for Mode Switch Network training.

    Loads mode switch data collected during online RL:
    Each sample: (rl_token, state, ref_action, label)
    """

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.data[idx]
        rl_token = sample["rl_token"]
        state = sample["state"]
        ref_action = sample["ref_action"]
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return rl_token, state, ref_action, label


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModeSwitchTrainConfig:
    """Configuration for Mode Switch Network training."""

    # Data
    data_path: str = "outputs/online_rl/data"
    """Unified data directory (produced by lerobot-online-rl).
    Must contain ``mode_switch.pkl`` (and optionally ``meta.json``)."""

    # Model dimensions (must match the VLA/Actor-Critic dims)
    rl_token_dim: int = 2048
    state_dim: int = 32
    action_dim: int = 32
    chunk_size_rl: int = 10

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 2

    # Training
    steps: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.1

    # Output
    output_dir: str = "outputs/mode_switch"
    save_freq: int = 500
    log_freq: int = 50

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@parser.wrap()
def train_mode_switch(cfg: ModeSwitchTrainConfig):
    """Train the Mode Switch MLP classifier."""
    init_logging()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"[ModeSwitchTrain] Using device: {device}")

    # ------- Load data -------
    data_manager = RLTDataManager(cfg.data_path)
    ms_file = data_manager.path / data_manager.MODE_SWITCH_FILE
    if not ms_file.exists():
        raise FileNotFoundError(
            f"Mode switch data not found at {ms_file}. "
            "Please run online RL with save_mode_switch_data=true first."
        )

    logging.info(f"[ModeSwitchTrain] Loading data from {ms_file}")
    data = data_manager.load_mode_switch()
    logging.info(f"[ModeSwitchTrain] Loaded {len(data)} samples")

    if len(data) == 0:
        raise ValueError("No mode switch training data found.")

    # Count class balance
    labels = [d["label"] for d in data]
    pos_ratio = sum(labels) / len(labels)
    logging.info(f"[ModeSwitchTrain] Class balance: {pos_ratio:.2%} positive (mode=1 / RL active)")

    # ------- Dataset split -------
    full_dataset = ModeSwitchDataset(data)
    val_size = max(1, int(len(full_dataset) * cfg.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ------- Model -------
    model = ModeSwitchMLP(
        rl_token_dim=cfg.rl_token_dim,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        chunk_size=cfg.chunk_size_rl,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    logging.info(
        f"[ModeSwitchTrain] Model params: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ------- Optimizer -------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps)

    # Use weighted BCE loss to handle class imbalance
    pos_weight = torch.tensor([(1 - pos_ratio) / max(pos_ratio, 1e-6)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ------- Training loop -------
    step = 0
    train_iter = iter(train_loader)
    best_val_acc = 0.0

    logging.info(f"[ModeSwitchTrain] Starting training for {cfg.steps} steps")

    while step < cfg.steps:
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rl_token, state, ref_action, labels_batch = batch
        rl_token = rl_token.to(device)
        state = state.to(device)
        ref_action = ref_action.to(device)
        labels_batch = labels_batch.to(device)

        # Forward
        logits = model(rl_token, state, ref_action).squeeze(-1)   # (B,)
        loss = loss_fn(logits, labels_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        # Logging
        if step % cfg.log_freq == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == labels_batch).float().mean().item()
            logging.info(
                f"  Step {step:5d}/{cfg.steps} | "
                f"loss={loss.item():.4f} | train_acc={acc:.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # Validation
        if step % cfg.save_freq == 0:
            val_acc, val_loss = _evaluate(model, val_loader, loss_fn, device)
            logging.info(
                f"  [Val] step={step} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "step": step,
                        "mode_switch": model.state_dict(),
                        "config": cfg.__dict__,
                        "val_acc": val_acc,
                    },
                    output_dir / "mode_switch_best.pth",
                )
                logging.info(f"  [Val] New best model saved (val_acc={val_acc:.3f})")

    # Save final model
    torch.save(
        {
            "step": step,
            "mode_switch": model.state_dict(),
            "config": cfg.__dict__,
        },
        output_dir / "mode_switch_final.pth",
    )
    logging.info(f"[ModeSwitchTrain] Training complete. Best val_acc={best_val_acc:.3f}")
    logging.info(f"[ModeSwitchTrain] Models saved to {output_dir}")


def _evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    """Evaluate model on a dataloader. Returns (accuracy, loss)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for rl_token, state, ref_action, labels_batch in loader:
            rl_token = rl_token.to(device)
            state = state.to(device)
            ref_action = ref_action.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(rl_token, state, ref_action).squeeze(-1)
            loss = loss_fn(logits, labels_batch)

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels_batch).float().sum().item()
            total_loss += loss.item() * len(labels_batch)
            total_samples += len(labels_batch)

    return total_correct / total_samples, total_loss / total_samples


def main():
    register_third_party_plugins()
    train_mode_switch()


if __name__ == "__main__":
    main()
