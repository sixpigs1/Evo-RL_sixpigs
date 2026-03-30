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
Joint training script for PI05-RLT:
  VLA finetuning (L_vla) + RL Token encoder-decoder training (L_ro)

  L_total = L_ro(phi) + alpha * L_vla(theta_vla)

  where:
    phi:       RL Token encoder/decoder parameters
    theta_vla: PI0.5 VLA parameters

Usage examples
--------------

# 1. Joint training (VLA + RL Token) from a pretrained PI0.5 checkpoint:
python -m lerobot.scripts.lerobot_train_rlt \\
    --policy.type=pi05_rlt \\
    --policy.pretrained_path=<hf_repo_or_local_dir> \\
    --dataset.repo_id=<dataset_repo_id> \\
    --output_dir=outputs/rlt_joint \\
    --steps=5000 \\
    --batch_size=8

# 2. Freeze VLA, only train RL Token encoder-decoder (alpha=0):
python -m lerobot.scripts.lerobot_train_rlt \\
    --policy.type=pi05_rlt \\
    --policy.pretrained_path=<hf_repo_or_local_dir> \\
    --rlt_alpha=0 \\
    --freeze_vla \\
    --dataset.repo_id=<dataset_repo_id> \\
    --output_dir=outputs/rlt_only \\
    --steps=5000

# 3. Freeze RL Token, only fine-tune VLA (standard PI0.5 training):
python -m lerobot.scripts.lerobot_train_rlt \\
    --policy.type=pi05_rlt \\
    --policy.pretrained_path=<hf_repo_or_local_dir> \\
    --freeze_rlt \\
    --dataset.repo_id=<dataset_repo_id> \\
    --output_dir=outputs/vla_only \\
    --steps=5000
"""

import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    init_logging,
)


def _freeze_vla(policy: PI05RLTPolicy):
    """Freeze all VLA parameters (PaliGemma + action expert + projections)."""
    model = policy.model
    # Freeze the entire PI05Pytorch backbone (action expert, attention layers, etc.)
    for name, param in model.named_parameters():
        if not name.startswith("rlt_encoder") and not name.startswith("rlt_decoder"):
            param.requires_grad = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    logging.info(f"[RLT] Froze VLA parameters ({n_frozen} tensors). Only RL Token trained.")


def _freeze_rlt(policy: PI05RLTPolicy):
    """Freeze RL Token encoder+decoder parameters."""
    model = policy.model
    for name, param in model.named_parameters():
        if name.startswith("rlt_encoder") or name.startswith("rlt_decoder"):
            param.requires_grad = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    logging.info(f"[RLT] Froze RL Token encoder/decoder ({n_frozen} tensors). Only VLA trained.")


def update_policy_rlt(
    train_metrics: MetricsTracker,
    policy: PI05RLTPolicy,
    batch,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    """
    Single training step for joint RLT training.
    Computes L_total = L_ro + alpha * L_vla.

    Returns (updated metrics, output_dict) where output_dict contains:
      loss, loss_vla, loss_recon  -- for logging both sub-losses and total.
    """
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = float(grad_norm)
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train_rlt(cfg: TrainPipelineConfig):
    """
    Main function for PI05-RLT joint training.

    Freeze behaviour is controlled via policy config fields:
      --policy.freeze_vla=true   freeze VLA, only train RL Token encoder/decoder
      --policy.freeze_rlt=true   freeze RL Token, only fine-tune VLA
    """
    # Read freeze flags from policy config (set via --policy.freeze_vla / --policy.freeze_rlt)
    freeze_vla = getattr(cfg.policy, "freeze_vla", False)
    freeze_rlt = getattr(cfg.policy, "freeze_rlt", False)

    cfg.validate()

    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = getattr(cfg.policy, "device", None) == "cpu"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
    )

    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

    if is_main:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True

    # ------- Dataset -------
    if is_main:
        logging.info("Creating dataset for RLT training")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main:
        dataset = make_dataset(cfg)

    # ------- Policy -------
    if is_main:
        logging.info("Creating PI05-RLT policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    if not isinstance(policy, PI05RLTPolicy):
        raise ValueError(
            f"Expected PI05RLTPolicy but got {type(policy).__name__}. "
            "Set --policy.type=pi05_rlt"
        )

    # Apply optional freezing
    unwrapped = accelerator.unwrap_model(policy) if hasattr(accelerator, "unwrap_model") else policy
    if freeze_vla and freeze_rlt:
        raise ValueError("Cannot freeze both VLA and RL Token simultaneously.")
    if freeze_vla:
        _freeze_vla(unwrapped)
    elif freeze_rlt:
        _freeze_rlt(unwrapped)

    # ------- Preprocessors -------
    processor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    # ------- Optimizer -------
    if is_main:
        logging.info("Creating optimizer")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # ------- Dataloader -------
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    if is_main:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"Steps: {cfg.steps} | Learnable params: {format_big_number(num_learnable)} / {format_big_number(num_total)}")
        if freeze_vla:
            logging.info(colored("[RLT] Mode: RL Token only (VLA frozen)", "cyan"))
        elif freeze_rlt:
            logging.info(colored("[RLT] Mode: VLA only (RL Token frozen)", "cyan"))
        else:
            alpha = policy.config.rlt_alpha if not hasattr(accelerator.unwrap_model(policy), "config") else accelerator.unwrap_model(policy).config.rlt_alpha
            logging.info(colored(f"[RLT] Mode: Joint training, alpha={alpha}", "cyan"))

    train_metrics = {
        "loss":          AverageMeter("loss",          ":.4f"),
        "grad_norm":     AverageMeter("grad_norm",     ":.3f"),
        "lr":            AverageMeter("lr",            ":0.1e"),
        "update_s":      AverageMeter("update_s",      ":.3f"),
        "dataloading_s": AverageMeter("dataloading_s", ":.3f"),
    }
    effective_bs = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_bs,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    policy.train()
    if is_main:
        logging.info("[RLT] Starting joint VLA + RL Token training")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy_rlt(
            train_tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, accelerator, lr_scheduler
        )

        step += 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            # Human-readable console log always shows total + sub-losses
            log_str = str(train_tracker)
            if output_dict:
                vla_loss   = output_dict.get("loss_vla",   "n/a")
                recon_loss = output_dict.get("loss_recon", "n/a")
                if isinstance(vla_loss, float):
                    log_str += f"  [loss_vla={vla_loss:.4f}  loss_recon={recon_loss:.4f}]"
            logging.info(log_str)

            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                # Explicitly add sub-losses so they appear as separate wandb charts
                if output_dict:
                    wandb_log_dict["train/loss_vla"]   = output_dict.get("loss_vla",   0.0)
                    wandb_log_dict["train/loss_recon"] = output_dict.get("loss_recon", 0.0)
                    wandb_log_dict["train/loss_total"] = output_dict.get("loss",       0.0)
                    # Per-action-dim flow losses (helpful for debugging action heads)
                    if "loss_per_dim" in output_dict:
                        for dim_i, v in enumerate(output_dict["loss_per_dim"]):
                            wandb_log_dict[f"train/loss_flow_dim{dim_i}"] = v
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step and is_main:
            logging.info(f"[RLT] Checkpoint at step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=accelerator.unwrap_model(policy),
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)

        accelerator.wait_for_everyone()

    if is_main:
        logging.info("[RLT] Training complete.")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train_rlt()


if __name__ == "__main__":
    main()
