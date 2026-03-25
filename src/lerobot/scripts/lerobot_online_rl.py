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
Online RL Training Script for RL Token (RLT).

Implements the full Online RL loop described in the RLT paper:
  - Sparse +1 reward: human presses SPACE after episode success
  - Three action sources: warmup (VLA), rollout (actor), intervention (human teleop)
  - RL phase toggle: press 'r' to enter / exit the RL-needed segment of an episode
  - Subsampled replay buffer: every `subsample_stride` chunks in the RL phase
  - Mode-switch data recorded every frame with label 0/1
  - Update-to-data ratio G=5, async updates within the same process
  - Unified data directory:  data_path/{meta.json, play_buffer.pkl, mode_switch.pkl}
  - Teleoperator mirrors robot motion (policy_sync) so intervention is always ready
  - Full dual-arm (bimanual) support

Key controls:
  SPACE    -- SUCCESS: end episode, reward = +1, reset arms
  f        -- FAILURE: end episode, reward = 0, reset arms
  i        -- toggle INTERVENTION (human teleop overrides robot)
  r        -- toggle RL PHASE (enter/exit the segment needing RL)
  q        -- quit

Usage
-----
lerobot-online-rl \\
    --vla_checkpoint=outputs/rlt_joint/last_checkpoint \\
    --data_path=outputs/online_rl/data \\
    --output_dir=outputs/online_rl \\
    --robot.type=so101_follower \\
    --robot.port=/dev/ttyUSB0 \\
    --robot.cameras="{front:{type:opencv,index_or_path:0,width:640,height:480,fps:30}}" \\
    --teleop.type=so101_leader \\
    --teleop.port=/dev/ttyUSB1 \\
    --task="pick and place the red cube"
"""

import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.rl.rlt_actor_critic import DoubleCritic, GaussianActor, RLTActorCritic
from lerobot.rl.rlt_buffer import RLTDataManager, RLTReplayBuffer, RLTTransition
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    make_robot_from_config,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so_leader,
    gamepad,
    make_teleoperator_from_config,
    so_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OnlineRLConfig:
    """Configuration for RLT Online RL training."""

    robot: RobotConfig
    teleop: TeleoperatorConfig

    # Paths
    vla_checkpoint: str = "outputs/rlt_joint/last_checkpoint"
    actor_critic_checkpoint: Optional[str] = None
    # Unified data directory (contains meta.json, play_buffer.pkl, mode_switch.pkl)
    data_path: str = "outputs/online_rl/data"
    output_dir: str = "outputs/online_rl"

    # Task instruction
    task: str = "perform the manipulation task"

    # RL hyperparameters (defaults from paper)
    chunk_size_rl: int = 10           # C=10 per paper
    warmup_steps: int = 500           # N_warmup: pure VLA steps before RL starts
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 0.5                 # actor reference regularization
    ref_dropout: float = 0.5
    actor_sigma: float = 0.1
    rl_batch_size: int = 256
    updates_per_step: int = 5         # G: update-to-data ratio
    buffer_capacity: int = 100_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Subsampling: record a buffer transition every `subsample_stride` chunks
    # e.g. stride=2 means we record chunks at t=0,2,4,... (in RL phase)
    subsample_stride: int = 2

    # Actor/Critic MLP architecture
    actor_hidden_dim: int = 512
    actor_num_layers: int = 3
    critic_hidden_dim: int = 512
    critic_num_layers: int = 3

    # Mode switch data collection
    save_mode_switch_data: bool = False

    # Control
    fps: int = 10                     # robot control frequency
    max_episodes: int = 1000
    episode_timeout_s: float = 30.0   # max seconds per episode
    reset_duration_s: float = 3.0     # seconds to interpolate back to reset pose

    # Save
    save_freq_episodes: int = 10      # save data every N episodes
    log_freq: int = 10

    # WandB
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Debug
    debug: bool = False               # print extra debug info each step

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Keyboard listener (non-blocking)
# ---------------------------------------------------------------------------

class KeyboardController:
    """
    Non-blocking keyboard listener for online RL control.

    Keys:
      SPACE -- SUCCESS: end episode with reward +1
      f     -- FAILURE: end episode with reward 0
      i     -- toggle INTERVENTION mode (human teleop overrides robot)
      r     -- toggle RL PHASE (enter/exit the segment that needs RL)
      q     -- quit
    """

    def __init__(self):
        self._success = threading.Event()
        self._failure = threading.Event()
        self._intervention = threading.Event()
        self._rl_phase = threading.Event()   # NEW: enter/exit RL phase within episode
        self._quit = threading.Event()
        self._thread = None

    def start(self):
        try:
            from pynput import keyboard as kb

            def on_press(key):
                try:
                    if key == kb.Key.space:
                        self._success.set()
                        logging.info("[KEY] SUCCESS signal received (SPACE)")
                    elif hasattr(key, "char") and key.char == "f":
                        self._failure.set()
                        logging.info("[KEY] FAILURE signal received (f)")
                    elif hasattr(key, "char") and key.char == "i":
                        if self._intervention.is_set():
                            self._intervention.clear()
                            logging.info("[KEY] Intervention DISABLED")
                        else:
                            self._intervention.set()
                            logging.info("[KEY] Intervention ENABLED")
                    elif hasattr(key, "char") and key.char == "r":
                        if self._rl_phase.is_set():
                            self._rl_phase.clear()
                            logging.info("[KEY] RL Phase EXIT")
                        else:
                            self._rl_phase.set()
                            logging.info("[KEY] RL Phase ENTER")
                    elif hasattr(key, "char") and key.char == "q":
                        self._quit.set()
                        logging.info("[KEY] Quit signal received")
                except Exception:
                    pass

            listener = kb.Listener(on_press=on_press)
            self._thread = listener
            listener.start()
        except ImportError:
            logging.warning("[KEY] pynput not available. Keyboard control disabled.")

    def stop(self):
        if self._thread is not None:
            self._thread.stop()

    def pop_success(self) -> bool:
        if self._success.is_set():
            self._success.clear()
            return True
        return False

    def pop_failure(self) -> bool:
        if self._failure.is_set():
            self._failure.clear()
            return True
        return False

    @property
    def is_intervention(self) -> bool:
        return self._intervention.is_set()

    @property
    def is_rl_phase(self) -> bool:
        """True when operator has toggled into the RL-needed phase."""
        return self._rl_phase.is_set()

    @property
    def should_quit(self) -> bool:
        return self._quit.is_set()


# ---------------------------------------------------------------------------
# Helper: mirror robot action back to teleoperator ("policy_sync")
# ---------------------------------------------------------------------------

def _sync_teleop_to_robot(teleop, action_dict: dict):
    """
    Send robot's current action as feedback to the teleoperator so that
    the teleop arm mirrors the robot arm even when not intervening.
    Silently ignores if teleop does not support send_feedback.
    """
    try:
        if hasattr(teleop, "send_feedback"):
            teleop.send_feedback(action_dict)
        elif isinstance(teleop, list):
            for t in teleop:
                if hasattr(t, "send_feedback"):
                    t.send_feedback(action_dict)
    except Exception as e:
        logging.debug(f"[sync_teleop] send_feedback error: {e}")


# ---------------------------------------------------------------------------
# Helper: slow reset arms to initial pose
# ---------------------------------------------------------------------------

def _slow_reset_to_pose(robot, teleop, target_pose: dict, duration_s: float = 3.0):
    """Interpolate robot (and teleop) back to target_pose over duration_s seconds."""
    if not target_pose:
        return
    try:
        current_obs = robot.get_observation()
        joint_keys = [k for k in target_pose if k in current_obs]
        if not joint_keys:
            return

        start = {k: float(current_obs[k]) for k in joint_keys}
        goal  = {k: float(target_pose[k]) for k in joint_keys}

        step_dt = 0.05
        steps = max(int(duration_s / step_dt), 1)
        for i in range(1, steps + 1):
            alpha = i / steps
            action = {k: start[k] + (goal[k] - start[k]) * alpha for k in joint_keys}
            robot.send_action(action)
            _sync_teleop_to_robot(teleop, action)
            time.sleep(step_dt)
        logging.info(f"[OnlineRL] Arms reset to initial pose ({duration_s:.1f}s).")
    except Exception as e:
        logging.warning(f"[OnlineRL] Failed to reset arms: {e}")


# ---------------------------------------------------------------------------
# Actor-Critic update functions
# ---------------------------------------------------------------------------

def compute_target_q(
    batch: dict,
    actor: GaussianActor,
    critic: DoubleCritic,
    gamma: float,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    """
    Compute TD target Q:
      Q_hat = gamma^{C-1} * r + gamma^C * (1-done) * min(Q1', Q2')(x', a')
    where a' ~ pi_theta(x', zero_ref)  [zero_ref: ref_dropout handles masking]
    """
    rl_token_next = batch["next_rl_token"].to(device)
    state_next    = batch["next_state"].to(device)
    reward        = batch["reward"].to(device)
    done          = batch["done"].to(device)

    with torch.no_grad():
        B = rl_token_next.shape[0]
        action_dim = batch["action"].shape[-1]
        zero_ref = torch.zeros(B, chunk_size, action_dim, device=device)
        next_actions, _, _ = actor.sample(rl_token_next, state_next, zero_ref)

        q_next   = critic.target_q_min(rl_token_next, state_next, next_actions).squeeze(-1)
        q_target = reward * (gamma ** (chunk_size - 1)) + (1.0 - done) * (gamma ** chunk_size) * q_next

    return q_target.detach()


def update_critic(
    batch: dict,
    critic: DoubleCritic,
    critic_optimizer: torch.optim.Optimizer,
    actor: GaussianActor,
    gamma: float,
    chunk_size: int,
    device: str,
) -> dict:
    """Update both Q networks. Returns loss dict."""
    rl_token = batch["rl_token"].to(device)
    state    = batch["state"].to(device)
    action   = batch["action"].to(device)

    q_target = compute_target_q(batch, actor, critic, gamma, chunk_size, device)

    q1, q2 = critic(rl_token, state, action)
    q1 = q1.squeeze(-1)
    q2 = q2.squeeze(-1)

    loss_q1 = F.mse_loss(q1, q_target)
    loss_q2 = F.mse_loss(q2, q_target)
    critic_loss = loss_q1 + loss_q2

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return {
        "critic_loss": critic_loss.item(),
        "critic_loss_q1": loss_q1.item(),
        "critic_loss_q2": loss_q2.item(),
        "q1_mean": q1.mean().item(),
        "q_target_mean": q_target.mean().item(),
    }


def update_actor(
    batch: dict,
    actor: GaussianActor,
    critic: DoubleCritic,
    actor_optimizer: torch.optim.Optimizer,
    beta: float,
    device: str,
) -> dict:
    """
    Update actor:
      L_pi = E[-Q(x,a) + beta * ||a - a_ref||^2]
    Returns loss dict.
    """
    rl_token   = batch["rl_token"].to(device)
    state      = batch["state"].to(device)
    ref_action = batch["ref_action"].to(device)

    actor.training = True
    action, mu, _ = actor.sample(rl_token, state, ref_action)

    q_val   = critic.q_min(rl_token, state, action).squeeze(-1)
    ref_reg = F.mse_loss(action, ref_action)

    loss_q_term   = -q_val.mean()
    loss_ref_term = beta * ref_reg
    actor_loss    = loss_q_term + loss_ref_term

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return {
        "actor_loss":     actor_loss.item(),
        "actor_loss_q":   loss_q_term.item(),
        "actor_loss_ref": loss_ref_term.item(),
        "actor_q_mean":   q_val.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main Online RL loop
# ---------------------------------------------------------------------------

@parser.wrap()
def online_rl(cfg: OnlineRLConfig):
    """Main online RL training loop."""
    init_logging()
    logging.info("[OnlineRL] Config:\n" + pformat(cfg.__dict__))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_manager = RLTDataManager(cfg.data_path)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"[OnlineRL] Using device: {device}")

    # ---------- WandB ----------
    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config=cfg.__dict__,
            )
            logging.info(f"[OnlineRL] WandB run: {wandb_run.name}")
        except Exception as e:
            logging.warning(f"[OnlineRL] WandB init failed: {e}")

    # ---------- Load VLA policy ----------
    logging.info(f"[OnlineRL] Loading VLA checkpoint: {cfg.vla_checkpoint}")
    vla_policy = PI05RLTPolicy.from_pretrained(cfg.vla_checkpoint)
    vla_policy.to(device)
    vla_policy.eval()
    for param in vla_policy.parameters():
        param.requires_grad = False

    rl_token_dim = vla_policy.config.rl_token_dim
    state_dim    = vla_policy.config.max_state_dim
    action_dim   = vla_policy.config.max_action_dim
    chunk_size   = cfg.chunk_size_rl
    logging.info(
        f"[OnlineRL] Dims: rl_token={rl_token_dim}, state={state_dim}, "
        f"action={action_dim}, chunk={chunk_size}"
    )

    # ---------- Actor-Critic ----------
    actor_critic = RLTActorCritic(
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        actor_hidden_dim=cfg.actor_hidden_dim,
        actor_num_layers=cfg.actor_num_layers,
        critic_hidden_dim=cfg.critic_hidden_dim,
        critic_num_layers=cfg.critic_num_layers,
        sigma=cfg.actor_sigma,
        ref_dropout=cfg.ref_dropout,
    ).to(device)

    ac_step = 0
    if cfg.actor_critic_checkpoint and Path(cfg.actor_critic_checkpoint).exists():
        logging.info(f"[OnlineRL] Loading actor-critic from {cfg.actor_critic_checkpoint}")
        ckpt = torch.load(cfg.actor_critic_checkpoint, map_location=device)
        actor_critic.load_state_dict(ckpt["actor_critic"])
        ac_step = ckpt.get("step", 0)

    actor_optimizer  = torch.optim.Adam(actor_critic.actor.parameters(),                   lr=cfg.actor_lr)
    critic_optimizer = torch.optim.Adam(actor_critic.critic.get_online_parameters(), lr=cfg.critic_lr)

    # ---------- Replay Buffer ----------
    existing_buf = data_manager.load_buffer(device=device)
    if existing_buf is not None:
        replay_buffer = existing_buf
        logging.info(f"[OnlineRL] Loaded replay buffer ({len(replay_buffer)} transitions)")
    else:
        replay_buffer = RLTReplayBuffer(capacity=cfg.buffer_capacity, device=device, storage_device="cpu")

    # Load existing mode-switch data
    mode_switch_list: list[dict] = data_manager.load_mode_switch() if cfg.save_mode_switch_data else []

    # ---------- Robot & Teleop ----------
    robot  = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot.connect()
    teleop.connect()
    logging.info("[OnlineRL] Robot and teleop connected.")

    # Capture initial (reset) pose
    init_obs  = robot.get_observation()
    init_pose = {k: float(v) for k, v in init_obs.items() if str(k).endswith(".pos")}
    logging.info(f"[OnlineRL] Captured reset pose ({len(init_pose)} joints).")

    # ---------- Keyboard ----------
    keyboard = KeyboardController()
    keyboard.start()

    # ---------- Statistics ----------
    total_env_steps = ac_step
    total_episodes  = 0
    total_updates   = 0
    episode_rewards: list[float] = []

    logging.info("[OnlineRL] Controls:")
    logging.info("  SPACE = episode SUCCESS (reward +1)")
    logging.info("  f     = episode FAILURE (reward 0)")
    logging.info("  i     = toggle INTERVENTION")
    logging.info("  r     = toggle RL PHASE")
    logging.info("  q     = quit")

    # ---------- Save helper ----------
    def _save_all(tag: str = ""):
        data_manager.save_buffer(replay_buffer)
        if cfg.save_mode_switch_data:
            data_manager.save_mode_switch(mode_switch_list)
        meta = RLTDataManager.build_meta(
            replay_buffer,
            robot_type=getattr(cfg.robot, "type", "unknown"),
            mode_switch_count=len(mode_switch_list),
            extra={
                "total_episodes": total_episodes,
                "total_env_steps": total_env_steps,
                "total_updates": total_updates,
            },
        )
        data_manager.save_meta(meta)
        ckpt_path = ckpt_dir / f"actor_critic{tag}.pth"
        torch.save({"actor_critic": actor_critic.state_dict(), "step": total_env_steps}, ckpt_path)
        logging.info(f"[OnlineRL] Saved checkpoint at step {total_env_steps} → {ckpt_path}")

    # Graceful Ctrl+C
    _quit = [False]
    def _sigint(_s, _f):
        _quit[0] = True
        logging.info("[OnlineRL] Ctrl+C — finishing current episode …")
    signal.signal(signal.SIGINT, _sigint)

    try:
        for ep_idx in range(cfg.max_episodes):
            if _quit[0] or keyboard.should_quit:
                break

            logging.info(
                f"\n[OnlineRL] === Episode {ep_idx + 1}/{cfg.max_episodes} | "
                f"buffer={len(replay_buffer)} | updates={total_updates} | "
                f"env_steps={total_env_steps} ==="
            )

            episode_reward = 0.0
            episode_done   = False
            ep_start       = time.perf_counter()

            vla_policy.reset()

            prev_rl_token: torch.Tensor | None = None
            prev_state_t:  torch.Tensor | None = None
            prev_was_rl_phase = False

            chunk_idx       = 0   # chunk counter within episode
            rl_chunk_idx    = 0   # chunk counter within current RL phase (for subsampling)

            while not episode_done:
                if _quit[0] or keyboard.should_quit:
                    episode_done = True
                    break

                loop_start = time.perf_counter()

                # ---- Observation ----
                obs       = robot.get_observation()
                batch_obs = _obs_to_batch(obs, cfg.task, vla_policy, device)

                # ---- VLA inference ----
                with torch.no_grad():
                    vla_action_chunk = _predict_action_chunk(vla_policy, batch_obs)  # (1,C,D)
                    rl_token         = vla_policy.extract_rl_token(batch_obs)         # (1,T)

                proprioceptive = _get_proprioceptive_state(obs, vla_policy, device)   # (1,S)
                rl_tok, prop_state = rl_token.squeeze(0), proprioceptive.squeeze(0)
                ref_action = vla_action_chunk.squeeze(0)   # (C, D)

                # ---- Mode selection ----
                is_intervention = keyboard.is_intervention
                is_warmup       = total_env_steps < cfg.warmup_steps
                is_rl_phase     = keyboard.is_rl_phase and not is_warmup

                if is_intervention:
                    executed_action = _get_teleop_action(teleop, robot, chunk_size, action_dim, device)
                    ref_override    = executed_action.clone()
                    mode_label      = "intervention"
                elif is_warmup:
                    executed_action = ref_action.clone()
                    ref_override    = ref_action.clone()
                    mode_label      = "warmup"
                elif is_rl_phase:
                    with torch.no_grad():
                        actor_critic.actor.training = False
                        actor_action, _, _ = actor_critic.actor.sample(
                            rl_tok.unsqueeze(0), prop_state.unsqueeze(0), ref_action.unsqueeze(0)
                        )
                    executed_action = actor_action.squeeze(0)
                    ref_override    = ref_action.clone()
                    mode_label      = "actor"
                else:
                    # VLA-only phase (outside RL phase)
                    executed_action = ref_action.clone()
                    ref_override    = ref_action.clone()
                    mode_label      = "vla"

                # ---- Execute action + mirror to teleop ----
                action_dict = _execute_action_chunk(robot, executed_action, cfg.fps)
                _sync_teleop_to_robot(teleop, action_dict)

                # ---- Check episode termination ----
                step_reward = 0.0
                if keyboard.pop_success():
                    step_reward  = 1.0
                    episode_done = True
                    logging.info("[OnlineRL] SUCCESS → reward=+1")
                elif keyboard.pop_failure():
                    episode_done = True
                    logging.info("[OnlineRL] FAILURE → reward=0")
                elif time.perf_counter() - ep_start > cfg.episode_timeout_s:
                    episode_done = True
                    logging.info("[OnlineRL] Timeout → reward=0")

                episode_reward  += step_reward
                total_env_steps += chunk_size

                # ---- Mode-switch data: record every frame ----
                if cfg.save_mode_switch_data:
                    ms_label = 1 if is_rl_phase else 0
                    replay_buffer.add_mode_switch_sample(
                        rl_tok.cpu(), prop_state.cpu(), ref_override.cpu(), ms_label
                    )
                    mode_switch_list = replay_buffer.get_mode_switch_data()

                # ---- Replay buffer: subsampled, RL phase only ----
                if prev_rl_token is not None and is_rl_phase:
                    if rl_chunk_idx % cfg.subsample_stride == 0:
                        transition = RLTTransition(
                            rl_token      = prev_rl_token,
                            state         = prev_state_t,
                            action        = executed_action.cpu(),
                            ref_action    = ref_override.cpu(),
                            reward        = step_reward,
                            next_rl_token = rl_tok.cpu(),
                            next_state    = prop_state.cpu(),
                            done          = float(episode_done),
                        )
                        replay_buffer.add(transition)

                # Track rl_chunk_idx (reset when phase changes)
                if is_rl_phase and not prev_was_rl_phase:
                    rl_chunk_idx = 0
                if is_rl_phase:
                    rl_chunk_idx += 1

                prev_rl_token     = rl_tok.cpu()
                prev_state_t      = prop_state.cpu()
                prev_was_rl_phase = is_rl_phase

                # ---- Online updates ----
                update_logs: dict = {}
                if len(replay_buffer) >= cfg.rl_batch_size and is_rl_phase:
                    for _ in range(cfg.updates_per_step):
                        batch = replay_buffer.sample(cfg.rl_batch_size)
                        c_log = update_critic(batch, actor_critic.critic, critic_optimizer,
                                              actor_critic.actor, cfg.gamma, chunk_size, device)
                        a_log = update_actor(batch, actor_critic.actor, actor_critic.critic,
                                             actor_optimizer, cfg.beta, device)
                        actor_critic.soft_update_targets(cfg.tau)
                        total_updates += 1
                        update_logs = {**c_log, **a_log}

                # ---- Logging ----
                should_log = (chunk_idx % cfg.log_freq == 0)
                if should_log or cfg.debug:
                    log_msg = (
                        f"  step={total_env_steps} mode={mode_label} "
                        f"buffer={len(replay_buffer)} updates={total_updates}"
                    )
                    if update_logs:
                        log_msg += (
                            f" | critic={update_logs.get('critic_loss', 0):.4f}"
                            f" actor={update_logs.get('actor_loss', 0):.4f}"
                            f" q_target={update_logs.get('q_target_mean', 0):.3f}"
                        )
                    logging.info(log_msg)

                if wandb_run and update_logs:
                    wandb_run.log(
                        {
                            "env/step": total_env_steps,
                            "env/mode": mode_label,
                            "buffer/size": len(replay_buffer),
                            **{f"train/{k}": v for k, v in update_logs.items()},
                        },
                        step=total_env_steps,
                    )

                # ---- Timing ----
                dt = time.perf_counter() - loop_start
                precise_sleep(max(chunk_size / cfg.fps - dt, 0.0))
                chunk_idx += 1

            # ---- End of episode ----
            episode_rewards.append(episode_reward)
            total_episodes += 1
            recent_avg = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            logging.info(
                f"[OnlineRL] Episode {ep_idx + 1} done | "
                f"reward={episode_reward:.1f} | avg10={recent_avg:.2f}"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "episode/reward": episode_reward,
                        "episode/avg10_reward": recent_avg,
                        "episode/total": total_episodes,
                    },
                    step=total_env_steps,
                )

            # Reset arms
            _slow_reset_to_pose(robot, teleop, init_pose, cfg.reset_duration_s)

            if total_episodes % cfg.save_freq_episodes == 0:
                _save_all(tag=f"_ep{total_episodes}")

    finally:
        keyboard.stop()
        logging.info("[OnlineRL] Saving final checkpoint …")
        _save_all(tag="_final")
        if wandb_run:
            wandb_run.finish()
        robot.disconnect()
        teleop.disconnect()
        logging.info("[OnlineRL] Cleanup complete.")


# ---------------------------------------------------------------------------
# Robot interface helpers
# ---------------------------------------------------------------------------

def _predict_action_chunk(policy: PI05RLTPolicy, batch: dict) -> torch.Tensor:
    """Run VLA policy to get action chunk (1, C, action_dim)."""
    with torch.no_grad():
        return policy.select_action(batch).unsqueeze(0)  # (1, C, D)


def _obs_to_batch(obs: dict, task: str, policy: PI05RLTPolicy, device: str) -> dict:
    """
    Convert raw robot observation dict to a model-compatible batch dict.
    Copies tensors to device; adds dummy language tokens if needed.
    In practice this should use the policy's real tokenizer/processor.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    batch: dict = {}
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device)

    if OBS_STATE in obs:
        s = obs[OBS_STATE]
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        batch[OBS_STATE] = s.unsqueeze(0).to(device)

    tokenizer_max_len = getattr(policy.config, "tokenizer_max_length", 48)
    batch[OBS_LANGUAGE_TOKENS]         = torch.zeros(1, tokenizer_max_len, dtype=torch.long,  device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros(1, tokenizer_max_len, dtype=torch.bool,  device=device)
    return batch


def _get_proprioceptive_state(obs: dict, policy: PI05RLTPolicy, device: str) -> torch.Tensor:
    """Extract and pad proprioceptive state → (1, max_state_dim)."""
    from lerobot.policies.pi05.modeling_pi05 import pad_vector
    from lerobot.utils.constants import OBS_STATE

    s = obs.get(OBS_STATE)
    if s is None:
        return torch.zeros(1, policy.config.max_state_dim, device=device)
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, dtype=torch.float32)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    return pad_vector(s.to(device), policy.config.max_state_dim)


def _get_teleop_action(teleop, robot, chunk_size: int, action_dim: int, device: str) -> torch.Tensor:
    """Get a (C, action_dim) action tensor from the teleoperator."""
    try:
        raw = teleop.get_action()
        vals = list(raw.values()) if isinstance(raw, dict) else list(raw)
        t = torch.tensor(vals, dtype=torch.float32, device=device)
        if t.shape[0] < action_dim:
            t = F.pad(t, (0, action_dim - t.shape[0]))
        else:
            t = t[:action_dim]
        return t.unsqueeze(0).expand(chunk_size, -1)   # (C, D)
    except Exception as e:
        logging.warning(f"[OnlineRL] teleop.get_action() error: {e}")
        return torch.zeros(chunk_size, action_dim, device=device)


def _make_rl_state(
    rl_token: torch.Tensor,
    proprioceptive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper for deployment: extract (rl_tok, prop_state) both squeezed to (D,).

    Args:
        rl_token:       (1, rl_token_dim) tensor from ``policy.extract_rl_token``
        proprioceptive: (1, state_dim)    tensor from ``_get_proprioceptive_state``

    Returns:
        Tuple of 1-D tensors (rl_tok, prop_state).
    """
    rl_tok   = rl_token.squeeze(0)       # (rl_token_dim,)
    prop_state = proprioceptive.squeeze(0)  # (state_dim,)
    return rl_tok, prop_state


def _execute_action_chunk(robot, action_chunk: torch.Tensor, fps: int) -> dict:
    """
    Execute each step of an action chunk on the robot at `fps` Hz.
    Returns the last action dict sent (for teleop feedback).
    """
    C = action_chunk.shape[0]
    last_action: dict = {}
    for t in range(C):
        action = action_chunk[t]
        action_dict = {f"joint_{i}": float(action[i]) for i in range(len(action))}
        try:
            robot.send_action(action_dict)
            last_action = action_dict
        except Exception as e:
            logging.warning(f"[OnlineRL] robot.send_action error at t={t}: {e}")
        precise_sleep(1.0 / fps)
    return last_action


def main():
    register_third_party_plugins()
    online_rl()


if __name__ == "__main__":
    main()
