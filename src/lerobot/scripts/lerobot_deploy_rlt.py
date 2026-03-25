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
Policy Deployment Script for RL Token (RLT).

After online RL training, this script deploys the combined VLA + Actor-Critic
policy on the robot. It supports two modes for deciding which action to execute:

  MANUAL mode (default):
    - Press 'm' to toggle RL-correction ON/OFF
    - When ON:  execute actor-corrected action
    - When OFF: execute VLA reference action

  AUTO mode (requires trained Mode Switch Network):
    - MLP predicts whether to use VLA or Actor-corrected action
    - VLA inference + Actor inference + MLP inference run in parallel after VLA completes

  Per-step flow:
    1. Get robot observation
    2. Run VLA → get a_ref (reference action chunk) + z_rl (RL token)
    3. Run Actor → get a_actor (corrected action chunk)
    4. Mode decision: manual toggle or MLP prediction
    5. Execute chosen action

Controls:
  m        -- toggle manual RL-correction mode (manual mode only)
  r        -- reset / start a new episode
  q / C-c  -- quit

Usage:
------
# Manual mode (toggle RL correction with 'm'):
python -m lerobot.scripts.lerobot_deploy_rlt \\
    --vla_checkpoint=outputs/rlt_joint/last_checkpoint \\
    --actor_critic_checkpoint=outputs/online_rl/checkpoints/actor_critic_final.pth \\
    --mode=manual \\
    --task="pick and place the red cube" \\
    --robot.type=so101_follower \\
    --robot.port=/dev/ttyUSB0

# Auto mode (MLP decides):
python -m lerobot.scripts.lerobot_deploy_rlt \\
    --vla_checkpoint=outputs/rlt_joint/last_checkpoint \\
    --actor_critic_checkpoint=outputs/online_rl/checkpoints/actor_critic_final.pth \\
    --mode=auto \\
    --mode_switch_checkpoint=outputs/mode_switch/mode_switch_best.pth \\
    --task="pick and place the red cube" \\
    --robot.type=so101_follower \\
    --robot.port=/dev/ttyUSB0
"""

import logging
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.rl.rlt_actor_critic import ModeSwitchMLP, RLTActorCritic
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    bi_so_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

# Reuse helpers from online RL script
from lerobot.scripts.lerobot_online_rl import (
    _obs_to_batch,
    _get_proprioceptive_state,
    _execute_action_chunk,
    _make_rl_state,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DeployRLTConfig:
    """Configuration for RLT policy deployment."""

    robot: RobotConfig

    # Checkpoints
    vla_checkpoint: str = "outputs/rlt_joint/last_checkpoint"
    actor_critic_checkpoint: str = "outputs/online_rl/checkpoints/actor_critic_final.pth"
    mode_switch_checkpoint: Optional[str] = None   # Required for auto mode

    # Mode: manual (keyboard toggle) or auto (MLP decides)
    mode: Literal["manual", "auto"] = "manual"

    # Task
    task: str = "perform the manipulation task"

    # Control
    fps: int = 10
    chunk_size_rl: int = 10

    # Actor inference: deterministic=True for deployment (use mean, no noise)
    deterministic: bool = True

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Keyboard controller for deployment
# ---------------------------------------------------------------------------

class DeployKeyboardController:
    """Keyboard controller for deployment mode."""

    def __init__(self):
        self._rl_active = threading.Event()  # False by default (VLA mode)
        self._reset = threading.Event()
        self._quit = threading.Event()
        self._thread = None

    def start(self):
        try:
            from pynput import keyboard as kb

            def on_press(key):
                try:
                    if hasattr(key, "char"):
                        if key.char == "m":
                            if self._rl_active.is_set():
                                self._rl_active.clear()
                                logging.info("[DEPLOY] Mode: VLA (RL correction OFF)")
                            else:
                                self._rl_active.set()
                                logging.info("[DEPLOY] Mode: Actor-corrected (RL correction ON)")
                        elif key.char == "r":
                            self._reset.set()
                            logging.info("[DEPLOY] Episode reset")
                        elif key.char == "q":
                            self._quit.set()
                            logging.info("[DEPLOY] Quit signal received")
                except Exception:
                    pass

            listener = kb.Listener(on_press=on_press)
            self._thread = listener
            listener.start()
        except ImportError:
            logging.warning("[DEPLOY] pynput not available. Keyboard control disabled.")

    def stop(self):
        if self._thread is not None:
            self._thread.stop()

    @property
    def rl_active(self) -> bool:
        return self._rl_active.is_set()

    def pop_reset(self) -> bool:
        if self._reset.is_set():
            self._reset.clear()
            return True
        return False

    @property
    def should_quit(self) -> bool:
        return self._quit.is_set()


# ---------------------------------------------------------------------------
# Main deployment loop
# ---------------------------------------------------------------------------

@parser.wrap()
def deploy_rlt(cfg: DeployRLTConfig):
    """Deploy the RLT policy on a real robot."""
    init_logging()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"[DeployRLT] Using device: {device}")
    logging.info(f"[DeployRLT] Deployment mode: {cfg.mode}")

    # ------- Load VLA policy -------
    logging.info(f"[DeployRLT] Loading VLA checkpoint: {cfg.vla_checkpoint}")
    vla_policy = PI05RLTPolicy.from_pretrained(cfg.vla_checkpoint)
    vla_policy.to(device)
    vla_policy.eval()
    for p in vla_policy.parameters():
        p.requires_grad = False

    rl_token_dim = vla_policy.config.rl_token_dim
    state_dim = vla_policy.config.max_state_dim
    action_dim = vla_policy.config.max_action_dim
    chunk_size = cfg.chunk_size_rl

    # ------- Load Actor-Critic -------
    logging.info(f"[DeployRLT] Loading actor-critic: {cfg.actor_critic_checkpoint}")
    ckpt = torch.load(cfg.actor_critic_checkpoint, map_location=device)
    actor_critic = RLTActorCritic(
        rl_token_dim=rl_token_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval()
    for p in actor_critic.parameters():
        p.requires_grad = False
    logging.info(f"[DeployRLT] Actor-Critic loaded (trained for {ckpt.get('step', '?')} steps)")

    # ------- Load Mode Switch Network (auto mode) -------
    mode_switch_net = None
    if cfg.mode == "auto":
        if cfg.mode_switch_checkpoint is None or not Path(cfg.mode_switch_checkpoint).exists():
            raise ValueError(
                "Auto mode requires a trained Mode Switch Network. "
                "Please specify --mode_switch_checkpoint or use --mode=manual"
            )
        logging.info(f"[DeployRLT] Loading Mode Switch Network: {cfg.mode_switch_checkpoint}")
        ms_ckpt = torch.load(cfg.mode_switch_checkpoint, map_location=device)
        mode_switch_net = ModeSwitchMLP(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
        ).to(device)
        mode_switch_net.load_state_dict(ms_ckpt["mode_switch"])
        mode_switch_net.eval()
        logging.info("[DeployRLT] Mode Switch Network loaded.")

    # ------- Robot -------
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logging.info("[DeployRLT] Robot connected.")

    # ------- Keyboard -------
    keyboard = DeployKeyboardController()
    keyboard.start()

    # ------- Print controls -------
    logging.info("\n" + "=" * 50)
    logging.info("[DeployRLT] Controls:")
    if cfg.mode == "manual":
        logging.info("  m  = toggle RL correction (currently OFF)")
    else:
        logging.info("  [AUTO] Mode Switch MLP will decide automatically")
    logging.info("  r  = reset episode")
    logging.info("  q  = quit")
    logging.info("=" * 50 + "\n")

    _quit = [False]
    def _sigint(sig, frame):
        _quit[0] = True
    signal.signal(signal.SIGINT, _sigint)

    episode_idx = 0
    step_count = 0

    try:
        while not _quit[0] and not keyboard.should_quit:
            episode_idx += 1
            logging.info(f"\n[DeployRLT] === Episode {episode_idx} ===")
            vla_policy.reset()

            # Episode loop
            while not _quit[0] and not keyboard.should_quit:
                loop_start = time.perf_counter()

                if keyboard.pop_reset():
                    logging.info("[DeployRLT] Episode reset by user.")
                    vla_policy.reset()
                    break

                # --- Get observation ---
                obs = robot.get_observation()
                batch_obs = _obs_to_batch(obs, cfg.task, vla_policy, device)

                # --- VLA inference ---
                with torch.no_grad():
                    vla_action_chunk = vla_policy.predict_action_chunk(batch_obs)
                    rl_token = vla_policy.extract_rl_token(batch_obs)

                proprioceptive = _get_proprioceptive_state(obs, vla_policy, device)
                rl_tok, prop_state = _make_rl_state(rl_token, proprioceptive)
                ref_action = vla_action_chunk.squeeze(0)   # (C, action_dim)

                # --- Actor inference (run in parallel with VLA in future) ---
                with torch.no_grad():
                    actor_action = actor_critic.select_action(
                        rl_tok.unsqueeze(0),
                        prop_state.unsqueeze(0),
                        ref_action.unsqueeze(0),
                        deterministic=cfg.deterministic,
                    ).squeeze(0)   # (C, action_dim)

                # --- Mode decision ---
                use_rl = False
                if cfg.mode == "manual":
                    use_rl = keyboard.rl_active
                elif cfg.mode == "auto" and mode_switch_net is not None:
                    with torch.no_grad():
                        mode_pred = mode_switch_net.predict_mode(
                            rl_tok.unsqueeze(0),
                            prop_state.unsqueeze(0),
                            ref_action.unsqueeze(0),
                        ).item()
                    use_rl = bool(mode_pred)

                # --- Select action ---
                if use_rl:
                    chosen_action = actor_action
                    mode_str = "ACTOR (RL)"
                else:
                    chosen_action = ref_action
                    mode_str = "VLA"

                step_count += 1
                if step_count % 10 == 0:
                    logging.info(
                        f"  Step {step_count:5d} | mode={mode_str:12s} | "
                        f"action_norm={chosen_action.norm().item():.3f}"
                    )

                # --- Execute action ---
                _execute_action_chunk(robot, chosen_action.cpu(), cfg.fps)

                # Timing
                dt = time.perf_counter() - loop_start
                precise_sleep(max(chunk_size / cfg.fps - dt, 0.0))

    finally:
        keyboard.stop()
        robot.disconnect()
        logging.info("[DeployRLT] Deployment ended.")


def main():
    register_third_party_plugins()
    deploy_rlt()


if __name__ == "__main__":
    main()
