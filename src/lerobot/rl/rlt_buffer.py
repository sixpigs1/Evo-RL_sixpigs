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
Replay Buffer for RL Token (RLT) Online RL.

Stores transitions of the form:
    <x_t, a_{t:t+C-1}, a_ref_{t:t+C-1}, r_t, x_{t+1}, done>

where:
  x = {z_rl, s^p}  -- RL state (RL token + proprioceptive state)
  a                -- executed action chunk (from actor or human or VLA warmup)
  a_ref            -- VLA reference action chunk
  r                -- sparse reward at end of episode (0/1)
  x'               -- next state

Unified data directory format:
  data_path/
    meta.json           -- metadata (dims, robot type, data counts, etc.)
    play_buffer.pkl     -- replay buffer transitions
    mode_switch.pkl     -- mode switch MLP training data (optional)

Supports:
  - save / load for resumable online RL
  - mode switch data collection: records (x_t, a_ref, label) every frame
    label=1 when in RL phase, label=0 otherwise
"""

import json
import logging
import pickle
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor


class RLTTransition(TypedDict):
    """A single transition stored in the RLT replay buffer."""
    rl_token: Tensor          # (rl_token_dim,)
    state: Tensor             # (state_dim,)
    action: Tensor            # (C, action_dim)
    ref_action: Tensor        # (C, action_dim)
    reward: float
    next_rl_token: Tensor     # (rl_token_dim,)
    next_state: Tensor        # (state_dim,)
    done: bool


class RLTBatch(TypedDict):
    """A sampled batch from the RLT replay buffer (batched tensors)."""
    rl_token: Tensor        # (B, rl_token_dim)
    state: Tensor           # (B, state_dim)
    action: Tensor          # (B, C, action_dim)
    ref_action: Tensor      # (B, C, action_dim)
    reward: Tensor          # (B,)
    next_rl_token: Tensor   # (B, rl_token_dim)
    next_state: Tensor      # (B, state_dim)
    done: Tensor            # (B,) float


class RLTDataManager:
    """
    Unified data manager for RLT online RL data.

    Manages a data directory with the following structure::

        data_path/
            meta.json           -- metadata (dims, robot type, data counts ...)
            play_buffer.pkl     -- replay buffer transitions
            mode_switch.pkl     -- mode switch MLP data (optional)

    Example usage::

        manager = RLTDataManager("outputs/online_rl/data")
        manager.save_buffer(replay_buffer)
        manager.save_mode_switch(mode_switch_list)
        manager.save_meta(meta_dict)
    """

    PLAY_BUFFER_FILE = "play_buffer.pkl"
    MODE_SWITCH_FILE = "mode_switch.pkl"
    META_FILE = "meta.json"

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def _ensure_dir(self):
        self.data_path.mkdir(parents=True, exist_ok=True)

    # ---- meta ----

    def save_meta(self, meta: dict):
        """Write meta.json."""
        self._ensure_dir()
        with open(self.data_path / self.META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logging.info(f"[RLTDataManager] Saved meta to {self.data_path / self.META_FILE}")

    def load_meta(self) -> dict:
        """Load meta.json. Returns empty dict if not found."""
        meta_file = self.data_path / self.META_FILE
        if not meta_file.exists():
            return {}
        with open(meta_file) as f:
            return json.load(f)

    # ---- replay buffer ----

    def save_buffer(self, buffer: "RLTReplayBuffer"):
        """Save replay buffer to play_buffer.pkl."""
        self._ensure_dir()
        buffer.save(self.data_path / self.PLAY_BUFFER_FILE)

    def load_buffer(self, device: str = "cuda") -> "RLTReplayBuffer | None":
        """Load replay buffer. Returns None if not found."""
        buf_file = self.data_path / self.PLAY_BUFFER_FILE
        if not buf_file.exists():
            return None
        return RLTReplayBuffer.load(buf_file, target_device=device)

    # ---- mode switch ----

    def save_mode_switch(self, data: list[dict]):
        """Save mode switch data to mode_switch.pkl."""
        self._ensure_dir()
        path = self.data_path / self.MODE_SWITCH_FILE
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"[RLTDataManager] Saved {len(data)} mode-switch samples to {path}")

    def load_mode_switch(self) -> list[dict]:
        """Load mode switch data. Returns empty list if not found."""
        path = self.data_path / self.MODE_SWITCH_FILE
        if not path.exists():
            return []
        with open(path, "rb") as f:
            data = pickle.load(f)
        logging.info(f"[RLTDataManager] Loaded {len(data)} mode-switch samples from {path}")
        return data

    # ---- helpers ----

    def exists(self) -> bool:
        """Check if a saved play_buffer exists."""
        return (self.data_path / self.PLAY_BUFFER_FILE).exists()

    @staticmethod
    def build_meta(
        buffer: "RLTReplayBuffer",
        robot_type: str = "unknown",
        mode_switch_count: int = 0,
        extra: dict | None = None,
    ) -> dict:
        """Build a meta.json dict from buffer and run info."""
        meta: dict = {
            "robot_type": robot_type,
            "buffer_size": buffer.size,
            "buffer_capacity": buffer.capacity,
            "mode_switch_samples": mode_switch_count,
        }
        if buffer._initialized and buffer.rl_tokens is not None:
            meta.update({
                "rl_token_dim": int(buffer.rl_tokens.shape[-1]),
                "state_dim": int(buffer.states.shape[-1]),
                "chunk_size": int(buffer.actions.shape[-2]),
                "action_dim": int(buffer.actions.shape[-1]),
            })
        if extra:
            meta.update(extra)
        return meta


class RLTReplayBuffer:
    """
    Circular replay buffer for RLT Online RL.

    Stores (x, a, a_ref, r, x', done) transitions.

    Supports serialization to disk for training resumption.

    Also optionally stores (x_t, a_ref) data for Mode Switch Network training.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        device: str = "cuda",
        storage_device: str = "cpu",
    ):
        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device

        self.position = 0
        self.size = 0
        self._initialized = False

        # Storage tensors (allocated on first add)
        self.rl_tokens: Tensor | None = None
        self.states: Tensor | None = None
        self.actions: Tensor | None = None
        self.ref_actions: Tensor | None = None
        self.rewards: Tensor | None = None
        self.next_rl_tokens: Tensor | None = None
        self.next_states: Tensor | None = None
        self.dones: Tensor | None = None

        # Mode switch data (optional)
        self._mode_switch_data: list[dict] = []

    def _initialize_storage(self, transition: RLTTransition):
        """Allocate storage tensors based on first transition shapes."""
        dev = self.storage_device
        cap = self.capacity

        rl_token_dim = transition["rl_token"].shape[-1]
        state_dim = transition["state"].shape[-1]
        C = transition["action"].shape[-2]
        action_dim = transition["action"].shape[-1]

        self.rl_tokens      = torch.zeros(cap, rl_token_dim, device=dev)
        self.states         = torch.zeros(cap, state_dim,    device=dev)
        self.actions        = torch.zeros(cap, C, action_dim, device=dev)
        self.ref_actions    = torch.zeros(cap, C, action_dim, device=dev)
        self.rewards        = torch.zeros(cap,                device=dev)
        self.next_rl_tokens = torch.zeros(cap, rl_token_dim, device=dev)
        self.next_states    = torch.zeros(cap, state_dim,    device=dev)
        self.dones          = torch.zeros(cap,               device=dev)

        self._initialized = True
        logging.info(
            f"[RLTReplayBuffer] Initialized storage: "
            f"capacity={cap}, rl_token_dim={rl_token_dim}, "
            f"state_dim={state_dim}, chunk={C}x{action_dim}"
        )

    def add(self, transition: RLTTransition):
        """Add a transition to the buffer."""
        if not self._initialized:
            self._initialize_storage(transition)

        pos = self.position
        dev = self.storage_device

        def _to(t):
            return t.detach().to(dev).squeeze(0) if isinstance(t, Tensor) else torch.tensor(t, device=dev)

        self.rl_tokens[pos]      = _to(transition["rl_token"])
        self.states[pos]         = _to(transition["state"])
        self.actions[pos]        = _to(transition["action"])
        self.ref_actions[pos]    = _to(transition["ref_action"])
        self.rewards[pos]        = float(transition["reward"])
        self.next_rl_tokens[pos] = _to(transition["next_rl_token"])
        self.next_states[pos]    = _to(transition["next_state"])
        self.dones[pos]          = float(transition["done"])

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> RLTBatch:
        """
        Sample a random batch of transitions.

        Returns tensors on self.device.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        batch_size = min(batch_size, self.size)
        idx = torch.randint(0, self.size, (batch_size,))
        dev = self.device

        return RLTBatch(
            rl_token=self.rl_tokens[idx].to(dev),
            state=self.states[idx].to(dev),
            action=self.actions[idx].to(dev),
            ref_action=self.ref_actions[idx].to(dev),
            reward=self.rewards[idx].to(dev),
            next_rl_token=self.next_rl_tokens[idx].to(dev),
            next_state=self.next_states[idx].to(dev),
            done=self.dones[idx].to(dev),
        )

    def __len__(self) -> int:
        return self.size

    def save(self, path: str | Path):
        """
        Serialize the entire buffer to disk.
        Saves as a single pickle file containing all tensors and metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "capacity": self.capacity,
            "device": self.device,
            "storage_device": self.storage_device,
            "position": self.position,
            "size": self.size,
            "_initialized": self._initialized,
        }
        if self._initialized:
            state.update({
                "rl_tokens": self.rl_tokens[:self.size].cpu(),
                "states": self.states[:self.size].cpu(),
                "actions": self.actions[:self.size].cpu(),
                "ref_actions": self.ref_actions[:self.size].cpu(),
                "rewards": self.rewards[:self.size].cpu(),
                "next_rl_tokens": self.next_rl_tokens[:self.size].cpu(),
                "next_states": self.next_states[:self.size].cpu(),
                "dones": self.dones[:self.size].cpu(),
            })

        with open(path, "wb") as f:
            pickle.dump(state, f)
        logging.info(f"[RLTReplayBuffer] Saved buffer ({self.size} transitions) to {path}")

    @classmethod
    def load(cls, path: str | Path, target_device: str | None = None) -> "RLTReplayBuffer":
        """Load a previously saved replay buffer from disk.

        Args:
            path:          Path to the pkl file.
            target_device: If given, override the stored ``device`` field
                           (useful when loading on a different machine).
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        buf = cls(
            capacity=state["capacity"],
            device=target_device if target_device is not None else state["device"],
            storage_device=state["storage_device"],
        )
        buf.position = state["position"]
        buf.size = state["size"]
        buf._initialized = state["_initialized"]

    @classmethod
    def load(cls, path: str | Path, target_device: str | None = None) -> "RLTReplayBuffer":
        """Load a previously saved replay buffer from disk.

        Args:
            path:          Path to the pkl file.
            target_device: If provided, override the stored ``device`` field
                           (useful when resuming on a different machine).
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        device = target_device if target_device is not None else state["device"]
        buf = cls(
            capacity=state["capacity"],
            device=device,
            storage_device=state["storage_device"],
        )
        buf.position = state["position"]
        buf.size = state["size"]
        buf._initialized = state["_initialized"]

        if buf._initialized:
            dev = buf.storage_device
            cap = buf.capacity

            def _reload(key, shape_extra):
                data = state[key].to(dev)
                full = torch.zeros(cap, *shape_extra, device=dev)
                full[:buf.size] = data
                return full

            rl_dim    = state["rl_tokens"].shape[-1]
            state_dim = state["states"].shape[-1]
            C         = state["actions"].shape[-2]
            act_dim   = state["actions"].shape[-1]

            buf.rl_tokens      = _reload("rl_tokens",      (rl_dim,))
            buf.states         = _reload("states",          (state_dim,))
            buf.actions        = _reload("actions",         (C, act_dim))
            buf.ref_actions    = _reload("ref_actions",     (C, act_dim))
            buf.rewards        = _reload("rewards",         ())
            buf.next_rl_tokens = _reload("next_rl_tokens",  (rl_dim,))
            buf.next_states    = _reload("next_states",     (state_dim,))
            buf.dones          = _reload("dones",           ())

        logging.info(f"[RLTReplayBuffer] Loaded buffer ({buf.size} transitions) from {path}")
        return buf

    # ------------------------------------------------------------------
    # Mode Switch Data Collection
    # ------------------------------------------------------------------

    def add_mode_switch_sample(self, rl_token: Tensor, state: Tensor, ref_action: Tensor, label: int):
        """
        Store a single (x_t, a_ref, label) sample for Mode Switch Network training.

        Should be called EVERY frame (both inside and outside RL phase):
          label=1  when currently in RL phase (actor is acting)
          label=0  when not in RL phase (VLA warmup / VLA-only phase)

        Args:
            rl_token:   (rl_token_dim,)
            state:      (state_dim,)
            ref_action: (C, action_dim)
            label:      0 or 1
        """
        self._mode_switch_data.append({
            "rl_token": rl_token.detach().cpu(),
            "state": state.detach().cpu(),
            "ref_action": ref_action.detach().cpu(),
            "label": int(label),
        })

    def get_mode_switch_data(self) -> list[dict]:
        """Return collected mode switch training data."""
        return self._mode_switch_data

    def set_mode_switch_data(self, data: list[dict]):
        """Replace mode switch data (e.g. after loading from disk)."""
        self._mode_switch_data = data
