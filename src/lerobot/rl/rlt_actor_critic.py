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
Actor-Critic Network for RL Token (RLT) Online RL.

Architecture (TD3-style with Double Q):
  - State input: x = {z_rl, s^p}
    * z_rl: RL token from VLM encoder (rl_token_dim)
    * s^p:  proprioceptive state (state_dim)

  - Critic (Double Q):
    * Two Q networks Q_psi_1, Q_psi_2 each with target Q_psi_1', Q_psi_2'
    * Input: (x, a_{1:C}) flattened
    * Output: scalar Q value
    * Training: TD backup with smaller Q (Double Q strategy)

  - Actor (Gaussian policy):
    * Input: (x, a_ref_{1:C})  -- state + VLA reference action
    * Output: mean mu_theta(x, a_ref), using fixed std sigma
    * Samples corrected action: a = a_ref + delta ~ N(mu, sigma^2 * I)
    * Training: -Q_psi(x, a) + beta * ||a - a_ref||^2
    * Reference dropout: a_ref is randomly zeroed with probability ref_dropout

Per paper:
  - 3-layer MLP, hidden_dim=512 for critic/actor
  - gamma=0.99, tau=0.005
  - C=10 action chunk
  - sigma=0.1 (actor fixed std)
  - beta=0.5 (reference regularization weight)
  - ref_dropout=0.5
"""

import copy
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Utility: build a simple MLP
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    """Build a fully-connected MLP with ReLU activations."""
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers = []
    current_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Single Q Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Single Q-value network.

    Input:  (B, state_dim + action_chunk_dim)
    Output: (B, 1)  Q value
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        input_dim = rl_token_dim + state_dim + action_dim * chunk_size
        self.net = build_mlp(input_dim, hidden_dim, 1, num_layers)

    def forward(self, rl_token: Tensor, state: Tensor, action_chunk: Tensor) -> Tensor:
        """
        Args:
            rl_token:     (B, rl_token_dim)
            state:        (B, state_dim)    proprioceptive state
            action_chunk: (B, C, action_dim) or (B, C * action_dim)
        Returns:
            q_value: (B, 1)
        """
        if action_chunk.dim() == 3:
            action_chunk = action_chunk.flatten(1)   # (B, C*action_dim)
        x = torch.cat([rl_token, state, action_chunk], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Double Q Critic (with target networks)
# ---------------------------------------------------------------------------

class DoubleCritic(nn.Module):
    """
    TD3-style Double Q Critic with target networks.

    Contains:
      Q1, Q2  -- online networks
      Q1', Q2' -- target networks (soft-updated, no gradients)
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        kwargs = dict(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.q1 = QNetwork(**kwargs)
        self.q2 = QNetwork(**kwargs)

        # Target networks (deep copy, no gradients)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

    def forward(self, rl_token: Tensor, state: Tensor, action_chunk: Tensor):
        """
        Returns Q1 and Q2 values for the online networks.
        Returns: (q1_val, q2_val) each (B, 1)
        """
        return self.q1(rl_token, state, action_chunk), self.q2(rl_token, state, action_chunk)

    def q_min(self, rl_token: Tensor, state: Tensor, action_chunk: Tensor) -> Tensor:
        """Return element-wise minimum of Q1 and Q2 (online). Shape: (B, 1)"""
        q1, q2 = self.forward(rl_token, state, action_chunk)
        return torch.min(q1, q2)

    def target_q_min(self, rl_token: Tensor, state: Tensor, action_chunk: Tensor) -> Tensor:
        """Return element-wise minimum of target Q1' and Q2'. Shape: (B, 1)"""
        q1t = self.q1_target(rl_token, state, action_chunk)
        q2t = self.q2_target(rl_token, state, action_chunk)
        return torch.min(q1t, q2t)

    @torch.no_grad()
    def soft_update(self, tau: float = 0.005):
        """Polyak averaging: target = tau * online + (1 - tau) * target"""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_online_parameters(self):
        """Return only the online (trainable) network parameters."""
        return list(self.q1.parameters()) + list(self.q2.parameters())


# ---------------------------------------------------------------------------
# Gaussian Actor (corrects VLA reference actions)
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """
    Gaussian actor pi_theta(· | x, a_ref) = N(mu_theta(x, a_ref), sigma^2 * I).

    The actor outputs a *correction* delta to add to the VLA reference action,
    i.e., the actual action is a = a_ref + delta (clamped to valid range).

    Input:  (x, a_ref) = (rl_token || state || a_ref_flattened)
    Output: mu_theta   (B, C, action_dim)

    Reference dropout: with probability ref_dropout, a_ref is zeroed to prevent
    the actor from merely copying VLA actions.
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        sigma: float = 0.1,
        ref_dropout: float = 0.5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.sigma = sigma
        self.ref_dropout = ref_dropout

        input_dim = rl_token_dim + state_dim + action_dim * chunk_size
        self.net = build_mlp(input_dim, hidden_dim, action_dim * chunk_size, num_layers)

    def _apply_ref_dropout(self, ref_action: Tensor) -> Tensor:
        """
        Apply reference dropout: zero out entire ref_action chunk with probability ref_dropout.
        Each sample in the batch is independently dropped.
        """
        if not self.training:
            return ref_action
        B = ref_action.shape[0]
        # Bernoulli mask: 1 means keep, 0 means drop
        mask = torch.bernoulli(
            torch.ones(B, 1, 1, device=ref_action.device) * (1 - self.ref_dropout)
        )  # (B, 1, 1)
        return ref_action * mask

    def forward(
        self,
        rl_token: Tensor,
        state: Tensor,
        ref_action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            rl_token:   (B, rl_token_dim)
            state:      (B, state_dim)
            ref_action: (B, C, action_dim)  VLA reference action (may be zeroed by ref dropout)

        Returns:
            mu:         (B, C, action_dim)  mean of the Gaussian
            ref_dropped:(B, C, action_dim)  the (possibly zeroed) reference action used
        """
        ref_dropped = self._apply_ref_dropout(ref_action)
        ref_flat = ref_dropped.flatten(1)   # (B, C * action_dim)

        x = torch.cat([rl_token, state, ref_flat], dim=-1)
        mu_flat = self.net(x)   # (B, C * action_dim)
        mu = mu_flat.view(-1, self.chunk_size, self.action_dim)
        return mu, ref_dropped

    def sample(
        self,
        rl_token: Tensor,
        state: Tensor,
        ref_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Sample action: a = mu + sigma * eps  where eps ~ N(0, I).

        Returns:
            action:     (B, C, action_dim)
            mu:         (B, C, action_dim)
            ref_dropped:(B, C, action_dim)
        """
        mu, ref_dropped = self.forward(rl_token, state, ref_action)
        eps = torch.randn_like(mu)
        action = mu + self.sigma * eps
        return action, mu, ref_dropped

    def log_prob(self, action: Tensor, mu: Tensor) -> Tensor:
        """
        Gaussian log-probability of action under N(mu, sigma^2 I).
        Returns: (B,) summed over action and chunk dimensions.
        """
        dist = torch.distributions.Normal(mu, self.sigma)
        return dist.log_prob(action).sum(dim=(1, 2))


# ---------------------------------------------------------------------------
# Mode Switch MLP
# ---------------------------------------------------------------------------

class ModeSwitchMLP(nn.Module):
    """
    Binary classifier to decide whether the robot should use:
      0 -> VLA action (a_ref)
      1 -> Actor-corrected action (a from pi_theta)

    Input: (x, a_ref) = same as actor input
    Output: logit (B, 1)  -- sigmoid -> probability of mode=1

    Architecture: 2-layer MLP, hidden_dim=256 per design doc.
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        input_dim = rl_token_dim + state_dim + action_dim * chunk_size
        self.net = build_mlp(input_dim, hidden_dim, 1, num_layers)

    def forward(
        self,
        rl_token: Tensor,
        state: Tensor,
        ref_action: Tensor,
    ) -> Tensor:
        """
        Args:
            rl_token:   (B, rl_token_dim)
            state:      (B, state_dim)
            ref_action: (B, C, action_dim)
        Returns:
            logit: (B, 1)
        """
        ref_flat = ref_action.flatten(1)
        x = torch.cat([rl_token, state, ref_flat], dim=-1)
        return self.net(x)

    def predict_mode(self, rl_token: Tensor, state: Tensor, ref_action: Tensor) -> Tensor:
        """
        Returns binary mode decision (B, 1) as bool tensor.
        1 = use actor; 0 = use VLA.
        """
        logit = self.forward(rl_token, state, ref_action)
        return (torch.sigmoid(logit) > 0.5)


# ---------------------------------------------------------------------------
# RLT Actor-Critic (bundles actor + critic for convenient use)
# ---------------------------------------------------------------------------

class RLTActorCritic(nn.Module):
    """
    Full Actor-Critic module for RLT Online RL.

    Contains:
      - DoubleCritic (Q1, Q2 + targets)
      - GaussianActor (mu_theta)
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 10,
        actor_hidden_dim: int = 512,
        actor_num_layers: int = 3,
        critic_hidden_dim: int = 512,
        critic_num_layers: int = 3,
        sigma: float = 0.1,
        ref_dropout: float = 0.5,
    ):
        super().__init__()
        self.rl_token_dim = rl_token_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        self.critic = DoubleCritic(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=critic_hidden_dim,
            num_layers=critic_num_layers,
        )

        self.actor = GaussianActor(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=actor_hidden_dim,
            num_layers=actor_num_layers,
            sigma=sigma,
            ref_dropout=ref_dropout,
        )

    def soft_update_targets(self, tau: float = 0.005):
        self.critic.soft_update(tau)

    def select_action(
        self,
        rl_token: Tensor,
        state: Tensor,
        ref_action: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """
        Select action chunk for deployment.
        If deterministic=True, return mu directly; otherwise sample.
        """
        if deterministic:
            mu, _ = self.actor.forward(rl_token, state, ref_action)
            return mu
        action, _, _ = self.actor.sample(rl_token, state, ref_action)
        return action
