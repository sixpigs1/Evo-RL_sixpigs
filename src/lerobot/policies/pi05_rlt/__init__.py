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
PI05 with RL Token (PI05-RLT) policy package.

This package implements the RL Token method on top of PI0.5, following:
  "RL Token: Improving Robot Learning with RL-Enhanced Representations"

Architecture overview:
  - VLA backbone: PI0.5 (PaliGemma + Gemma action expert)
  - RL Token encoder: Transformer g_phi  -- extracts z_rl from image embeddings
  - RL Token decoder: Transformer d_phi  -- reconstructs image embeddings for compression loss
  - Actor-Critic network: TD3-style Double-Q critic + Gaussian actor that corrects VLA actions
"""

from .configuration_pi05_rlt import PI05RLTConfig
from .modeling_pi05_rlt import PI05RLTPolicy

__all__ = ["PI05RLTConfig", "PI05RLTPolicy"]
