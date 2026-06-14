# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Adaptive Feature Transfer (AFT) for the Diffusion Policy (vision-only).

This package is an *additive* extension on top of ``lerobot.policies.diffusion``.
It implements the vision branch of Adaptive Feature Transfer (Qiu et al., 2024,
arXiv:2406.07337): it regularizes the Diffusion Policy RGB encoder so that its
per-camera visual representation reproduces the (kernel) structure of pre-trained
PI0 ``features.vision_tower`` features that were extracted offline and stored in
``extracted_feature/``.

Nothing in the base ``diffusion`` policy is modified; everything here is new.
"""

from .configuration_aft_diffusion import AFTDiffusionConfig as AFTDiffusionConfig
