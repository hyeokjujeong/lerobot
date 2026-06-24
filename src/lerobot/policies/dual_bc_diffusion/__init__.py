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

"""Dual-BC Diffusion Policy: a teacher-feature behavioral-cloning auxiliary loss.

Additive extension of the stock Diffusion Policy. Instead of the AFT kernel-RMSE
regularizer, it adds a *second* behavioral-cloning loss that conditions the
*shared* diffusion U-Net on the (dimension-matched) PI0 teacher feature, and sums
the two BC losses naively. See ``DUAL_BC_DIFFUSION_IMPLEMENTATION.md``.
"""
