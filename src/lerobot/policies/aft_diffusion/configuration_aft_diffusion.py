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

"""Configuration for the AFT-regularized Diffusion Policy (vision-only).

``AFTDiffusionConfig`` is a strict superset of ``DiffusionConfig``: every base
Diffusion Policy field/behavior is inherited unchanged, and only AFT-specific
options are added. Selecting ``--policy.type=aft_diffusion`` enables the vision
Adaptive Feature Transfer regularizer described in
``adaptive-feature-transfer/`` (arXiv:2406.07337).
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("aft_diffusion")
@dataclass
class AFTDiffusionConfig(DiffusionConfig):
    """Diffusion Policy + vision Adaptive Feature Transfer.

    AFT-specific args:
        aft_enable: Master switch. If False, behaves exactly like ``DiffusionConfig``
            (the AFT prior is still constructed for checkpoint compatibility but
            contributes no loss).
        aft_feature_dir: Directory of pre-extracted PI0 ``features.vision_tower``
            shards (``*.safetensors``). Typically
            ``lerobot/extracted_feature/<run>``. If empty/missing, AFT is inactive.
        aft_beta: Regularization strength ``beta`` (the ``prec`` of the AFT paper).
            The AFT term added to the diffusion loss is ``beta * kernel_rmse``.
        aft_pretrained_dim: Dimensionality of the *pooled* PI0 feature, i.e. the
            size of the prior's adaptive-scale vector. Must match what the feature
            store produces. Default 2304 = 2 *real* cameras x 1152 (vision-tower
            dim) with ``aft_camera_reduce='concat'``. Use 1152 for
            ``camera_reduce='mean'``.
        aft_camera_indices: Which ``features.vision_tower`` camera slots to keep.
            Default ``(0, 1)`` keeps the two real LIBERO cameras and drops the
            ``pi0_base`` dummy right-wrist slot (index 2). This also works for
            ``pi0_libero_base`` shards (which only have 2 slots). Set to ``None``
            to keep every slot.
        aft_learn_scales: Whether the per-dimension adaptive gates are trainable.
        aft_kernel: Kernel for the AFT match (``"linear"`` or ``"rbf"``).
        aft_token_pool: Pooling over the PI0 image-token axis (``"mean"``/``"max"``/``"cls"``).
        aft_camera_reduce: How PI0 per-camera features are combined (``"concat"``/``"mean"``).
        aft_obs_step: Which Diffusion Policy observation step to align with the
            (single-frame) PI0 feature. ``-1`` = current step (default).
        aft_prior_lr: Learning rate for the AFT prior's adaptive scales (separate
            optimizer param group). If None, uses the model learning rate.
    """

    aft_enable: bool = True
    aft_feature_dir: str | None = None
    aft_beta: float = 1.0
    aft_pretrained_dim: int = 2304
    aft_learn_scales: bool = True
    aft_kernel: str = "linear"
    aft_token_pool: str = "mean"
    aft_camera_reduce: str = "concat"
    aft_camera_indices: tuple[int, ...] | None = (0, 1)
    aft_obs_step: int = -1
    aft_prior_lr: float | None = 1e-2

    def __post_init__(self):
        super().__post_init__()
        if self.aft_kernel not in ("linear", "rbf"):
            raise ValueError(f"`aft_kernel` must be 'linear' or 'rbf'. Got {self.aft_kernel}.")
        if self.aft_token_pool not in ("mean", "max", "cls"):
            raise ValueError(f"`aft_token_pool` must be 'mean', 'max', or 'cls'. Got {self.aft_token_pool}.")
        if self.aft_camera_reduce not in ("concat", "mean"):
            raise ValueError(f"`aft_camera_reduce` must be 'concat' or 'mean'. Got {self.aft_camera_reduce}.")
        if self.aft_enable and self.aft_pretrained_dim <= 0:
            raise ValueError(f"`aft_pretrained_dim` must be a positive int. Got {self.aft_pretrained_dim}.")
