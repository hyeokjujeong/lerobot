#!/usr/bin/env python

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

"""Diffusion Policy conditioned by frozen PI0/PaliGemma vision features."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


DEFAULT_PI0_VISION_PRETRAINED_PATH = (
    "/home/jhri626/.cache/huggingface/hub/models--lerobot--pi0_base/"
    "snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2"
)


@PreTrainedConfig.register_subclass("pi0_vision_diffusion")
@dataclass
class PI0VisionDiffusionConfig(DiffusionConfig):
    """Diffusion Policy with the ResNet RGB encoder replaced by frozen PI0 vision tower features."""

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Avoid unnecessary torchvision checkpoint loading when the base DiffusionPolicy
    # initializer constructs and then replaces its temporary RGB encoder.
    pretrained_backbone_weights: str | None = None

    pi0_vision_pretrained_path: str = DEFAULT_PI0_VISION_PRETRAINED_PATH
    pi0_vision_feature_dim: int = 1152
    pi0_vision_num_tokens: int = 256
    pi0_vision_adapter_hidden_dim: int = 256
    pi0_vision_adapter_out_dim: int = 64
    pi0_vision_forward_batch_size: int = 32
    pi0_vision_image_resolution: tuple[int, int] = (224, 224)
    pi0_vision_input_scale: str = "uint8"
    pi0_vision_feature_dir: str | None = None
    pi0_vision_feature_camera_indices: tuple[int, ...] | None = (0, 1)
    pi0_vision_feature_token_pool: str = "mean"

    def __post_init__(self):
        super().__post_init__()
        if self.pi0_vision_feature_dim <= 0:
            raise ValueError("`pi0_vision_feature_dim` must be positive.")
        if self.pi0_vision_num_tokens <= 0:
            raise ValueError("`pi0_vision_num_tokens` must be positive.")
        if self.pi0_vision_adapter_hidden_dim <= 0:
            raise ValueError("`pi0_vision_adapter_hidden_dim` must be positive.")
        if self.pi0_vision_adapter_out_dim <= 0:
            raise ValueError("`pi0_vision_adapter_out_dim` must be positive.")
        if self.pi0_vision_forward_batch_size <= 0:
            raise ValueError("`pi0_vision_forward_batch_size` must be positive.")
        if len(self.pi0_vision_image_resolution) != 2:
            raise ValueError("`pi0_vision_image_resolution` must be a 2-tuple of (height, width).")
        if self.pi0_vision_input_scale not in ("float", "uint8"):
            raise ValueError("`pi0_vision_input_scale` must be 'float' or 'uint8'.")
        if self.pi0_vision_feature_token_pool not in ("mean", "max", "cls"):
            raise ValueError("`pi0_vision_feature_token_pool` must be 'mean', 'max', or 'cls'.")
