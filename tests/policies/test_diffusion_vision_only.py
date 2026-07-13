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

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES
from lerobot.utils.import_utils import is_package_available


@pytest.mark.skipif(not is_package_available("diffusers"), reason="diffusers not installed")
def test_diffusion_policy_supports_vision_only_forward_and_generation():
    config = DiffusionConfig(
        input_features={
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        device="cpu",
        n_obs_steps=2,
        horizon=8,
        n_action_steps=3,
        down_dims=(16, 32),
        n_groups=4,
        diffusion_step_embed_dim=16,
        num_train_timesteps=4,
        num_inference_steps=2,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )
    policy = DiffusionPolicy(config)

    batch = {
        OBS_IMAGE: torch.rand(2, config.n_obs_steps, 3, 64, 64),
        ACTION: torch.randn(2, config.horizon, 2),
        "action_is_pad": torch.zeros(2, config.horizon, dtype=torch.bool),
    }

    loss, output_dict = policy.forward(batch)

    assert output_dict is None
    assert loss.ndim == 0

    policy.eval()
    actions = policy.diffusion.generate_actions(
        {
            OBS_IMAGES: batch[OBS_IMAGE].unsqueeze(2),
        }
    )

    assert actions.shape == (2, config.n_action_steps, 2)

    policy.reset()
    action = policy.select_action({OBS_IMAGE: torch.rand(2, 3, 64, 64)})

    assert action.shape == (2, 2)
