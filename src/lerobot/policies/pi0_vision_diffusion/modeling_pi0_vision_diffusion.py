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

"""Diffusion Policy conditioned by frozen PI0/PaliGemma vision tower features."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import einops
import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    _make_noise_scheduler,
)
from lerobot.policies.aft_diffusion.feature_store import PI0VisionFeatureStore
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, resize_with_pad_torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_pi0_vision_diffusion import PI0VisionDiffusionConfig


# ---------------------------------------------------------------------------
# DEBUG STOP FOR THE FIRST VERIFICATION RUN.
#
# This is intentionally hardcoded per the experiment plan. Run a tiny job once,
# inspect /PublicSSD/jhri626/tmp/pi0_vision_diffusion_debug/vision_tower_output.pt,
# then comment out this constant/block before launching real training.
# ---------------------------------------------------------------------------
DEBUG_SAVE_VISION_TOWER_OUTPUT_AND_QUIT = False
DEBUG_VISION_TOWER_OUTPUT_PATH = Path(
    "/PublicSSD/jhri626/tmp/pi0_vision_diffusion_debug/vision_tower_output.pt"
)
DEBUG_DATASET_EPISODES_PATH = Path(
    "/PublicSSD/jhri626/datasets/mimicgen_coffee_d2_lerobot_images/meta/episodes/chunk-000/file-000.parquet"
)


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def _tensor_debug_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _compute_debug_frame_indices(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    if "episode_index" not in batch or "index" not in batch or not DEBUG_DATASET_EPISODES_PATH.exists():
        return {}

    import pandas as pd

    episodes = pd.read_parquet(DEBUG_DATASET_EPISODES_PATH)
    episode_starts = {
        int(row.episode_index): int(row.dataset_from_index)
        for _, row in episodes.iterrows()
    }
    episode_ends = {
        int(row.episode_index): int(row.dataset_to_index)
        for _, row in episodes.iterrows()
    }

    episode_index = batch["episode_index"].detach().cpu().flatten()
    absolute_index = batch["index"].detach().cpu().flatten()

    frame_indices = []
    observation_frame_indices = []
    for ep_idx, abs_idx in zip(episode_index.tolist(), absolute_index.tolist(), strict=True):
        ep_start = episode_starts[int(ep_idx)]
        ep_end = episode_ends[int(ep_idx)]
        current_frame = int(abs_idx) - ep_start
        frame_indices.append(current_frame)

        obs_frames = []
        for delta in range(1 - len(batch[OBS_IMAGES][0]), 1):
            obs_abs_idx = max(ep_start, min(ep_end - 1, int(abs_idx) + delta))
            obs_frames.append(obs_abs_idx - ep_start)
        observation_frame_indices.append(obs_frames)

    return {
        "computed_frame_index": torch.tensor(frame_indices, dtype=torch.int64),
        "computed_observation_frame_indices": torch.tensor(observation_frame_indices, dtype=torch.int64),
    }


class PI0VisionDiffusionModel(DiffusionModel):
    """Diffusion model whose visual conditioning comes from a frozen PI0 vision tower."""

    def __init__(self, config: PI0VisionDiffusionConfig):
        nn.Module.__init__(self)
        self.config = config

        if not self.config.image_features:
            raise ValueError("`pi0_vision_diffusion` requires at least one visual observation feature.")

        self.vision_tower = None if config.pi0_vision_feature_dir else self._load_frozen_pi0_vision_tower(config)
        self._feature_store = self._load_feature_store(config)
        self._episode_from_index: dict[int, int] | None = None
        self.pi0_vision_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.pi0_vision_feature_dim),
                    nn.Linear(config.pi0_vision_feature_dim, config.pi0_vision_adapter_hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.pi0_vision_adapter_hidden_dim, config.pi0_vision_adapter_out_dim),
                )
                for _ in self.config.image_features
            ]
        )
        self.pi0_vision_metrics: dict[str, float] | None = None

        global_cond_dim = self.config.robot_state_feature.shape[0]
        global_cond_dim += len(self.config.image_features) * self.config.pi0_vision_adapter_out_dim
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

    @staticmethod
    def _load_feature_store(config: PI0VisionDiffusionConfig) -> PI0VisionFeatureStore | None:
        if not config.pi0_vision_feature_dir:
            return None
        store = PI0VisionFeatureStore(
            feature_dir=config.pi0_vision_feature_dir,
            token_pool=config.pi0_vision_feature_token_pool,
            camera_reduce="concat",
            camera_indices=config.pi0_vision_feature_camera_indices,
        )
        expected_dim = len(config.image_features) * config.pi0_vision_feature_dim
        if store.is_empty:
            raise ValueError(f"No usable PI0 vision features found in {config.pi0_vision_feature_dir}.")
        if store.feature_dim != expected_dim:
            raise ValueError(
                "PI0 vision feature store dim mismatch: "
                f"loaded {store.feature_dim}, expected {expected_dim} "
                f"({len(config.image_features)} cameras * {config.pi0_vision_feature_dim})."
            )
        return store

    @staticmethod
    def _load_frozen_pi0_vision_tower(config: PI0VisionDiffusionConfig) -> nn.Module:
        pi0_config = PreTrainedConfig.from_pretrained(config.pi0_vision_pretrained_path)
        pi0_config.device = "cpu"
        if hasattr(pi0_config, "freeze_vision_encoder"):
            pi0_config.freeze_vision_encoder = True

        pi0_policy = PI0Policy.from_pretrained(
            config.pi0_vision_pretrained_path,
            config=pi0_config,
            strict=False,
        )
        vision_tower = pi0_policy.model.paligemma_with_expert.paligemma.model.vision_tower
        _freeze_module(vision_tower)
        del pi0_policy
        gc.collect()
        return vision_tower

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_tower is not None:
            self.vision_tower.eval()
        return self

    def set_episode_offsets(self, episode_from_index: dict[int, int] | None) -> None:
        self._episode_from_index = episode_from_index

    def _ensure_vision_tower(self, device: torch.device) -> nn.Module:
        if self.vision_tower is None:
            self.vision_tower = self._load_frozen_pi0_vision_tower(self.config).to(device)
        self.vision_tower.eval()
        return self.vision_tower

    def _preprocess_pi0_images(self, images: Tensor) -> Tensor:
        was_uint8 = images.dtype == torch.uint8
        if images.dtype != torch.float32:
            images = images.to(dtype=torch.float32)
        if self.config.pi0_vision_input_scale == "uint8" and not was_uint8:
            images = images * 255.0

        is_channels_first = images.shape[1] == 3
        if is_channels_first:
            images = images.permute(0, 2, 3, 1)

        height, width = self.config.pi0_vision_image_resolution
        if images.shape[1:3] != (height, width):
            images = resize_with_pad_torch(images, height, width)

        images = images * 2.0 - 1.0

        if is_channels_first:
            images = images.permute(0, 3, 1, 2)

        return images

    def _run_vision_tower(self, images: Tensor) -> Tensor:
        chunks = []
        chunk_size = self.config.pi0_vision_forward_batch_size
        vision_tower = self._ensure_vision_tower(images.device)
        with torch.no_grad():
            for chunk in images.split(chunk_size, dim=0):
                output = vision_tower(pixel_values=chunk)
                chunks.append(output.last_hidden_state)
        tokens = torch.cat(chunks, dim=0)
        expected_shape = (self.config.pi0_vision_num_tokens, self.config.pi0_vision_feature_dim)
        if tokens.shape[1:] != expected_shape:
            raise ValueError(
                "Unexpected PI0 vision tower output shape "
                f"{tuple(tokens.shape)}; expected (*, {expected_shape[0]}, {expected_shape[1]})."
            )
        return tokens

    def _save_debug_output_and_quit(
        self,
        batch: dict[str, Tensor],
        flat_images: Tensor,
        preprocessed_images: Tensor,
        tokens: Tensor,
        pooled: Tensor,
    ) -> None:
        DEBUG_VISION_TOWER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        debug_payload = {
            "tokens": tokens.detach().cpu(),
            "pooled": pooled.detach().cpu(),
            "pi0_vision_input_scale": self.config.pi0_vision_input_scale,
            "pi0_vision_image_resolution": tuple(self.config.pi0_vision_image_resolution),
            "pi0_vision_pretrained_path": self.config.pi0_vision_pretrained_path,
            "flat_images_shape": tuple(flat_images.shape),
            "preprocessed_images_shape": tuple(preprocessed_images.shape),
            "tokens_shape": tuple(tokens.shape),
            "pooled_shape": tuple(pooled.shape),
            "image_min": float(flat_images.detach().amin().cpu()),
            "image_max": float(flat_images.detach().amax().cpu()),
            "preprocessed_image_min": float(preprocessed_images.detach().amin().cpu()),
            "preprocessed_image_max": float(preprocessed_images.detach().amax().cpu()),
        }
        for key in ("episode_index", "frame_index", "index", "timestamp", "task_index"):
            if key in batch:
                debug_payload[key] = _tensor_debug_value(batch[key])
        debug_payload.update(_compute_debug_frame_indices(batch))
        torch.save(debug_payload, DEBUG_VISION_TOWER_OUTPUT_PATH)
        print(f"Saved PI0 vision tower debug output to {DEBUG_VISION_TOWER_OUTPUT_PATH}")
        quit()

    def _adapt_pooled_features(self, pooled_by_camera: Tensor, batch_size: int, n_obs_steps: int) -> Tensor:
        pooled_per_camera = einops.rearrange(
            pooled_by_camera,
            "b s n f -> n (b s) f",
            b=batch_size,
            s=n_obs_steps,
            n=len(self.config.image_features),
        )
        img_features_list = torch.cat(
            [
                adapter(camera_pooled)
                for adapter, camera_pooled in zip(
                    self.pi0_vision_adapters, pooled_per_camera, strict=True
                )
            ],
            dim=0,
        )
        return einops.rearrange(
            img_features_list,
            "(n b s) f -> b s (n f)",
            b=batch_size,
            s=n_obs_steps,
            n=len(self.config.image_features),
        )

    def _batch_episode_observation_frames(
        self, batch: dict[str, Tensor], batch_size: int, n_obs_steps: int
    ) -> tuple[Tensor, Tensor] | None:
        if "episode_index" not in batch:
            return None

        episode_index = batch["episode_index"].detach().to("cpu").reshape(batch_size, -1)[:, -1].to(torch.int64)
        if "frame_index" in batch:
            current_frame = (
                batch["frame_index"].detach().to("cpu").reshape(batch_size, -1)[:, -1].to(torch.int64)
            )
        elif "index" in batch and self._episode_from_index is not None:
            absolute_index = batch["index"].detach().to("cpu").reshape(batch_size, -1)[:, -1].to(torch.int64)
            offsets = torch.tensor(
                [self._episode_from_index.get(int(ep), 1 << 60) for ep in episode_index.tolist()],
                dtype=torch.int64,
            )
            current_frame = absolute_index - offsets
        else:
            return None

        deltas = torch.arange(1 - n_obs_steps, 1, dtype=torch.int64)
        obs_frames = (current_frame[:, None] + deltas[None]).clamp_min(0)
        return episode_index, obs_frames

    def _encode_precomputed_features(
        self, batch: dict[str, Tensor], batch_size: int, n_obs_steps: int
    ) -> Tensor | None:
        if self._feature_store is None:
            return None
        episode_frames = self._batch_episode_observation_frames(batch, batch_size, n_obs_steps)
        if episode_frames is None:
            return None

        episode_index, obs_frames = episode_frames
        flat_episode_index = episode_index[:, None].expand(-1, n_obs_steps).reshape(-1)
        flat_frame_index = obs_frames.reshape(-1)
        features, mask = self._feature_store.lookup(flat_episode_index, flat_frame_index)
        if not bool(mask.all()):
            missing = [
                (int(ep), int(fr))
                for ep, fr, ok in zip(
                    flat_episode_index.tolist(), flat_frame_index.tolist(), mask.tolist(), strict=True
                )
                if not ok
            ]
            raise ValueError(f"Missing PI0 vision features for {len(missing)} frame(s), examples={missing[:8]}.")

        features = features.to(device=batch[OBS_STATE].device, dtype=batch[OBS_STATE].dtype)
        pooled_by_camera = einops.rearrange(
            features,
            "(b s) (n f) -> b s n f",
            b=batch_size,
            s=n_obs_steps,
            n=len(self.config.image_features),
            f=self.config.pi0_vision_feature_dim,
        )
        img_features = self._adapt_pooled_features(pooled_by_camera, batch_size, n_obs_steps)
        self.pi0_vision_metrics = {
            "pi0_vision_feature_store_mask": float(mask.float().mean()),
            "pi0_vision_pooled_mean": float(pooled_by_camera.detach().mean().cpu()),
            "pi0_vision_pooled_std": float(pooled_by_camera.detach().std().cpu()),
            "pi0_vision_adapter_mean": float(img_features.detach().mean().cpu()),
            "pi0_vision_adapter_std": float(img_features.detach().std().cpu()),
        }
        return img_features

    def _encode_images_online(self, batch: dict[str, Tensor], batch_size: int, n_obs_steps: int) -> Tensor:
        flat_images = einops.rearrange(batch[OBS_IMAGES], "b s n c h w -> (b s n) c h w")
        preprocessed_images = self._preprocess_pi0_images(flat_images)
        tokens = self._run_vision_tower(preprocessed_images)
        pooled = tokens.mean(dim=1)

        # Comment out this block after the first manual feature inspection.
        if DEBUG_SAVE_VISION_TOWER_OUTPUT_AND_QUIT:
            self._save_debug_output_and_quit(batch, flat_images, preprocessed_images, tokens, pooled)

        pooled_by_camera = einops.rearrange(
            pooled,
            "(b s n) f -> b s n f",
            b=batch_size,
            s=n_obs_steps,
            n=len(self.config.image_features),
        )
        img_features = self._adapt_pooled_features(pooled_by_camera, batch_size, n_obs_steps)
        self.pi0_vision_metrics = {
            "pi0_vision_token_mean": float(tokens.detach().mean().cpu()),
            "pi0_vision_token_std": float(tokens.detach().std().cpu()),
            "pi0_vision_pooled_mean": float(pooled.detach().mean().cpu()),
            "pi0_vision_pooled_std": float(pooled.detach().std().cpu()),
            "pi0_vision_adapter_mean": float(img_features.detach().mean().cpu()),
            "pi0_vision_adapter_std": float(img_features.detach().std().cpu()),
        }

        return img_features

    def _encode_images(self, batch: dict[str, Tensor], batch_size: int, n_obs_steps: int) -> Tensor:
        precomputed = self._encode_precomputed_features(batch, batch_size, n_obs_steps)
        if precomputed is not None:
            return precomputed
        return self._encode_images_online(batch, batch_size, n_obs_steps)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        self.pi0_vision_metrics = None

        if self.config.image_features:
            img_features = self._encode_images(batch, batch_size=batch_size, n_obs_steps=n_obs_steps)
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


class PI0VisionDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy with frozen PI0/PaliGemma vision tower visual conditioning."""

    config_class = PI0VisionDiffusionConfig
    name = "pi0_vision_diffusion"

    def __init__(self, config: PI0VisionDiffusionConfig, **kwargs):
        require_package("diffusers", extra="diffusion")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = PI0VisionDiffusionModel(config)
        self.diffusion.set_episode_offsets(self._build_episode_offsets(kwargs.get("dataset_meta")))
        self.reset()

    @staticmethod
    def _build_episode_offsets(dataset_meta) -> dict[int, int] | None:
        if dataset_meta is None:
            return None
        try:
            episodes = dataset_meta.episodes
            if episodes is None:
                return None
            return {
                int(e): int(fi)
                for e, fi in zip(episodes["episode_index"], episodes["dataset_from_index"], strict=True)
            }
        except Exception as e:
            logging.warning("PI0VisionDiffusion: could not build episode offsets (%s).", e)
            return None

    def get_optim_params(self) -> list[nn.Parameter]:
        return [param for param in self.diffusion.parameters() if param.requires_grad]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        return loss, self.diffusion.pi0_vision_metrics
