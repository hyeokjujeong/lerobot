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

"""AFT-regularized Diffusion Policy (vision-only).

This module is purely additive: it subclasses the stock
``lerobot.policies.diffusion`` model/policy and layers Adaptive Feature Transfer
on top of the RGB-encoder path. The base Diffusion Policy code is untouched.

What it does
------------
During ``compute_loss`` we (1) capture the per-camera RGB-encoder outputs via
forward hooks (no recompute, no change to the base forward), (2) take the
current observation step's visual feature, (3) look up the matching offline PI0
``features.vision_tower`` feature for each frame, and (4) add the AFT
kernel-matching regularizer (``VisionKernelPrior``) to the diffusion loss.

The regularized RGB feature is exactly the vector that the base policy
concatenates into ``global_cond`` and feeds to the U-Net's **FiLM** conditioning
(scale/bias) in every residual block. So aligning it with PI0's visual
representation steers the FiLM conditioning signal toward PI0's task-relevant
visual knowledge. See ``AFT_VISION_TRANSFER_IMPLEMENTATION.md`` for the full
FiLM-matching discussion.
"""

from __future__ import annotations

import logging

import einops
import torch
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_STATE

from ..diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from .aft_prior import VisionKernelPrior
from .configuration_aft_diffusion import AFTDiffusionConfig
from .feature_store import PI0ActionFeatureStore, PI0VisionFeatureStore


class AFTDiffusionModel(DiffusionModel):
    """Diffusion model that adds a vision AFT regularizer to ``compute_loss``."""

    def __init__(self, config: AFTDiffusionConfig):
        super().__init__(config)
        self.config: AFTDiffusionConfig = config

        # AFT prior is a submodule so its adaptive scales are trained and checkpointed
        # alongside the rest of the policy. Built unconditionally (when enabled) so that
        # checkpoints reload cleanly even when the feature directory is absent at eval time.
        self.aft_prior: VisionKernelPrior | None = None
        if config.aft_enable and config.image_features:
            self.aft_prior = VisionKernelPrior(
                num_features=config.aft_pretrained_dim,
                learn_scales=config.aft_learn_scales,
                kernel=config.aft_kernel,
            )
        self.action_aft_prior: VisionKernelPrior | None = None
        if config.action_aft_enable:
            self.action_aft_prior = VisionKernelPrior(
                num_features=config.action_aft_pretrained_dim,
                learn_scales=config.action_aft_learn_scales,
                kernel=config.action_aft_kernel,
            )

        # Offline PI0 feature store (plain attribute, NOT a submodule / no params).
        self._feature_store: PI0VisionFeatureStore | None = None
        self._action_feature_store: PI0ActionFeatureStore | None = None
        # Map episode_index -> global dataset start index, used to reconstruct a
        # frame_index from the batch's global `index` when the preprocessor has
        # dropped `frame_index` (the LeRobot diffusion preprocessor keeps `index`
        # and `episode_index` but not `frame_index`).
        self._episode_from_index: dict[int, int] | None = None
        # Last-computed AFT logging metrics, surfaced via the policy forward output dict.
        self.aft_metrics: dict | None = None

        # Forward hooks to capture per-camera RGB-encoder outputs during the normal forward.
        self._aft_captured: list[Tensor] = []
        self._action_aft_captured: Tensor | None = None
        if config.image_features:
            encoders = self.rgb_encoder if isinstance(self.rgb_encoder, nn.ModuleList) else [self.rgb_encoder]
            for enc in encoders:
                enc.register_forward_hook(self._aft_capture_hook)

    # ---- AFT plumbing -------------------------------------------------------
    def _aft_capture_hook(self, _module, _inputs, output):  # noqa: ANN001
        self._aft_captured.append(output)

    def attach_feature_store(self, store: PI0VisionFeatureStore | None) -> None:
        self._feature_store = store

    def attach_action_feature_store(self, store: PI0ActionFeatureStore | None) -> None:
        self._action_feature_store = store

    def set_episode_offsets(self, episode_from_index: dict[int, int] | None) -> None:
        self._episode_from_index = episode_from_index

    def _batch_episode_frame(self, batch: dict[str, Tensor]):
        """Return ``(episode_index, frame_index)`` tensors for AFT matching.

        Uses ``frame_index`` directly when present; otherwise reconstructs it from
        the global ``index`` and per-episode start offsets (the diffusion
        preprocessor drops ``frame_index`` but keeps ``index`` and ``episode_index``).
        Returns ``(None, None)`` if neither path is available.
        """
        if "episode_index" not in batch:
            return None, None
        ep = batch["episode_index"]
        if "frame_index" in batch:
            return ep, batch["frame_index"]
        if "index" in batch and self._episode_from_index is not None:
            ep_flat = ep.detach().to("cpu").reshape(ep.shape[0], -1)[:, -1].to(torch.int64)
            idx = batch["index"]
            idx_flat = idx.detach().to("cpu").reshape(idx.shape[0], -1)[:, -1].to(torch.int64)
            # Sentinel offset for unknown episodes -> reconstructed frame won't match (safe).
            offsets = torch.tensor(
                [self._episode_from_index.get(int(e), 1 << 60) for e in ep_flat], dtype=torch.int64
            )
            return ep_flat, idx_flat - offsets
        return None, None

    def _aft_active(self) -> bool:
        return (
            self.config.aft_enable
            and bool(self.config.image_features)
            and self.aft_prior is not None
            and self._feature_store is not None
            and not self._feature_store.is_empty
        )

    def _action_aft_active(self) -> bool:
        return (
            self.config.action_aft_enable
            and self.action_aft_prior is not None
            and self._action_feature_store is not None
            and not self._action_feature_store.is_empty
        )

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        # Reset the capture buffer right before the encoders run, then defer to the
        # unmodified base implementation (whose behavior is preserved exactly).
        self._aft_captured = []
        return super()._prepare_global_conditioning(batch)

    def _assemble_aft_vision_feat(self, batch_size: int, n_obs_steps: int) -> Tensor | None:
        """Rebuild the (B, d_down) downstream visual feature from captured encoder outputs."""
        if not self._aft_captured:
            return None
        n_cam = len(self.config.image_features)
        if self.config.use_separate_rgb_encoder_per_camera:
            if len(self._aft_captured) != n_cam:
                return None
            # Each capture is (B*S, feat); stack over cameras -> (n_cam, B*S, feat).
            feats = torch.stack(self._aft_captured, dim=0)
            feats = einops.rearrange(feats, "n (b s) f -> b s n f", b=batch_size, s=n_obs_steps)
        else:
            # Single shared-encoder capture: (B*S*n_cam, feat).
            out = self._aft_captured[0]
            feats = einops.rearrange(out, "(b s n) f -> b s n f", b=batch_size, s=n_obs_steps, n=n_cam)
        cur = feats[:, self.config.aft_obs_step]  # (B, n_cam, feat) at the chosen obs step
        if self.config.aft_camera_reduce == "concat":
            return cur.reshape(batch_size, -1)
        return cur.mean(dim=1)

    def _action_aft_hook(self, _module, _inputs, output):  # noqa: ANN001
        if torch.is_tensor(output):
            self._action_aft_captured = output
        elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            self._action_aft_captured = output[0]

    def _compute_action_aft_loss(
        self,
        batch: dict[str, Tensor],
        ep_ids: Tensor | None,
        fr_ids: Tensor | None,
    ) -> tuple[Tensor | None, dict[str, float | int | str]]:
        metrics: dict[str, float | int | str] = {
            "aft_action_matched": 0,
            "aft_action_timestep": int(self.config.action_aft_timestep),
            "aft_action_store_mode": self.config.action_aft_store_mode,
        }
        if not self._action_aft_active() or ep_ids is None or fr_ids is None:
            return None, metrics
        if self.config.action_aft_timestep >= self.noise_scheduler.config.num_train_timesteps:
            logging.warning(
                "Action AFT: timestep %d is outside scheduler range [0, %d); skipping.",
                self.config.action_aft_timestep,
                self.noise_scheduler.config.num_train_timesteps,
            )
            return None, metrics

        pre, mask = self._action_feature_store.lookup(ep_ids, fr_ids)
        n_matched = int(mask.sum().item())
        metrics["aft_action_matched"] = n_matched
        if n_matched < 2:
            metrics["aft_action_loss"] = 0.0
            return None, metrics

        global_cond = self._prepare_global_conditioning(batch)
        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.full(
            (trajectory.shape[0],),
            int(self.config.action_aft_timestep),
            dtype=torch.long,
            device=trajectory.device,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        if len(getattr(self.unet, "down_modules", [])) <= 2:
            logging.warning("Action AFT: U-Net has no down_modules[2]; skipping.")
            return None, metrics
        self._action_aft_captured = None
        handle = self.unet.down_modules[2][1].register_forward_hook(self._action_aft_hook)
        try:
            _ = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
        finally:
            handle.remove()

        down = self._action_aft_captured
        self._action_aft_captured = None
        if down is None:
            logging.warning("Action AFT: failed to capture U-Net down2 feature; skipping.")
            return None, metrics
        if down.dim() != 3:
            logging.warning("Action AFT: expected down2 shape (B,C,T), got %s; skipping.", tuple(down.shape))
            return None, metrics
        down = down.mean(dim=-1)

        device, dtype = down.device, down.dtype
        mask = mask.to(device)
        pre = pre.to(device=device, dtype=dtype)
        down_m = down[mask]
        pre_m = pre[mask]
        if pre_m.shape[1] != self.action_aft_prior.num_features:
            logging.warning(
                "Action AFT: pre-trained feature dim %d != prior dim %d; skipping. "
                "Set `action_aft_pretrained_dim` to match the action feature store.",
                pre_m.shape[1],
                self.action_aft_prior.num_features,
            )
            return None, metrics

        action_loss = self.action_aft_prior.loss(down_m, pre_m, self.config.action_aft_beta)
        metrics.update(
            {
                "aft_action_kernel_rmse": self.action_aft_prior.metrics.get("aft_kernel_rmse", 0.0),
                "aft_action_scales_mean": self.action_aft_prior.metrics.get("aft_scales_mean", 0.0),
                "aft_action_scales_std": self.action_aft_prior.metrics.get("aft_scales_std", 0.0),
                "aft_action_loss": action_loss.detach().item(),
            }
        )
        return action_loss, metrics

    # ---- loss ---------------------------------------------------------------
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        base_loss = super().compute_loss(batch)
        vision_active = self._aft_active()
        action_active = self._action_aft_active()
        if not vision_active and not action_active:
            self.aft_metrics = None
            return base_loss

        total_loss = base_loss
        self.aft_metrics = {"aft_base_loss": base_loss.detach().item()}
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        ep_ids, fr_ids = self._batch_episode_frame(batch)

        if vision_active:
            down = self._assemble_aft_vision_feat(batch_size, n_obs_steps)
            if down is not None and ep_ids is not None and fr_ids is not None:
                pre, mask = self._feature_store.lookup(ep_ids, fr_ids)
                n_matched = int(mask.sum().item())
                self.aft_metrics["aft_matched"] = n_matched
                if n_matched >= 2:  # the kernel (Gram) match needs at least 2 matched samples
                    device, dtype = down.device, down.dtype
                    mask = mask.to(device)
                    pre = pre.to(device=device, dtype=dtype)
                    down_m = down[mask]
                    pre_m = pre[mask]

                    if pre_m.shape[1] != self.aft_prior.num_features:
                        logging.warning(
                            "AFT: pre-trained feature dim %d != prior dim %d; skipping AFT term this step. "
                            "Set `aft_pretrained_dim` to match the feature store.",
                            pre_m.shape[1],
                            self.aft_prior.num_features,
                        )
                    else:
                        aft_loss = self.aft_prior.loss(down_m, pre_m, self.config.aft_beta)
                        total_loss = total_loss + aft_loss
                        self.aft_metrics.update(
                            {
                                **self.aft_prior.metrics,
                                "aft_loss": aft_loss.detach().item(),
                            }
                        )
                else:
                    self.aft_metrics["aft_loss"] = 0.0

        action_loss, action_metrics = self._compute_action_aft_loss(batch, ep_ids, fr_ids)
        self.aft_metrics.update(action_metrics)
        if action_loss is not None:
            total_loss = total_loss + action_loss
        self.aft_metrics["aft_total_loss"] = total_loss.detach().item()
        return total_loss


class AFTDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy with vision Adaptive Feature Transfer.

    Behaves exactly like ``DiffusionPolicy`` for inference/action selection; the
    only difference is the extra AFT regularization term in the training loss.
    """

    config_class = AFTDiffusionConfig
    name = "aft_diffusion"

    def __init__(self, config: AFTDiffusionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: AFTDiffusionConfig = config

        # Swap the base diffusion model for the AFT-aware one. The base model built by
        # super().__init__ is discarded (cheap with pretrained_backbone_weights=null).
        self.diffusion = AFTDiffusionModel(config)

        # Build the offline PI0 feature store and attach it (no parameters).
        store: PI0VisionFeatureStore | None = None
        if config.aft_enable and config.aft_feature_dir:
            store = PI0VisionFeatureStore(
                feature_dir=config.aft_feature_dir,
                token_pool=config.aft_token_pool,
                camera_reduce=config.aft_camera_reduce,
                camera_indices=config.aft_camera_indices,
            )
            if not store.is_empty and store.feature_dim != config.aft_pretrained_dim:
                logging.warning(
                    "AFT: feature store dim %d != aft_pretrained_dim %d. The AFT term will be skipped "
                    "until these match. Set --policy.aft_pretrained_dim=%d.",
                    store.feature_dim,
                    config.aft_pretrained_dim,
                    store.feature_dim,
                )
        elif config.aft_enable:
            logging.warning(
                "AFT enabled but `aft_feature_dir` is not set; training will run as plain Diffusion Policy."
            )
        self.diffusion.attach_feature_store(store)

        action_store: PI0ActionFeatureStore | None = None
        if config.action_aft_enable and config.action_aft_feature_dir:
            action_store = PI0ActionFeatureStore(
                feature_dir=config.action_aft_feature_dir,
                layer=config.action_aft_layer,
                timestep=config.action_aft_timestep,
                pool=config.action_aft_pool,
                store_mode=config.action_aft_store_mode,
            )
            if not action_store.is_empty and action_store.feature_dim != config.action_aft_pretrained_dim:
                logging.warning(
                    "Action AFT: feature store dim %d != action_aft_pretrained_dim %d. The action AFT "
                    "term will be skipped until these match. Set --policy.action_aft_pretrained_dim=%d.",
                    action_store.feature_dim,
                    config.action_aft_pretrained_dim,
                    action_store.feature_dim,
                )
        elif config.action_aft_enable:
            logging.warning(
                "Action AFT enabled but `action_aft_feature_dir` is not set; action AFT will be inactive."
            )
        self.diffusion.attach_action_feature_store(action_store)

        # Episode start offsets (episode_index -> global dataset start index) let the
        # model reconstruct frame_index from the batch's `index` after the preprocessor
        # drops `frame_index`. Derived from dataset metadata passed by `make_policy`.
        self.diffusion.set_episode_offsets(self._build_episode_offsets(kwargs.get("dataset_meta")))

        self.reset()

    @staticmethod
    def _build_episode_offsets(dataset_meta) -> dict[int, int] | None:
        """Map episode_index -> dataset_from_index using LeRobot dataset metadata."""
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
        except Exception as e:  # be permissive: AFT simply skips frames it can't key
            logging.warning("AFT: could not build episode offsets from dataset metadata (%s).", e)
            return None

    def get_optim_params(self) -> dict:
        """Include the AFT prior's adaptive scales, optionally at a separate LR."""
        prior = getattr(self.diffusion, "aft_prior", None)
        action_prior = getattr(self.diffusion, "action_aft_prior", None)
        separate_groups: list[tuple[nn.Module, float]] = []
        if prior is not None and self.config.aft_enable and self.config.aft_prior_lr is not None:
            separate_groups.append((prior, self.config.aft_prior_lr))
        if (
            action_prior is not None
            and self.config.action_aft_enable
            and self.config.action_aft_prior_lr is not None
        ):
            separate_groups.append((action_prior, self.config.action_aft_prior_lr))
        if not separate_groups:
            return self.diffusion.parameters()
        separate_param_ids = {id(p) for module, _lr in separate_groups for p in module.parameters()}
        model_params = [p for p in self.diffusion.parameters() if id(p) not in separate_param_ids]
        param_groups = [{"params": model_params}]
        param_groups.extend({"params": list(module.parameters()), "lr": lr} for module, lr in separate_groups)
        return param_groups

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run the base diffusion training step plus the AFT regularizer.

        Returns the total loss and an output dict with AFT logging metrics (or
        ``None`` when AFT is inactive).
        """
        loss, _ = super().forward(batch)
        return loss, getattr(self.diffusion, "aft_metrics", None)
