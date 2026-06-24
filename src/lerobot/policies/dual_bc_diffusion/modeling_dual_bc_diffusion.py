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

"""Dual-BC Diffusion Policy (teacher-feature behavioral-cloning auxiliary loss).

Purely additive: subclasses the stock ``lerobot.policies.diffusion`` model/policy.
The base Diffusion Policy code is untouched, and the PI0 offline feature store is
reused from the AFT extension (``..aft_diffusion.feature_store``).

What it does (vs. AFT)
----------------------
Instead of AFT's kernel-RMSE regularizer, ``compute_loss`` runs the **shared**
diffusion U-Net twice on the *same* noisy action trajectory:

1. **Student BC**: conditioned on ``global_cond`` built from the Diffusion
   Policy's own RGB-encoder features (the stock path) -> ``L_bc_student``.
2. **Teacher BC**: conditioned on ``global_cond`` built by swapping the vision
   block for a learnable projection of the offline PI0 teacher feature
   (dimension-matched to the student's per-step vision block) -> ``L_bc_teacher``.

The total loss is the **naive sum** ``L_bc_student + tbc_lambda * L_bc_teacher``
(no divergence / kernel term between them). The only coupling is the shared U-Net
(FiLM denoiser): training it to denoise the same actions from the teacher's
conditioning distills PI0's task-relevant visual structure into the action
decoder, and pulls the student encoder toward producing compatible conditioning.

Inference is identical to the stock Diffusion Policy (only the student path runs);
the projection is unused at eval but is checkpointed for clean reloads.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ..aft_diffusion.feature_store import PI0VisionFeatureStore
from ..diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from .configuration_dual_bc_diffusion import DualBCDiffusionConfig


class DualBCDiffusionModel(DiffusionModel):
    """Diffusion model with a teacher-feature BC auxiliary loss in ``compute_loss``."""

    def __init__(self, config: DualBCDiffusionConfig):
        super().__init__(config)
        self.config: DualBCDiffusionConfig = config

        # Learnable projection that matches the teacher feature dim to the student's
        # per-step vision conditioning block (the "AFT-style dimension matching").
        # Built unconditionally (when enabled + images present) so checkpoints reload
        # even when the feature dir is absent at eval time.
        self.teacher_proj: nn.Module | None = None
        self._vision_block_dim: int = 0
        # Optional AFT-style diagonal adaptive gate over the teacher dims (sigmoid(s),
        # dimension-preserving). NOT the dimension match — applied *before* the
        # projection. Initialised at 0 -> sigmoid(0)=0.5 (matches AFT).
        self.teacher_scale: nn.Parameter | None = None
        if config.tbc_enable and config.image_features:
            self._vision_block_dim = self._student_vision_block_dim()
            self.teacher_proj = self._build_projection(
                in_dim=config.tbc_pretrained_dim,
                out_dim=self._vision_block_dim,
                hidden=config.tbc_proj_hidden,
            )
            if config.tbc_adaptive_scale:
                self.teacher_scale = nn.Parameter(torch.zeros(config.tbc_pretrained_dim))

        # Offline PI0 feature store (plain attribute, NOT a submodule / no params).
        self._feature_store: PI0VisionFeatureStore | None = None
        # episode_index -> global dataset start index, to reconstruct frame_index from
        # the batch `index` (the diffusion preprocessor drops `frame_index`).
        self._episode_from_index: dict[int, int] | None = None
        # Last-computed teacher-BC logging metrics (surfaced via the policy forward dict).
        self.tbc_metrics: dict | None = None

    # ---- construction helpers ----------------------------------------------
    def _student_vision_block_dim(self) -> int:
        """Size of the per-obs-step vision block inside ``global_cond``.

        Mirrors ``DiffusionModel.__init__``: ``feature_dim * num_cameras`` (the cameras'
        features are concatenated into the conditioning). State / env-state are NOT
        included here — only the vision portion is replaced in the teacher path.
        """
        num_images = len(self.config.image_features)
        enc = self.rgb_encoder[0] if isinstance(self.rgb_encoder, nn.ModuleList) else self.rgb_encoder
        return int(enc.feature_dim) * num_images

    @staticmethod
    def _build_projection(in_dim: int, out_dim: int, hidden: int | None) -> nn.Module:
        if hidden is None or hidden <= 0:
            return nn.Linear(in_dim, out_dim)
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    # ---- plumbing (mirrors the AFT extension) -------------------------------
    def attach_feature_store(self, store: PI0VisionFeatureStore | None) -> None:
        self._feature_store = store

    def set_episode_offsets(self, episode_from_index: dict[int, int] | None) -> None:
        self._episode_from_index = episode_from_index

    def _batch_episode_frame(self, batch: dict[str, Tensor]):
        """Return ``(episode_index, frame_index)`` tensors for teacher matching.

        Uses ``frame_index`` when present; otherwise reconstructs it from the global
        ``index`` and per-episode start offsets (the diffusion preprocessor drops
        ``frame_index`` but keeps ``index`` and ``episode_index``).
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
            offsets = torch.tensor(
                [self._episode_from_index.get(int(e), 1 << 60) for e in ep_flat], dtype=torch.int64
            )
            return ep_flat, idx_flat - offsets
        return None, None

    def _tbc_active(self) -> bool:
        return (
            self.config.tbc_enable
            and bool(self.config.image_features)
            and self.teacher_proj is not None
            and self._feature_store is not None
            and not self._feature_store.is_empty
        )

    # ---- loss ---------------------------------------------------------------
    def _masked_mse(
        self, pred: Tensor, target: Tensor, batch: dict[str, Tensor], sample_mask: Tensor | None = None
    ) -> Tensor:
        """MSE matching the base DP reduction, optionally restricted to a sample subset.

        ``sample_mask`` (B,) keeps only those batch items (used for the teacher path,
        where only frames found in the store contribute). With ``sample_mask=None``
        and padding masking this is identical to ``DiffusionModel.compute_loss``.
        """
        loss = F.mse_loss(pred, target, reduction="none")  # (B, T, A)
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            mask = (~batch["action_is_pad"]).unsqueeze(-1).to(loss.dtype)  # (B, T, 1)
        else:
            mask = torch.ones_like(loss[..., :1])
        if sample_mask is not None:
            mask = mask * sample_mask.view(-1, 1, 1).to(loss.dtype)
        num_valid = mask.sum() * loss.shape[-1]
        return (loss * mask).sum() / num_valid.clamp_min(1)

    def _build_teacher_global_cond(self, batch: dict[str, Tensor], teacher_vision: Tensor) -> Tensor:
        """Build a ``global_cond`` with the same layout as the base, but with the
        vision block replaced by ``teacher_vision`` (B, vision_block_dim).

        Layout matches ``DiffusionModel._prepare_global_conditioning``:
        ``cat([state, vision, (env_state)], dim=-1).flatten(start_dim=1)``.
        """
        state = batch[OBS_STATE]  # (B, S, state_dim)
        batch_size, n_obs_steps = state.shape[:2]
        if self.config.tbc_broadcast_obs_steps:
            tv = teacher_vision.unsqueeze(1).expand(batch_size, n_obs_steps, teacher_vision.shape[-1])
        else:
            if n_obs_steps != 1:
                raise ValueError(
                    "tbc_broadcast_obs_steps=False requires n_obs_steps==1; "
                    f"got n_obs_steps={n_obs_steps}."
                )
            tv = teacher_vision.unsqueeze(1)
        feats = [state, tv]
        if self.config.env_state_feature:
            feats.append(batch[OBS_ENV_STATE])
        return torch.cat(feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        # ----- Student BC loss (stock Diffusion Policy path, shared noise) -----
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond_student = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        pred_student = self.unet(noisy_trajectory, timesteps, global_cond=global_cond_student)
        loss_student = self._masked_mse(pred_student, target, batch)

        self.tbc_metrics = None
        if not self._tbc_active():
            return loss_student

        # ----- Teacher BC loss (projected PI0 feature -> shared U-Net) -----
        ep_ids, fr_ids = self._batch_episode_frame(batch)
        if ep_ids is None or fr_ids is None:
            return loss_student

        pre, mask = self._feature_store.lookup(ep_ids, fr_ids)  # (B, tbc_pretrained_dim), (B,)
        n_matched = int(mask.sum().item())
        base_metrics = {
            "tbc_student_loss": loss_student.detach().item(),
            "tbc_matched": n_matched,
            "tbc_lambda": float(self.config.tbc_lambda),
        }
        if n_matched < 1:
            self.tbc_metrics = {**base_metrics, "tbc_teacher_loss": 0.0, "tbc_loss": loss_student.detach().item()}
            return loss_student

        device, dtype = global_cond_student.device, global_cond_student.dtype
        pre = pre.to(device=device, dtype=dtype)
        mask = mask.to(device)

        if pre.shape[1] != self.config.tbc_pretrained_dim:
            logging.warning(
                "DualBC: teacher feature dim %d != tbc_pretrained_dim %d; skipping teacher term. "
                "Set `tbc_pretrained_dim` to match the feature store.",
                pre.shape[1],
                self.config.tbc_pretrained_dim,
            )
            return loss_student

        # Optional AFT-style adaptive gate over teacher dims (dim-preserving), then project.
        if self.teacher_scale is not None:
            pre = pre * self.teacher_scale.sigmoid()
            base_metrics["tbc_scale_mean"] = self.teacher_scale.sigmoid().mean().item()
            base_metrics["tbc_scale_std"] = self.teacher_scale.sigmoid().std().item()
        teacher_vision = self.teacher_proj(pre)  # (B, vision_block_dim)
        global_cond_teacher = self._build_teacher_global_cond(batch, teacher_vision)

        if self.config.tbc_share_noise:
            noisy_t, ts_t, target_t = noisy_trajectory, timesteps, target
        else:
            eps_t = torch.randn(trajectory.shape, device=trajectory.device)
            ts_t = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(trajectory.shape[0],),
                device=trajectory.device,
            ).long()
            noisy_t = self.noise_scheduler.add_noise(trajectory, eps_t, ts_t)
            target_t = eps_t if self.config.prediction_type == "epsilon" else trajectory

        pred_teacher = self.unet(noisy_t, ts_t, global_cond=global_cond_teacher)
        loss_teacher = self._masked_mse(pred_teacher, target_t, batch, sample_mask=mask)

        total = loss_student + self.config.tbc_lambda * loss_teacher
        self.tbc_metrics = {
            **base_metrics,
            "tbc_teacher_loss": loss_teacher.detach().item(),
            "tbc_loss": total.detach().item(),
        }
        return total


class DualBCDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy with a teacher-feature BC auxiliary loss.

    Inference / action selection is identical to ``DiffusionPolicy``; only the
    training loss adds the teacher-conditioned BC term.
    """

    config_class = DualBCDiffusionConfig
    name = "dual_bc_diffusion"

    def __init__(self, config: DualBCDiffusionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: DualBCDiffusionConfig = config

        # Swap the base diffusion model for the dual-BC one (the base one built by
        # super().__init__ is discarded; cheap with pretrained_backbone_weights=null).
        self.diffusion = DualBCDiffusionModel(config)

        # Build + attach the offline PI0 feature store (no parameters).
        store: PI0VisionFeatureStore | None = None
        if config.tbc_enable and config.tbc_feature_dir:
            store = PI0VisionFeatureStore(
                feature_dir=config.tbc_feature_dir,
                token_pool=config.tbc_token_pool,
                camera_reduce=config.tbc_camera_reduce,
                camera_indices=config.tbc_camera_indices,
            )
            if not store.is_empty and store.feature_dim != config.tbc_pretrained_dim:
                logging.warning(
                    "DualBC: feature store dim %d != tbc_pretrained_dim %d. The teacher term will be "
                    "skipped until these match. Set --policy.tbc_pretrained_dim=%d.",
                    store.feature_dim,
                    config.tbc_pretrained_dim,
                    store.feature_dim,
                )
        elif config.tbc_enable:
            logging.warning(
                "DualBC enabled but `tbc_feature_dir` is not set; training will run as plain Diffusion Policy."
            )
        self.diffusion.attach_feature_store(store)
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
        except Exception as e:  # be permissive: teacher BC simply skips frames it can't key
            logging.warning("DualBC: could not build episode offsets from dataset metadata (%s).", e)
            return None

    def get_optim_params(self) -> dict:
        """Optionally place the teacher adapter (projection + adaptive scale) in a separate LR group."""
        proj = getattr(self.diffusion, "teacher_proj", None)
        if proj is None or not self.config.tbc_enable or self.config.tbc_proj_lr is None:
            return self.diffusion.parameters()
        adapter_params = list(proj.parameters())
        scale = getattr(self.diffusion, "teacher_scale", None)
        if scale is not None:
            adapter_params.append(scale)
        adapter_ids = {id(p) for p in adapter_params}
        model_params = [p for p in self.diffusion.parameters() if id(p) not in adapter_ids]
        return [
            {"params": model_params},
            {"params": adapter_params, "lr": self.config.tbc_proj_lr},
        ]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run the dual-BC training step; return total loss + teacher-BC metrics."""
        loss, _ = super().forward(batch)
        return loss, getattr(self.diffusion, "tbc_metrics", None)
