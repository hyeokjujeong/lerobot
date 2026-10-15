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

Auxiliary objectives
--------------------
``--policy.tbc_objective`` selects which auxiliary term is added. All arms share the
feature store, the keyed teacher frame, the sampled noise/timesteps, the masked-MSE
reduction and ``tbc_lambda``, so a comparison between them isolates the loss alone:

* ``dual_bc``  (default) the second BC pass described above.
* ``pullback`` the branch disagreement ``||eps(a_t,t,c_S) - eps(a_t,t,c_T)||^2``.
  ``dual_bc`` descends an **upper bound** on this quantity
  (``E||D(f_T) - D(f_S)||^2 <= 2 (L_bc(f_S) + L_bc(f_T))`` by the triangle
  inequality); ``pullback`` descends it directly. The gap between the two is logged
  as ``tbc_pullback_bound`` so it can be read off a run.
* ``feature``  naive feature distillation ``||phi_S - P(g)||^2`` — the same adapter
  ``P`` and the same teacher vector as the two arms above, measured with the identity
  metric instead of the decoder pullback metric ``M = J_D^T G J_D``.
* ``kd``       normalized feature KD through a student->teacher head ("x-distill").
* ``aft``      the AFT kernel (Gram) prior, reusing ``..aft_diffusion.aft_prior``.
* ``none``     no auxiliary term (plain Diffusion Policy control).

``dual_bc`` is unchanged by the selector: it is covered by a numerical regression test
(``tests/policies/test_dual_bc_objectives.py``) so runs recorded before the selector
existed stay comparable to new ones.
"""

from __future__ import annotations

import logging

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..aft_diffusion.aft_prior import VisionKernelPrior
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
        # Student -> teacher head, used only by the ``kd`` ("x-distill") objective.
        self.student_kd_proj: nn.Module | None = None
        # AFT kernel prior (owns its own dimension-preserving ``sigmoid(s)`` gates),
        # used only by the ``aft`` objective.
        self.aft_prior: VisionKernelPrior | None = None

        objective = config.tbc_objective
        if config.tbc_enable and config.image_features and objective != "none":
            self._vision_block_dim = self._student_vision_block_dim()
            if objective in ("dual_bc", "pullback", "feature"):
                # teacher -> student adapter, shared by all three so that the arms differ
                # ONLY in the loss applied on top of the very same projected vector.
                self.teacher_proj = self._build_projection(
                    in_dim=config.tbc_pretrained_dim,
                    out_dim=self._vision_block_dim,
                    hidden=config.tbc_proj_hidden,
                )
            elif objective == "kd":
                # Classic distillation direction: lift the student into the teacher's
                # space and match there (the AFT repo's `kd` baseline prior).
                self.student_kd_proj = self._build_projection(
                    in_dim=self._vision_block_dim,
                    out_dim=config.tbc_pretrained_dim,
                    hidden=config.tbc_kd_proj_hidden,
                )
            elif objective == "aft":
                self.aft_prior = VisionKernelPrior(
                    num_features=config.tbc_pretrained_dim,
                    learn_scales=config.tbc_aft_learn_scales,
                    kernel=config.tbc_aft_kernel,
                )
            # AFT's prior already carries its own gates; adding a second one would
            # double-count the adaptive selection, so it is skipped for that arm.
            if config.tbc_adaptive_scale and objective != "aft":
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

    def _has_adapter(self) -> bool:
        """Whether the module required by the selected objective was built."""
        objective = self.config.tbc_objective
        if objective in ("dual_bc", "pullback", "feature"):
            return self.teacher_proj is not None
        if objective == "kd":
            return self.student_kd_proj is not None
        if objective == "aft":
            return self.aft_prior is not None
        return False

    def adapter_parameters(self) -> list[nn.Parameter]:
        """Every teacher-side adapter parameter, for the optional separate LR group."""
        params: list[nn.Parameter] = []
        for module in (self.teacher_proj, self.student_kd_proj, self.aft_prior):
            if module is not None:
                params.extend(module.parameters())
        if self.teacher_scale is not None:
            params.append(self.teacher_scale)
        return params

    def _tbc_active(self) -> bool:
        return (
            self.config.tbc_enable
            and self.config.tbc_objective != "none"
            and bool(self.config.image_features)
            and self._has_adapter()
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

    @staticmethod
    def _masked_sq_distance(pred: Tensor, target: Tensor, sample_mask: Tensor) -> Tensor:
        """Mean over matched rows of the squared distance ``||pred - target||^2``.

        The sum runs over the feature dimension (not a per-element mean), so the term is
        a genuine squared distance: under L2-normalized inputs it equals ``2 (1 - cos)``,
        and it does not shrink by ``1/D`` when the feature is wide. That keeps the
        ``feature`` and ``kd`` arms on a sane ``tbc_lambda`` scale despite their very
        different widths (64 vs 2304 here).
        """
        sq = F.mse_loss(pred, target, reduction="none").sum(dim=-1)  # (B,)
        mask = sample_mask.to(sq.dtype)
        return (sq * mask).sum() / mask.sum().clamp_min(1)

    def _prepare_global_conditioning_split(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        """``_prepare_global_conditioning`` that also returns the per-step image features.

        Numerically identical to ``DiffusionModel._prepare_global_conditioning`` — it is
        duplicated only so the feature-space objectives can read the student's vision
        block without running the RGB encoder a second time.

        Returns:
            ``(global_cond (B, global_cond_dim), img_features (B, n_obs_steps, vision_block_dim))``.
            ``img_features`` is ``None`` when the policy has no image features.
        """
        batch_size, n_obs_steps = self._get_batch_size_and_n_obs_steps(batch)
        global_cond_feats = []
        if self.config.robot_state_feature:
            global_cond_feats.append(batch[OBS_STATE])

        img_features: Tensor | None = None
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1), img_features

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
                    f"tbc_broadcast_obs_steps=False requires n_obs_steps==1; got n_obs_steps={n_obs_steps}."
                )
            tv = teacher_vision.unsqueeze(1)
        feats = [state, tv]
        if self.config.env_state_feature:
            feats.append(batch[OBS_ENV_STATE])
        return torch.cat(feats, dim=-1).flatten(start_dim=1)

    # ---- auxiliary objectives ------------------------------------------------
    def _aux_two_pass(
        self,
        batch: dict[str, Tensor],
        pre: Tensor,
        mask: Tensor,
        trajectory: Tensor,
        noisy_trajectory: Tensor,
        timesteps: Tensor,
        target: Tensor,
        pred_student: Tensor,
        extra: dict,
    ) -> Tensor:
        """``dual_bc`` and ``pullback``: run the shared U-Net a second time on the
        teacher-conditioned input, then either clone the action (``dual_bc``) or match
        the two branches directly (``pullback``).

        With a shared decoder ``D`` and a shared target ``a``, the triangle inequality
        gives ``E||D(f_T) - D(f_S)||^2 <= 2 (L_bc(f_S) + L_bc(f_T))``: ``dual_bc``
        descends that upper bound, ``pullback`` descends the left-hand side itself.
        Both are alignment under the decoder pullback metric ``M = J_D^T G J_D``.
        """
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
        teacher_bc = self._masked_mse(pred_teacher, target_t, batch, sample_mask=mask)
        extra["tbc_teacher_loss"] = teacher_bc.detach().item()

        if self.config.tbc_objective == "dual_bc":
            return teacher_bc

        # pullback: the branch disagreement at the SAME (a_t, t) — config validation
        # guarantees tbc_share_noise=True here, so the comparison is well defined.
        branch_student, branch_teacher = pred_student, pred_teacher
        if self.config.tbc_pullback_detach == "teacher":
            branch_teacher = branch_teacher.detach()
        elif self.config.tbc_pullback_detach == "student":
            branch_student = branch_student.detach()
        disagreement = self._masked_mse(branch_teacher, branch_student, batch, sample_mask=mask)
        # The bound the dual-BC arm descends instead, logged so the gap is observable.
        with torch.no_grad():
            student_bc = self._masked_mse(pred_student, target, batch, sample_mask=mask)
            extra["tbc_pullback_bound"] = 2.0 * (student_bc + teacher_bc).item()
        return disagreement

    def _aux_feature(self, student_vision: Tensor, pre: Tensor, mask: Tensor) -> Tensor:
        """``feature``: naive feature distillation, ``||phi_S - P(g)||^2``.

        Same adapter ``P`` and same teacher vector as ``dual_bc``/``pullback``; the only
        difference is that the mismatch is measured with the identity metric instead of
        the decoder pullback metric. This is the arm that isolates the geometry claim.
        """
        return self._masked_sq_distance(student_vision, self.teacher_proj(pre), mask)

    def _aux_kd(self, student_vision: Tensor, pre: Tensor, mask: Tensor, extra: dict) -> Tensor:
        """``kd`` ("x-distill"): lift the student into the teacher space and match there.

        With ``tbc_kd_normalize=True`` both sides are L2-normalized first, so the term is
        ``2 * (1 - cos)`` — the usual regression-KD form.
        """
        student = self.student_kd_proj(student_vision)  # (B, tbc_pretrained_dim)
        teacher = pre
        if self.config.tbc_kd_normalize:
            student = F.normalize(student, dim=-1)
            teacher = F.normalize(teacher, dim=-1)
        with torch.no_grad():
            cos = F.cosine_similarity(student, teacher, dim=-1)
            sel = mask.to(cos.dtype)
            extra["tbc_kd_cos"] = ((cos * sel).sum() / sel.sum().clamp_min(1)).item()
        return self._masked_sq_distance(student, teacher, mask)

    def _aux_aft(self, student_vision: Tensor, pre: Tensor, mask: Tensor, extra: dict) -> Tensor:
        """``aft``: the AFT kernel (Gram) prior over the matched rows of the batch.

        AFT compares ``B x B`` kernels, so it needs at least two matched rows and never
        maps the two feature dimensions onto each other.
        """
        sel = mask.bool()
        rmse = self.aft_prior.kernel_rmse(student_vision[sel], pre[sel])
        # Re-prefix the prior's own metrics so everything stays under `tbc_*`.
        extra.update({f"tbc_{k}": v for k, v in self.aft_prior.metrics.items()})
        return rmse

    # ---- loss ---------------------------------------------------------------
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        # ----- Student BC loss (stock Diffusion Policy path) -----
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond_student, img_features = self._prepare_global_conditioning_split(batch)

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

        # ----- Teacher lookup (shared by every objective) -----
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
        # The AFT kernel is a Gram matrix over the batch, so it needs >= 2 matched rows.
        min_matched = 2 if self.config.tbc_objective == "aft" else 1
        if n_matched < min_matched:
            self.tbc_metrics = {
                **base_metrics,
                "tbc_aux_loss": 0.0,
                "tbc_teacher_loss": 0.0,
                "tbc_loss": loss_student.detach().item(),
            }
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

        extra: dict = {}
        # Optional AFT-style adaptive gate over the teacher dims (dimension-preserving).
        # Not built for the `aft` arm, whose prior owns an equivalent gate already.
        if self.teacher_scale is not None:
            pre = pre * self.teacher_scale.sigmoid()
            extra["tbc_scale_mean"] = self.teacher_scale.sigmoid().mean().item()
            extra["tbc_scale_std"] = self.teacher_scale.sigmoid().std().item()

        # Student vision block at the observation step the teacher feature is keyed to.
        student_vision = img_features[:, self.config.tbc_obs_step]  # (B, vision_block_dim)

        objective = self.config.tbc_objective
        if objective in ("dual_bc", "pullback"):
            aux = self._aux_two_pass(
                batch,
                pre,
                mask,
                trajectory,
                noisy_trajectory,
                timesteps,
                target,
                pred_student,
                extra,
            )
        elif objective == "feature":
            aux = self._aux_feature(student_vision, pre, mask)
        elif objective == "kd":
            aux = self._aux_kd(student_vision, pre, mask, extra)
        elif objective == "aft":
            aux = self._aux_aft(student_vision, pre, mask, extra)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown tbc_objective {objective!r}")

        total = loss_student + self.config.tbc_lambda * aux
        self.tbc_metrics = {
            **base_metrics,
            **extra,
            "tbc_aux_loss": aux.detach().item(),
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
        if config.tbc_enable and config.tbc_objective == "none":
            logging.info(
                "DualBC: tbc_objective='none' -> training as a plain Diffusion Policy "
                "(control arm; no teacher term, no adapter)."
            )
        elif config.tbc_enable and config.tbc_feature_dir:
            store = PI0VisionFeatureStore(
                feature_dir=config.tbc_feature_dir,
                feature_key=config.tbc_feature_key,
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
        if store is not None and not store.is_empty:
            logging.info(
                "DualBC: objective=%s lambda=%s teacher_dim=%d",
                config.tbc_objective,
                config.tbc_lambda,
                store.feature_dim,
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
        """Optionally place the teacher-side adapters in a separate LR group.

        Covers whichever adapter the selected objective built: the teacher->student
        projection, the KD head, the AFT prior's scales and the optional adaptive gate.
        """
        if not self.config.tbc_enable or self.config.tbc_proj_lr is None:
            return self.diffusion.parameters()
        adapter_params = self.diffusion.adapter_parameters()
        if not adapter_params:
            return self.diffusion.parameters()
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
