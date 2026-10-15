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

"""Configuration for the Dual-BC Diffusion Policy.

``DualBCDiffusionConfig`` is a strict superset of ``DiffusionConfig``: every base
Diffusion Policy field/behavior is inherited unchanged, and only the teacher-BC
options are added. Selecting ``--policy.type=dual_bc_diffusion`` enables a second
behavioral-cloning loss computed from the dimension-matched PI0 teacher feature
(NOT the AFT kernel regularizer).

Idea (vs. AFT)
--------------
AFT (the kernel prior) adds ``beta * kernel_rmse(student_feat, teacher_feat)`` and
**never** matches the two feature dimensions — it compares ``B x B`` kernels, which
are dimension-agnostic; the only learnable teacher-side parameter is a *dimension-
preserving* diagonal adaptive scale ``sigmoid(s)``.

Here, because we feed the teacher feature into the U-Net's **fixed-dim**
conditioning, some dimension map is structurally required. We use a learnable
**projection** mapping the PI0 teacher feature to the size of the Diffusion
Policy's per-step *vision conditioning block*, then feed that as an alternative
``global_cond`` to the **same** U-Net to obtain a second action-prediction (BC)
loss. The total loss is the naive sum::

    L = L_bc(student_vision) + tbc_lambda * L_bc(teacher_vision)

No divergence/kernel term couples the two losses; the only coupling is the shared
U-Net (FiLM denoiser), through which PI0's task-relevant visual structure is
distilled into the action decoder.

NOTE on terminology: the projection is a *feature-distillation* adapter — it
mirrors the AFT repo's ``feature``/``ft`` *baseline* priors
(``Linear(num_features -> feat_dim)``), NOT the AFT (``kernel``) method itself.
Optionally (``tbc_adaptive_scale=True``) we prepend AFT's actual adaptive
mechanism — a learnable diagonal gate ``sigmoid(s)`` over the teacher dims
(dimension-preserving) — *before* the projection, so the model can down-weight
irrelevant teacher dimensions the AFT way and then map what remains into the
conditioning.

Auxiliary objectives (``tbc_objective``)
----------------------------------------
All arms share the **same** feature store, the same keyed teacher frame, the same
sampled noise/timesteps, the same masked-MSE reduction and the same ``tbc_lambda``
knob. Only the auxiliary term changes, so arms are directly comparable::

    L = L_bc(student) + tbc_lambda * L_aux

=============  =======================================================  ===========================
``tbc_objective``  L_aux                                                 feature-space metric
=============  =======================================================  ===========================
``dual_bc``    ``L_bc(P(g))``  (second BC pass through the shared U-Net)  implicit pullback (upper bound)
``pullback``   ``||eps(a_t,t,c_S) - eps(a_t,t,c_T)||^2``                  explicit pullback ``J^T G J``
``feature``    ``||phi_S - P(g)||^2``                                     isotropic (identity)
``kd``         ``||norm(Q(phi_S)) - norm(g)||^2``  (student -> teacher)   isotropic in teacher space
``aft``        ``RMSE(K_S - K_g)``  (AFT kernel prior)                    data-statistics (Gram)
``none``       ``0``  (plain Diffusion Policy; ablation control)          --
=============  =======================================================  ===========================

``dual_bc`` vs ``pullback`` is the upper-bound comparison. With the shared decoder
``D`` and shared target ``a``, the triangle inequality gives::

    E||D(f_T) - D(f_S)||^2  <=  2 * ( L_bc(f_S) + L_bc(f_T) )

so ``dual_bc`` minimizes an **upper bound** on the branch disagreement while
``pullback`` minimizes that disagreement **directly**. Both align the two
representations under the decoder pullback metric ``M = J_D^T G J_D``; they differ
only in whether the bound is tightened explicitly.

.. warning::
   The auxiliary terms have **different natural scales** (a BC-MSE, a feature-MSE, a
   cosine-type term and a Gram RMSE are not commensurate). ``tbc_lambda`` must be
   swept **per objective**; a value tuned for ``dual_bc`` is not meaningful for
   ``aft`` or ``kd``.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

#: Auxiliary objectives selectable via ``--policy.tbc_objective=...``.
#:
#: ``dual_bc``  second BC pass through the shared U-Net (minimizes an upper bound on
#:              the branch disagreement) — the original behaviour, the default.
#: ``pullback`` explicit branch-disagreement matching (minimizes that disagreement
#:              directly, i.e. alignment under the decoder pullback metric).
#: ``feature``  naive feature distillation in the student's feature space.
#: ``kd``       normalized feature KD with a student->teacher head ("x-distill").
#: ``aft``      the AFT kernel (Gram) prior.
#: ``none``     no auxiliary term and no adapter — a plain Diffusion Policy control
#:              reachable from the same CLI as the other arms.
TBC_OBJECTIVES = ("dual_bc", "pullback", "feature", "kd", "aft", "none")

#: Objectives that run a *second* U-Net forward pass (and therefore cost ~2x the
#: forward compute of the feature-space arms).
TBC_TWO_PASS_OBJECTIVES = ("dual_bc", "pullback")


@PreTrainedConfig.register_subclass("dual_bc_diffusion")
@dataclass
class DualBCDiffusionConfig(DiffusionConfig):
    """Diffusion Policy + teacher-feature behavioral-cloning auxiliary loss.

    Teacher-BC args (prefix ``tbc_``):
        tbc_enable: Master switch. If False, behaves exactly like ``DiffusionConfig``
            (the projection is still constructed for checkpoint compatibility but
            contributes no loss).
        tbc_feature_dir: Directory of pre-extracted PI0 ``features.vision_tower``
            shards (``*.pt`` / ``*.safetensors``), e.g.
            ``lerobot/extracted_feature/libero10_task0_pi0base``. If empty/missing,
            the teacher-BC term is inactive (plain Diffusion Policy).
        tbc_feature_key: Feature tensor key to read from each shard. Defaults to
            ``features.vision_tower``. For Cosmos3 G-V-KV0 experiments, use
            ``features.cosmos3_g_v_kv0``.
        tbc_lambda: Weight on the teacher-feature BC loss. Total loss is
            ``L_bc(student) + tbc_lambda * L_bc(teacher)``.
        tbc_pretrained_dim: Dimensionality of the *pooled* PI0 feature produced by
            the feature store (input dim of the projection). Default 2304 =
            2 real cameras x 1152 vision-tower dim with ``camera_reduce='concat'``.
        tbc_camera_indices: Which ``features.vision_tower`` camera slots to keep.
            Default ``(0, 1)`` drops the ``pi0_base`` dummy right-wrist slot.
        tbc_token_pool: Pooling over the PI0 image-token axis (``"mean"``/``"max"``/``"cls"``).
        tbc_camera_reduce: How PI0 per-camera features are combined (``"concat"``/``"mean"``).
        tbc_obs_step: Which Diffusion Policy observation step the (single-frame) PI0
            feature is keyed to. ``-1`` = current step (default).
        tbc_proj_hidden: Hidden width of the projection MLP. ``None`` => a single
            ``Linear(tbc_pretrained_dim -> vision_block_dim)``; an int adds one
            ReLU hidden layer of that width.
        tbc_adaptive_scale: If True, prepend AFT's actual adaptive mechanism — a
            learnable diagonal gate ``sigmoid(s)`` over the ``tbc_pretrained_dim``
            teacher dimensions (dimension-preserving, initialised at
            ``sigmoid(0)=0.5``) — *before* the projection. This is NOT dimension
            matching (the projection still does that); it lets the model select
            which teacher dims to transfer, the AFT way. Default False = plain
            feature-distillation projection.
        tbc_broadcast_obs_steps: If True, the single matched teacher feature is
            broadcast across all ``n_obs_steps`` of the teacher ``global_cond``
            (the student path uses a real per-step feature; the store only holds
            the keyed frame). If False, only ``n_obs_steps==1`` is supported.
        tbc_share_noise: If True, the teacher BC pass reuses the same sampled
            noise / timesteps / noisy trajectory as the student pass (paired
            comparison; recommended). If False, the teacher pass resamples.
        tbc_proj_lr: Optional separate learning rate for the adapter params
            (projection / KD head / AFT scales — separate optimizer group). ``None``
            uses the model learning rate.
        tbc_objective: Which auxiliary loss to add. One of ``"dual_bc"`` (default,
            the original behaviour), ``"pullback"``, ``"feature"``, ``"kd"``,
            ``"aft"``, ``"none"``. See the module docstring for the table.
        tbc_pullback_detach: Only for ``tbc_objective="pullback"``. Which branch of
            the disagreement term is treated as a constant target:
            ``"none"`` (default, both branches get gradient — the apples-to-apples
            setting vs. ``dual_bc``), ``"teacher"`` (teacher branch frozen: align the
            student to the teacher), ``"student"`` (student branch frozen: align the
            teacher adapter to the student).
        tbc_kd_proj_hidden: Only for ``tbc_objective="kd"``. Hidden width of the
            *student -> teacher* KD head ``Q``. ``None`` => a single
            ``Linear(vision_block_dim -> tbc_pretrained_dim)``.
        tbc_kd_normalize: Only for ``tbc_objective="kd"``. L2-normalize both sides
            before the MSE (so the term is ``2 * (1 - cos)``, the standard
            regression-KD form). Default True.
        tbc_aft_kernel: Only for ``tbc_objective="aft"``. ``"linear"`` (cosine Gram)
            or ``"rbf"``.
        tbc_aft_learn_scales: Only for ``tbc_objective="aft"``. Whether AFT's
            per-dimension gates ``sigmoid(s)`` are trainable.
    """

    tbc_enable: bool = True
    tbc_feature_dir: str | None = None
    tbc_feature_key: str = "features.vision_tower"
    tbc_lambda: float = 1.0
    tbc_pretrained_dim: int = 2304
    tbc_camera_indices: tuple[int, ...] | None = (0, 1)
    tbc_token_pool: str = "mean"
    tbc_camera_reduce: str = "concat"
    tbc_obs_step: int = -1
    tbc_proj_hidden: int | None = None
    tbc_adaptive_scale: bool = False
    tbc_broadcast_obs_steps: bool = True
    tbc_share_noise: bool = True
    tbc_proj_lr: float | None = None
    # --- auxiliary-objective selector (all arms share everything else) ---
    tbc_objective: str = "dual_bc"
    tbc_pullback_detach: str = "none"
    tbc_kd_proj_hidden: int | None = None
    tbc_kd_normalize: bool = True
    tbc_aft_kernel: str = "linear"
    tbc_aft_learn_scales: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.tbc_token_pool not in ("mean", "max", "cls"):
            raise ValueError(f"`tbc_token_pool` must be 'mean', 'max', or 'cls'. Got {self.tbc_token_pool}.")
        if self.tbc_camera_reduce not in ("concat", "mean"):
            raise ValueError(f"`tbc_camera_reduce` must be 'concat' or 'mean'. Got {self.tbc_camera_reduce}.")
        if self.tbc_enable and self.tbc_pretrained_dim <= 0:
            raise ValueError(f"`tbc_pretrained_dim` must be a positive int. Got {self.tbc_pretrained_dim}.")
        if self.tbc_lambda < 0:
            raise ValueError(f"`tbc_lambda` must be >= 0. Got {self.tbc_lambda}.")
        if self.tbc_objective not in TBC_OBJECTIVES:
            raise ValueError(
                f"`tbc_objective` must be one of {sorted(TBC_OBJECTIVES)}. Got {self.tbc_objective!r}."
            )
        if self.tbc_pullback_detach not in ("none", "student", "teacher"):
            raise ValueError(
                "`tbc_pullback_detach` must be 'none', 'student' or 'teacher'. "
                f"Got {self.tbc_pullback_detach!r}."
            )
        if self.tbc_aft_kernel not in ("linear", "rbf"):
            raise ValueError(f"`tbc_aft_kernel` must be 'linear' or 'rbf'. Got {self.tbc_aft_kernel!r}.")
        if self.tbc_objective == "pullback" and not self.tbc_share_noise:
            # ||eps(a_t, t, c_S) - eps(a_t, t, c_T)||^2 is only defined when both
            # branches see the same noisy trajectory and timestep.
            raise ValueError(
                "`tbc_objective='pullback'` requires `tbc_share_noise=True`: the branch "
                "disagreement is only defined at a shared (noisy_trajectory, timestep)."
            )
