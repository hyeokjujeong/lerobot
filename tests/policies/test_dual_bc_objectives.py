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

"""Tests for the ``dual_bc_diffusion`` auxiliary-objective arms.

These cover the properties the LIBERO comparison depends on:

* every arm produces a finite loss and reaches both the RGB encoder and its adapter;
* ``dual_bc`` is **numerically unchanged** by the objective-selector refactor, so runs
  recorded before it remain comparable to new ones;
* ``_prepare_global_conditioning_split`` matches the base method it duplicates;
* the ``pullback`` arm stays under the ``dual_bc`` bound (the triangle inequality the
  two arms are meant to separate) and its detach modes route gradients as documented.

No dataset or feature shard is needed: the PI0 feature store is faked.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.dual_bc_diffusion.configuration_dual_bc_diffusion import (
    TBC_OBJECTIVES,
    DualBCDiffusionConfig,
)
from lerobot.policies.dual_bc_diffusion.modeling_dual_bc_diffusion import DualBCDiffusionModel
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

BATCH, N_OBS, HORIZON, ACTION_DIM, STATE_DIM, PRETRAINED_DIM = 6, 2, 16, 7, 8, 32
N_MATCHED = 4


class _FakeFeatureStore:
    """Stand-in for ``PI0VisionFeatureStore`` matching the first ``N_MATCHED`` rows."""

    is_empty = False
    feature_dim = PRETRAINED_DIM

    def __init__(self, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(BATCH, PRETRAINED_DIM, generator=generator)

    def lookup(self, episode_index, frame_index):
        mask = torch.zeros(BATCH)
        mask[:N_MATCHED] = 1.0
        return self.features.clone(), mask


def _make_config(**kwargs) -> DualBCDiffusionConfig:
    kwargs.setdefault("n_obs_steps", N_OBS)
    kwargs.setdefault("horizon", HORIZON)
    kwargs.setdefault("n_action_steps", 8)
    kwargs.setdefault("crop_shape", None)
    kwargs.setdefault("pretrained_backbone_weights", None)
    kwargs.setdefault("tbc_pretrained_dim", PRETRAINED_DIM)
    kwargs.setdefault("tbc_feature_dir", "unused-but-non-empty")
    config = DualBCDiffusionConfig(**kwargs)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return config


def _make_batch(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        OBS_STATE: torch.randn(BATCH, N_OBS, STATE_DIM, generator=generator),
        OBS_IMAGES: torch.rand(BATCH, N_OBS, 1, 3, 96, 96, generator=generator),
        ACTION: torch.randn(BATCH, HORIZON, ACTION_DIM, generator=generator),
        "action_is_pad": torch.zeros(BATCH, HORIZON, dtype=torch.bool),
        "episode_index": torch.zeros(BATCH, dtype=torch.long),
        "frame_index": torch.arange(BATCH, dtype=torch.long),
    }


def _build(objective: str, **kwargs) -> DualBCDiffusionModel:
    torch.manual_seed(1234)
    model = DualBCDiffusionModel(_make_config(tbc_objective=objective, **kwargs))
    model.attach_feature_store(_FakeFeatureStore())
    return model


def _encoder_grad_norm(model: DualBCDiffusionModel) -> float:
    encoder = (
        model.rgb_encoder[0] if isinstance(model.rgb_encoder, torch.nn.ModuleList) else model.rgb_encoder
    )
    return sum(p.grad.norm().item() ** 2 for p in encoder.parameters() if p.grad is not None) ** 0.5


@pytest.mark.parametrize("objective", TBC_OBJECTIVES)
def test_objective_trains(objective):
    """Each arm yields a finite loss whose gradient reaches the encoder and the adapter."""
    model = _build(objective, tbc_lambda=0.5)
    loss = model.compute_loss(_make_batch())
    assert torch.isfinite(loss)
    loss.backward()

    assert _encoder_grad_norm(model) > 0, f"{objective}: no gradient reached the RGB encoder"

    adapter_params = model.adapter_parameters()
    if objective == "none":
        assert adapter_params == []
        assert model.tbc_metrics is None
        return
    assert adapter_params, f"{objective}: no adapter was built"
    adapter_grad = sum(p.grad.norm().item() ** 2 for p in adapter_params if p.grad is not None) ** 0.5
    assert adapter_grad > 0, f"{objective}: no gradient reached the adapter"
    assert model.tbc_metrics["tbc_matched"] == N_MATCHED


def test_dual_bc_is_unchanged_by_the_refactor():
    """``dual_bc`` must match the pre-refactor loss exactly, or old runs stop comparing."""
    model = _build("dual_bc", tbc_lambda=0.5).eval()
    batch = _make_batch()

    def pre_refactor_loss() -> torch.Tensor:
        global_cond = model._prepare_global_conditioning(batch)
        trajectory = batch[ACTION]
        torch.manual_seed(77)
        eps = torch.randn(trajectory.shape)
        timesteps = torch.randint(
            0, model.noise_scheduler.config.num_train_timesteps, (trajectory.shape[0],)
        ).long()
        noisy = model.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred_student = model.unet(noisy, timesteps, global_cond=global_cond)
        loss_student = model._masked_mse(pred_student, eps, batch)

        pre, mask = model._feature_store.lookup(None, None)
        teacher_cond = model._build_teacher_global_cond(batch, model.teacher_proj(pre))
        pred_teacher = model.unet(noisy, timesteps, global_cond=teacher_cond)
        loss_teacher = model._masked_mse(pred_teacher, eps, batch, sample_mask=mask)
        return loss_student + model.config.tbc_lambda * loss_teacher

    with torch.no_grad():
        torch.manual_seed(77)
        refactored = model.compute_loss(batch)
        original = pre_refactor_loss()
    torch.testing.assert_close(refactored, original, rtol=0, atol=1e-6)


def test_split_conditioning_matches_base():
    """The duplicated conditioning helper must stay numerically identical to the base."""
    model = _build("feature").eval()
    batch = _make_batch()
    with torch.no_grad():
        base = model._prepare_global_conditioning(batch)
        split, img_features = model._prepare_global_conditioning_split(batch)
    torch.testing.assert_close(base, split, rtol=0, atol=1e-6)
    assert img_features.shape[:2] == (BATCH, N_OBS)


@pytest.mark.parametrize("detach", ["none", "teacher", "student"])
def test_pullback_stays_under_the_dual_bc_bound(detach):
    """``E||D(f_T) - D(f_S)||^2 <= 2 (L_bc(f_S) + L_bc(f_T))`` — the bound dual-BC descends."""
    model = _build("pullback", tbc_lambda=0.5, tbc_pullback_detach=detach)
    model.compute_loss(_make_batch()).backward()
    metrics = model.tbc_metrics
    assert metrics["tbc_aux_loss"] <= metrics["tbc_pullback_bound"] + 1e-6

    proj_grad = (
        sum(p.grad.norm().item() ** 2 for p in model.teacher_proj.parameters() if p.grad is not None) ** 0.5
    )
    if detach == "teacher":
        # The teacher branch is a frozen target, so only the student path is updated.
        assert proj_grad == pytest.approx(0.0, abs=1e-9)
    else:
        assert proj_grad > 0


def test_pullback_requires_shared_noise():
    """The disagreement is undefined unless both branches see the same (a_t, t)."""
    with pytest.raises(ValueError, match="tbc_share_noise"):
        _make_config(tbc_objective="pullback", tbc_share_noise=False)


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="tbc_objective"):
        _make_config(tbc_objective="not-an-objective")


# --------------------------------------------------------------------------------------
# Option / plumbing coverage: the paths the arm smoke test above does not exercise.
# --------------------------------------------------------------------------------------


def _build_policy(objective: str, **kwargs):
    from lerobot.policies.dual_bc_diffusion.modeling_dual_bc_diffusion import DualBCDiffusionPolicy

    torch.manual_seed(1234)
    config = _make_config(tbc_objective=objective, push_to_hub=False, **kwargs)
    policy = DualBCDiffusionPolicy(config)
    policy.diffusion.attach_feature_store(_FakeFeatureStore())
    return policy


@pytest.mark.parametrize("objective", ["dual_bc", "pullback", "feature", "kd", "aft"])
def test_separate_adapter_lr_group_covers_every_arm(objective):
    """``tbc_proj_lr`` must put *this* arm's adapter in the second group, whatever it is."""
    policy = _build_policy(objective, tbc_proj_lr=1e-3)
    groups = policy.get_optim_params()
    assert isinstance(groups, list) and len(groups) == 2, f"{objective}: no separate LR group"
    assert groups[1]["lr"] == 1e-3

    adapter_ids = {id(p) for p in groups[1]["params"]}
    assert adapter_ids == {id(p) for p in policy.diffusion.adapter_parameters()}
    assert adapter_ids, f"{objective}: adapter group is empty"
    # The two groups must partition the parameters: no duplicates, nothing dropped.
    model_ids = {id(p) for p in groups[0]["params"]}
    assert not (model_ids & adapter_ids)
    assert len(model_ids) + len(adapter_ids) == len(list(policy.diffusion.parameters()))


def test_no_separate_lr_group_without_proj_lr():
    policy = _build_policy("pullback")
    assert not isinstance(policy.get_optim_params(), list)


@pytest.mark.parametrize("objective", TBC_OBJECTIVES)
def test_checkpoint_roundtrip(objective, tmp_path):
    """Every arm must save and reload: adapters are checkpointed, eval must still run."""
    policy = _build_policy(objective)
    policy.save_pretrained(tmp_path / objective)

    from lerobot.policies.dual_bc_diffusion.modeling_dual_bc_diffusion import DualBCDiffusionPolicy

    reloaded = DualBCDiffusionPolicy.from_pretrained(tmp_path / objective)
    assert reloaded.config.tbc_objective == objective

    # `from_pretrained` moves the policy to `config.device`, so compare on CPU.
    original_state = {k: v.cpu() for k, v in policy.state_dict().items()}
    reloaded_state = {k: v.cpu() for k, v in reloaded.state_dict().items()}
    assert original_state.keys() == reloaded_state.keys(), f"{objective}: state_dict keys changed"
    for key, value in reloaded_state.items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0, msg=f"{objective}/{key}")


@pytest.mark.parametrize("objective", TBC_OBJECTIVES)
def test_inference_runs_without_the_feature_store(objective):
    """Eval only runs the student path, so it must work with no teacher features attached."""
    policy = _build_policy(objective)
    policy.diffusion.attach_feature_store(None)
    policy.eval()
    policy.reset()
    # `select_action` takes ONE timestep and keeps its own n_obs_steps queue, and it
    # stacks the per-camera keys itself — so pass a single frame per camera key.
    observation = {
        OBS_STATE: torch.randn(1, STATE_DIM),
        "observation.images.cam": torch.rand(1, 3, 96, 96),
    }
    with torch.no_grad():
        action = policy.select_action(observation)
    assert action.shape == (1, ACTION_DIM)
    assert torch.isfinite(action).all()


@pytest.mark.parametrize("separate_encoder", [True, False])
def test_split_conditioning_matches_base_for_both_encoder_layouts(separate_encoder):
    """``_prepare_global_conditioning_split`` duplicates both encoder branches."""
    model = _build("feature", use_separate_rgb_encoder_per_camera=separate_encoder).eval()
    batch = _make_batch()
    with torch.no_grad():
        base = model._prepare_global_conditioning(batch)
        split, img_features = model._prepare_global_conditioning_split(batch)
    torch.testing.assert_close(base, split, rtol=0, atol=1e-6)
    assert img_features.shape[:2] == (BATCH, N_OBS)


@pytest.mark.parametrize(
    "objective,kwargs",
    [
        ("dual_bc", {"tbc_adaptive_scale": True}),
        ("pullback", {"tbc_adaptive_scale": True}),
        ("feature", {"tbc_adaptive_scale": True}),
        ("kd", {"tbc_adaptive_scale": True}),
        ("dual_bc", {"tbc_proj_hidden": 16}),
        ("kd", {"tbc_kd_proj_hidden": 16}),
        ("kd", {"tbc_kd_normalize": False}),
        ("aft", {"tbc_aft_kernel": "rbf"}),
        ("aft", {"tbc_aft_learn_scales": False}),
        ("dual_bc", {"tbc_share_noise": False}),
        ("dual_bc", {"tbc_broadcast_obs_steps": False, "n_obs_steps": 1}),
    ],
)
def test_option_combinations_train(objective, kwargs):
    """Each documented knob must actually run on the arm it applies to."""
    model = _build(objective, tbc_lambda=0.5, **kwargs)
    batch = _make_batch()
    if kwargs.get("n_obs_steps") == 1:
        batch[OBS_STATE] = batch[OBS_STATE][:, :1]
        batch[OBS_IMAGES] = batch[OBS_IMAGES][:, :1]
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert _encoder_grad_norm(model) > 0
    trainable = [p for p in model.adapter_parameters() if p.requires_grad]
    if trainable:
        adapter_grad = sum(p.grad.norm().item() ** 2 for p in trainable if p.grad is not None) ** 0.5
        assert adapter_grad > 0
    else:
        # `aft` with tbc_aft_learn_scales=False has no trainable adapter, by design.
        assert kwargs.get("tbc_aft_learn_scales") is False


def test_adaptive_scale_is_skipped_for_aft():
    """AFT's prior owns an equivalent gate; a second one would double-count it."""
    model = _build("aft", tbc_adaptive_scale=True)
    assert model.teacher_scale is None
    assert model.aft_prior is not None


def test_unmatched_batch_falls_back_to_student_loss():
    """A batch the feature store keys nothing in must not poison the step."""

    class _EmptyMatchStore(_FakeFeatureStore):
        def lookup(self, episode_index, frame_index):
            return self.features.clone(), torch.zeros(BATCH)

    for objective in ("dual_bc", "pullback", "feature", "kd", "aft"):
        model = _build(objective, tbc_lambda=0.5)
        model.attach_feature_store(_EmptyMatchStore())
        loss = model.compute_loss(_make_batch())
        assert torch.isfinite(loss)
        assert model.tbc_metrics["tbc_aux_loss"] == 0.0
