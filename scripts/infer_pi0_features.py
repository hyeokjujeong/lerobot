#!/usr/bin/env python
"""Run offline PI0 inference on selected LIBERO dataset episodes and save features.

This script reads frames from a LeRobot dataset, runs a pretrained PI0 policy,
and stores selected internal activations.

Example:
    python scripts/infer_libero_pi0_features.py \
        --policy-path lerobot/pi0_libero_base \
        --dataset-repo-id yzembodied/libero_10_image_task_0 \
        --dataset-root /PublicSSD/jhri626/datasets/libero_10_image_task_0 \
        --episodes 0:15 \
        --output-path /PublicSSD/jhri626/outputs/pi0_features/libero_task0_ep0_14.safetensors
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pi0 import modeling_pi0
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


DEFAULT_TARGETS = (
    "vision_tower",
    "paligemma_last",
)
AVAILABLE_TARGETS = (
    "vision_tower",
    "paligemma_last",
    "expert_last",
    "action_out_proj",
)
DEFAULT_DP_ACTION_SOURCE_CHECKPOINT = (
    "/PublicSSD/jhri626/outputs/libero10_task0_diffusion_obs2/"
    "checkpoints/080000/pretrained_model"
)


def parse_episodes(value: str) -> list[int] | None:
    """Parse comma-separated episodes, a half-open range like 0:15, or all."""
    value = value.strip()
    if value.lower() == "all":
        return None

    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Range syntax must be START:END, e.g. 0:15")
        start, end = (int(part) for part in parts)
        return list(range(start, end))

    episodes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not episodes:
        raise argparse.ArgumentTypeError("At least one episode is required")
    return episodes


def parse_targets(value: str) -> list[str]:
    targets = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(targets) - set(AVAILABLE_TARGETS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown feature target(s): {unknown}. Available: {list(AVAILABLE_TARGETS)}"
        )
    return targets


def parse_timesteps(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    timesteps: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(piece) for piece in part.split(":")]
            if len(pieces) == 2:
                start, end = pieces
                step = 1
            elif len(pieces) == 3:
                start, end, step = pieces
            else:
                raise argparse.ArgumentTypeError("Timestep range must be START:END or START:END:STEP.")
            timesteps.extend(range(start, end, step))
        else:
            timesteps.append(int(part))
    return sorted(set(timesteps))


def parse_key_map(value: str) -> dict[str, str]:
    """Parse comma-separated SRC=DST key mappings."""
    value = value.strip()
    if not value:
        return {}

    mapping: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "Key map entries must use SRC=DST syntax, e.g. observation.images.image=observation.images.base_0_rgb"
            )
        src, dst = (part.strip() for part in item.split("=", 1))
        if not src or not dst:
            raise argparse.ArgumentTypeError("Key map entries must have non-empty SRC and DST")
        mapping[src] = dst
    return mapping


def add_mapped_batch_keys(batch: dict[str, Any], key_map: dict[str, str]) -> dict[str, Any]:
    """Add destination keys expected by a policy while preserving the original dataset keys."""
    if not key_map:
        return batch

    mapped = dict(batch)
    for src, dst in key_map.items():
        if src in batch:
            mapped[dst] = batch[src]
    return mapped


def select_current_observation_for_pi0(batch: dict[str, Any]) -> dict[str, Any]:
    """PI0 expects one current observation; DP action-source batches may contain n_obs_steps."""
    current = dict(batch)
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("observation.images.") and value.ndim == 5:
            current[key] = value[:, -1]
        elif key == OBS_STATE and value.ndim == 3:
            current[key] = value[:, -1]
    return current


def register_pi0_base_processor_aliases() -> None:
    """Register compatibility aliases used by older PI0 processor configs."""
    if "relative_actions_processor" not in ProcessorStepRegistry.list():
        ProcessorStepRegistry.register("relative_actions_processor")(RelativeActionsProcessorStep)


def first_tensor(output: Any) -> Tensor | None:
    """Best-effort extraction of the main tensor from common module outputs."""
    if torch.is_tensor(output):
        return output

    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state

    if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
        return output.pooler_output

    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor

    return None


def cast_feature(tensor: Tensor, dtype: str) -> Tensor:
    if dtype == "float16":
        return tensor.to(torch.float16)
    if dtype == "bfloat16":
        return tensor.to(torch.bfloat16)
    return tensor.to(torch.float32)


def reduce_feature(tensor: Tensor, mode: str, dtype: str) -> Tensor:
    """Move a captured activation to CPU, optionally mean-pooling token/time dimensions."""
    tensor = tensor.detach().float().cpu()
    if mode == "full":
        return cast_feature(tensor, dtype)

    # "pooled" is a frame-level summary for lightweight alignment probes.
    # It averages the token/time dimension, e.g. [B, tokens, D] -> [B, D].
    # Use --feature-mode full when token-level or patch-level structure matters.
    #
    # Common activation shapes:
    #   [batch, tokens, hidden] -> [batch, hidden]
    #   [batch, chunk, action_dim] -> [batch, action_dim]
    if tensor.ndim >= 3:
        tensor = tensor.mean(dim=1)
    return cast_feature(tensor, dtype)


def reduce_action_tokens(tensor: Tensor, horizon: int, mode: str, dtype: str) -> Tensor:
    """Reduce or keep PI0 action-token activations over the DP horizon."""
    tensor = tensor.detach().float()
    action_tokens = tensor[:, :horizon]
    if mode == "pooled":
        action_tokens = action_tokens.mean(dim=1)
    return cast_feature(action_tokens.cpu(), dtype)


def make_noise_like(reference: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    return torch.randn(reference.shape, dtype=reference.dtype, device=reference.device, generator=generator)


def build_pi0_padded_actions(pi0_policy, dp_actions: Tensor, dp_eps: Tensor) -> tuple[Tensor, Tensor]:
    batch_size, horizon, action_dim = dp_actions.shape
    chunk_size = pi0_policy.config.chunk_size
    max_action_dim = pi0_policy.config.max_action_dim
    actions = torch.zeros(batch_size, chunk_size, max_action_dim, device=dp_actions.device, dtype=dp_actions.dtype)
    noise = torch.zeros_like(actions)
    actions[:, :horizon, :action_dim] = dp_actions
    noise[:, :horizon, :action_dim] = dp_eps
    return actions, noise


@torch.inference_mode()
def build_pi0_prefix_cache(pi0_policy, batch: dict[str, Tensor]):
    images, img_masks = pi0_policy._preprocess_images(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    state = pi0_policy.prepare_state(batch)
    prefix_embs, prefix_pad_masks, prefix_att_masks = pi0_policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = modeling_pi0.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = pi0_policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)
    pi0_policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, past_key_values = pi0_policy.model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, past_key_values, state


@torch.inference_mode()
def collect_pi0_action_layer_feature(
    pi0_policy,
    prefix_pad_masks: Tensor,
    past_key_values,
    state: Tensor,
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    layer_index: int,
    horizon: int,
    mode: str,
    dtype: str,
) -> Tensor:
    feature_cache: dict[str, Tensor] = {}
    layer = pi0_policy.model.paligemma_with_expert.gemma_expert.model.layers[layer_index]

    def hook(_module, _inputs, output):
        tensor = first_tensor(output)
        if tensor is not None:
            action_tokens = tensor[:, -pi0_policy.config.chunk_size :]
            feature_cache["feature"] = reduce_action_tokens(action_tokens, horizon, mode, dtype)

    handle = layer.register_forward_hook(hook)
    try:
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pi0_policy.model.embed_suffix(
            state, x_t, time
        )
        if (
            pi0_policy.model.paligemma_with_expert.paligemma.model.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = modeling_pi0.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = pi0_policy.model._prepare_attention_masks_4d(full_att_2d_masks)
        pi0_policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        _ = pi0_policy.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
    finally:
        handle.remove()

    if "feature" not in feature_cache:
        raise RuntimeError(f"Failed to capture PI0 action expert layer {layer_index}.")
    return feature_cache["feature"]


def stack_feature_list(values: list[Tensor]) -> Tensor:
    """Stack feature tensors, falling back to an object list only if shapes differ."""
    try:
        return torch.stack(values)
    except RuntimeError:
        return values


def build_feature_hooks(policy: torch.nn.Module, targets: Iterable[str], mode: str, dtype: str):
    """Register forward hooks and return (feature_cache, handles)."""
    feature_cache: dict[str, list[Tensor]] = {target: [] for target in targets}
    modules = {
        # The PI0 prefix embeds image features camera-by-camera in config.image_features order.
        # For yzembodied/libero_10_image_task_0 with the pi0_libero_base camera names this means:
        #   vision_tower[0] -> observation.images.image
        #   vision_tower[1] -> observation.images.wrist_image
        # For pi0_base with --image-key-map, the same frames are copied to:
        #   vision_tower[0] -> observation.images.base_0_rgb
        #   vision_tower[1] -> observation.images.left_wrist_0_rgb
        #   vision_tower[2] -> observation.images.right_wrist_0_rgb dummy image, mask=False
        # when batch_size=1 and --feature-mode pooled/full both preserve call order.
        "vision_tower": policy.model.paligemma_with_expert.paligemma.model.vision_tower,
        "paligemma_last": policy.model.paligemma_with_expert.paligemma.model.language_model.layers[-1],
        "expert_last": policy.model.paligemma_with_expert.gemma_expert.model.layers[-1],
        "action_out_proj": policy.model.action_out_proj,
    }

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is not None:
                feature_cache[name].append(reduce_feature(tensor, mode, dtype))

        return hook

    handles = [modules[target].register_forward_hook(make_hook(target)) for target in targets]
    return feature_cache, handles


def tensor_item(batch: dict[str, Any], key: str) -> int | float | list | None:
    value = batch.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        flattened = value.detach().cpu().flatten()
        if flattened.numel() == 1:
            return flattened.item()
        return flattened.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="lerobot/pi0_libero_base")
    parser.add_argument("--dataset-repo-id", default="yzembodied/libero_10_image_task_0")
    parser.add_argument("--dataset-root", default="/PublicSSD/jhri626/datasets/libero_10_image_task_0")
    parser.add_argument("--episodes", type=parse_episodes, default=parse_episodes("0:15"))
    parser.add_argument(
        "--output-path",
        default="/PublicSSD/jhri626/outputs/pi0_features/libero10_task0_ep0_14_features_full_fp16.safetensors",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--feature-targets", type=parse_targets, default=list(DEFAULT_TARGETS))
    parser.add_argument(
        "--feature-mode",
        choices=("pooled", "full"),
        default="full",
        help=(
            "full stores raw activations for patch/token-level analysis; "
            "pooled averages token/time dimensions for compact frame-level features."
        ),
    )
    parser.add_argument(
        "--save-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Activation dtype to store on disk. float16 roughly halves full-feature file size.",
    )
    parser.add_argument(
        "--shard-by",
        choices=("none", "episode"),
        default="episode",
        help="Save one output file per episode, or one combined output file.",
    )
    parser.add_argument(
        "--output-format",
        choices=("safetensors", "pt"),
        default="safetensors",
        help="safetensors stores stacked tensor banks; pt stores the original list-of-dicts rows.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick inspection before running all selected episodes.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Write a checkpoint every N batches. Use 0 to save only at the end.",
    )
    parser.add_argument(
        "--image-key-map",
        type=parse_key_map,
        default={},
        help=(
            "Comma-separated SRC=DST mappings added to each batch before preprocessing. "
            "Useful when a base policy expects different camera names."
        ),
    )
    parser.add_argument(
        "--pi0-chunk-size",
        type=int,
        default=None,
        help="Override PI0 chunk_size. Use 16 for DP-horizon-matched action expert features.",
    )
    parser.add_argument(
        "--action-feature-timesteps",
        type=parse_timesteps,
        default=[],
        help="If set, extract PI0 action-expert features at these DP diffusion timesteps, e.g. '95' or '94:98'.",
    )
    parser.add_argument(
        "--action-feature-layer",
        type=int,
        default=17,
        help="PI0 action expert layer index to save in action-feature mode. Default: 17.",
    )
    parser.add_argument(
        "--action-feature-mode",
        choices=("pooled", "full"),
        default="pooled",
        help="pooled stores [D] per row; full stores raw action tokens [horizon, D] per row.",
    )
    parser.add_argument(
        "--action-seed-repeats",
        type=int,
        default=1,
        help="Number of deterministic noise repeats per frame in action-feature mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=100_000)
    parser.add_argument(
        "--action-source-policy-checkpoint",
        default=DEFAULT_DP_ACTION_SOURCE_CHECKPOINT,
        help=(
            "Diffusion Policy checkpoint whose preprocessing/delta timestamps define the action horizon and "
            "normalization for action-feature extraction."
        ),
    )
    return parser


def episode_shard_path(output_path: Path, episode_index: int) -> Path:
    return output_path.with_name(f"{output_path.stem}_episode_{episode_index:06d}{output_path.suffix}")


def save_rows(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if args.output_format == "pt":
        torch.save({"args": vars(args), "rows": rows}, path)
        return

    tensors = rows_to_tensor_dict(rows)
    metadata = {
        "args": json.dumps(vars(args), sort_keys=True),
        "num_rows": str(len(rows)),
        "format": "pi0_feature_episode_shard_v1",
    }
    save_file(tensors, path, metadata=metadata)


def stack_row_tensor(rows: list[dict[str, Any]], key: str) -> Tensor | None:
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return torch.stack([value.detach().cpu() for value in values]).contiguous()


def rows_to_tensor_dict(rows: list[dict[str, Any]]) -> dict[str, Tensor]:
    if not rows:
        return {}

    tensor_dict: dict[str, Tensor] = {
        "batch_index": torch.tensor([row["batch_index"] for row in rows], dtype=torch.int64),
        "episode_index": torch.tensor([row["episode_index"] for row in rows], dtype=torch.int64),
        "frame_index": torch.tensor([row["frame_index"] for row in rows], dtype=torch.int64),
        "timestamp": torch.tensor([row["timestamp"] for row in rows], dtype=torch.float32),
        "task_index": torch.tensor(
            [-1 if row["task_index"] is None else row["task_index"] for row in rows], dtype=torch.int64
        ),
    }
    if "diffusion_timestep" in rows[0]:
        tensor_dict["diffusion_timestep"] = torch.tensor(
            [row["diffusion_timestep"] for row in rows], dtype=torch.int64
        )
    if "seed_repeat" in rows[0]:
        tensor_dict["seed_repeat"] = torch.tensor([row["seed_repeat"] for row in rows], dtype=torch.int64)
    if "noise_seed" in rows[0]:
        tensor_dict["noise_seed"] = torch.tensor([row["noise_seed"] for row in rows], dtype=torch.int64)

    feature_names = rows[0]["internal_features"].keys()
    for feature_name in feature_names:
        values = [row["internal_features"][feature_name] for row in rows]
        if not all(torch.is_tensor(value) for value in values):
            raise ValueError(
                f"Feature '{feature_name}' had inconsistent shapes and could not be stacked for safetensors."
            )
        tensor_dict[f"features.{feature_name}"] = torch.stack(values).contiguous()

    return tensor_dict


def main() -> None:
    args = make_argparser().parse_args()
    register_pi0_base_processor_aliases()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.shard_by == "episode" and args.batch_size != 1:
        raise ValueError("--shard-by episode currently expects --batch-size 1 to preserve episode boundaries.")
    action_feature_mode = bool(args.action_feature_timesteps)
    if action_feature_mode and args.action_seed_repeats < 1:
        raise ValueError("--action-seed-repeats must be >= 1.")

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = args.device
    if args.pi0_chunk_size is not None:
        cfg.chunk_size = args.pi0_chunk_size
        cfg.n_action_steps = min(cfg.n_action_steps, args.pi0_chunk_size)

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    action_source_cfg = None
    if action_feature_mode:
        action_source_cfg = PreTrainedConfig.from_pretrained(args.action_source_policy_checkpoint)
        action_source_cfg.pretrained_path = args.action_source_policy_checkpoint
        action_source_cfg.device = args.device
        delta_timestamps = resolve_delta_timestamps(action_source_cfg, ds_meta)
    else:
        delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=args.episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    policy_cls = get_policy_class(cfg.type)
    # Some PI0 checkpoints include legacy normalization buffers in the state dict
    # that are not module parameters in the current LeRobot PI0Policy. Ignore
    # those extra keys while still loading the matching model weights.
    policy = policy_cls.from_pretrained(args.policy_path, config=cfg, strict=False).to(args.device).eval()
    if args.pi0_chunk_size is not None:
        policy.config.chunk_size = args.pi0_chunk_size
        policy.model.config.chunk_size = args.pi0_chunk_size
    if action_feature_mode:
        num_expert_layers = len(policy.model.paligemma_with_expert.gemma_expert.model.layers)
        if args.action_feature_layer < 0 or args.action_feature_layer >= num_expert_layers:
            raise ValueError(
                f"--action-feature-layer must be in [0, {num_expert_layers - 1}], "
                f"got {args.action_feature_layer}."
            )
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    action_source_preprocessor = None
    action_horizon = None
    num_train_timesteps = None
    if action_feature_mode:
        action_source_preprocessor, _ = make_pre_post_processors(
            action_source_cfg,
            pretrained_path=args.action_source_policy_checkpoint,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        action_horizon = action_source_cfg.horizon
        if policy.config.chunk_size < action_horizon:
            raise ValueError(
                f"PI0 chunk_size ({policy.config.chunk_size}) must be >= action horizon ({action_horizon})."
            )
        num_train_timesteps = action_source_cfg.num_train_timesteps

    feature_cache, hook_handles = ({}, [])
    if not action_feature_mode:
        feature_cache, hook_handles = build_feature_hooks(
            policy, args.feature_targets, args.feature_mode, args.save_dtype
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows: list[dict[str, Any]] = []
    current_episode_index: int | None = None
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                if args.max_frames is not None and batch_idx * args.batch_size >= args.max_frames:
                    break

                for values in feature_cache.values():
                    values.clear()

                mapped_batch = add_mapped_batch_keys(batch, args.image_key_map)
                episode_index = tensor_item(batch, "episode_index")
                if args.shard_by == "episode":
                    if not isinstance(episode_index, int):
                        raise ValueError(f"Expected scalar episode_index, got {episode_index}")
                    if current_episode_index is None:
                        current_episode_index = episode_index
                    elif episode_index != current_episode_index:
                        shard_path = episode_shard_path(output_path, current_episode_index)
                        save_rows(shard_path, args, rows)
                        print(f"[episode] saved {len(rows)} batches to {shard_path}")
                        rows = []
                        current_episode_index = episode_index

                if action_feature_mode:
                    if action_source_preprocessor is None or action_horizon is None or num_train_timesteps is None:
                        raise RuntimeError("Action feature mode was not initialized correctly.")
                    pi0_current_batch = select_current_observation_for_pi0(mapped_batch)
                    pi0_processed = preprocessor(pi0_current_batch)
                    prefix_pad_masks, past_key_values, state = build_pi0_prefix_cache(policy, pi0_processed)
                    action_batch = action_source_preprocessor(batch)
                    dp_actions = action_batch[ACTION]
                    frame_index = tensor_item(batch, "frame_index")
                    timestamp = tensor_item(batch, "timestamp")
                    task_index = tensor_item(batch, "task_index")
                    for repeat_idx in range(args.action_seed_repeats):
                        noise_seed = args.seed + args.seed_stride * repeat_idx + batch_idx
                        eps = make_noise_like(dp_actions, noise_seed)
                        pi0_actions, pi0_noise = build_pi0_padded_actions(policy, dp_actions, eps)
                        for timestep in args.action_feature_timesteps:
                            if timestep < 0 or timestep >= num_train_timesteps:
                                raise ValueError(
                                    f"Invalid timestep {timestep}; action source scheduler has "
                                    f"{num_train_timesteps} timesteps."
                                )
                            pi0_time = torch.full(
                                (dp_actions.shape[0],),
                                timestep / max(num_train_timesteps - 1, 1),
                                dtype=torch.float32,
                                device=dp_actions.device,
                            )
                            feature = collect_pi0_action_layer_feature(
                                policy,
                                prefix_pad_masks,
                                past_key_values,
                                state,
                                pi0_actions,
                                pi0_noise,
                                pi0_time,
                                args.action_feature_layer,
                                action_horizon,
                                args.action_feature_mode,
                                args.save_dtype,
                            )
                            rows.append(
                                {
                                    "batch_index": batch_idx,
                                    "episode_index": episode_index,
                                    "frame_index": frame_index,
                                    "timestamp": timestamp,
                                    "task_index": task_index,
                                    "task": batch.get("task"),
                                    "diffusion_timestep": timestep,
                                    "seed_repeat": repeat_idx,
                                    "noise_seed": noise_seed,
                                    "internal_features": {
                                        f"pi0.layer{args.action_feature_layer:02d}": feature[0]
                                    },
                                }
                            )
                else:
                    processed = preprocessor(mapped_batch)
                    _ = policy.predict_action_chunk(processed)
                    internal_features = {
                        name: stack_feature_list(values.copy()) for name, values in feature_cache.items()
                    }
                    rows.append(
                        {
                            "batch_index": batch_idx,
                            "episode_index": episode_index,
                            "frame_index": tensor_item(batch, "frame_index"),
                            "timestamp": tensor_item(batch, "timestamp"),
                            "task_index": tensor_item(batch, "task_index"),
                            "task": batch.get("task"),
                            "internal_features": internal_features,
                        }
                    )

                if args.save_every > 0 and (batch_idx + 1) % args.save_every == 0:
                    checkpoint_path = (
                        episode_shard_path(output_path, current_episode_index)
                        if args.shard_by == "episode" and current_episode_index is not None
                        else output_path
                    )
                    save_rows(checkpoint_path, args, rows)
                    print(f"[checkpoint] saved {len(rows)} batches to {checkpoint_path}")
    finally:
        for handle in hook_handles:
            handle.remove()

    if args.shard_by == "episode" and current_episode_index is not None:
        output_path = episode_shard_path(output_path, current_episode_index)
    save_rows(output_path, args, rows)
    print(f"Saved {len(rows)} batches to {output_path}")


if __name__ == "__main__":
    main()
