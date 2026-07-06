#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
POLICY_PATH="${POLICY_PATH:-/home/jhri626/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/mimicgen_coffee_d2}"
DATASET_ROOT="${DATASET_ROOT:-/PublicSSD/jhri626/datasets/mimicgen_coffee_d2_lerobot_images}"
EPISODES="${EPISODES:-0:100}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
FEATURE_TARGETS="${FEATURE_TARGETS:-vision_tower}"
FEATURE_MODE="${FEATURE_MODE:-full}"
SAVE_DTYPE="${SAVE_DTYPE:-float32}"
IMAGE_KEY_MAP="${IMAGE_KEY_MAP:-observation.images.agentview=observation.images.base_0_rgb,observation.images.robot0_eye_in_hand=observation.images.left_wrist_0_rgb}"
OUTPUT_PATH="${OUTPUT_PATH:-/PublicSSD/ft_vla/outputs/pi0_features_float/mimicgen_coffee_d2_ep0_99_images_pi0_base_features_full_float32.safetensors}"
ACTION_FEATURE_TIMESTEPS="${ACTION_FEATURE_TIMESTEPS:-}"
ACTION_FEATURE_LAYER="${ACTION_FEATURE_LAYER:-17}"
ACTION_FEATURE_MODE="${ACTION_FEATURE_MODE:-pooled}"
ACTION_SEED_REPEATS="${ACTION_SEED_REPEATS:-1}"
ACTION_SOURCE_POLICY_CHECKPOINT="${ACTION_SOURCE_POLICY_CHECKPOINT:-/PublicSSD/jhri626/outputs/libero10_task0_diffusion_obs2/checkpoints/080000/pretrained_model}"
PI0_CHUNK_SIZE="${PI0_CHUNK_SIZE:-}"
SEED="${SEED:-0}"
SEED_STRIDE="${SEED_STRIDE:-100000}"

EXTRA_ARGS=()
if [[ -n "${ACTION_FEATURE_TIMESTEPS}" ]]; then
  EXTRA_ARGS+=(
    --action-feature-timesteps "${ACTION_FEATURE_TIMESTEPS}"
    --action-feature-layer "${ACTION_FEATURE_LAYER}"
    --action-feature-mode "${ACTION_FEATURE_MODE}"
    --action-seed-repeats "${ACTION_SEED_REPEATS}"
    --action-source-policy-checkpoint "${ACTION_SOURCE_POLICY_CHECKPOINT}"
    --seed "${SEED}"
    --seed-stride "${SEED_STRIDE}"
  )
fi
if [[ -n "${PI0_CHUNK_SIZE}" ]]; then
  EXTRA_ARGS+=(--pi0-chunk-size "${PI0_CHUNK_SIZE}")
fi

python "${SCRIPT_DIR}/infer_pi0_features.py" \
  --policy-path "${POLICY_PATH}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --dataset-root "${DATASET_ROOT}" \
  --episodes "${EPISODES}" \
  --device "cuda" \
  --batch-size 1 \
  --num-workers 4 \
  --video-backend "${VIDEO_BACKEND}" \
  --feature-targets "${FEATURE_TARGETS}" \
  --feature-mode "${FEATURE_MODE}" \
  --save-dtype "${SAVE_DTYPE}" \
  --image-key-map "${IMAGE_KEY_MAP}" \
  --shard-by "episode" \
  --output-format "safetensors" \
  --output-path "${OUTPUT_PATH}" \
  "${EXTRA_ARGS[@]}"
