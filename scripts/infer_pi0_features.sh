#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
POLICY_PATH="${POLICY_PATH:-/home/jhri626/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2}"

python "${SCRIPT_DIR}/infer_pi0_features.py" \
  --policy-path "${POLICY_PATH}" \
  --dataset-repo-id "local/mimicgen_coffee_d2" \
  --dataset-root "/PublicSSD/jhri626/datasets/mimicgen_coffee_d2_lerobot_images" \
  --episodes "0:100" \
  --device "cuda" \
  --batch-size 1 \
  --num-workers 4 \
  --video-backend "torchcodec" \
  --feature-targets "vision_tower" \
  --feature-mode "full" \
  --save-dtype "float32" \
  --image-key-map "observation.images.agentview=observation.images.base_0_rgb,observation.images.robot0_eye_in_hand=observation.images.left_wrist_0_rgb" \
  --shard-by "episode" \
  --output-format "safetensors" \
  --output-path "/PublicSSD/ft_vla/outputs/pi0_features_float/mimicgen_coffee_d2_ep0_99_images_pi0_base_features_full_float32.safetensors"
