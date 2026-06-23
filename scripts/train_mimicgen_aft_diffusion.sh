#!/usr/bin/env bash
set -euo pipefail

# Train Diffusion Policy + vision AFT on MimicGen Coffee_D2.
#
# This mirrors scripts/train_mimicgen_diffusion.sh and only swaps the policy to
# aft_diffusion with PI0 vision-feature regularization enabled.
#
# Expected feature shards are the per-episode safetensors produced by:
#   scripts/infer_pi0_features.sh
#
# Override examples:
#   AFT_FEATURE_DIR=/path/to/pi0_feature_shards bash scripts/train_mimicgen_aft_diffusion.sh
#   RUN_NAME=aft_b03 AFT_BETA=0.3 BATCH_SIZE=32 bash scripts/train_mimicgen_aft_diffusion.sh
#   INIT_POLICY_PATH=/path/to/dp/checkpoint/pretrained_model \
#     RUN_NAME=aft_from_dp120000 STEPS=120000 bash scripts/train_mimicgen_aft_diffusion.sh

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

EPISODES="${EPISODES:-$(python -c 'print("[" + ",".join(map(str, range(100))) + "]")')}"
export LEROBOT_VAL_EPISODES="${LEROBOT_VAL_EPISODES:-100,101,102,103,104,105,106,107,108,109}"
export LEROBOT_VAL_FREQ="${LEROBOT_VAL_FREQ:-1000}"
export LEROBOT_VAL_BATCHES="${LEROBOT_VAL_BATCHES:-32}"

AFT_FEATURE_DIR="${AFT_FEATURE_DIR:-/PublicSSD/ft_vla/outputs/pi0_features_float}"
AFT_BETA="${AFT_BETA:-0.3}"
AFT_ENABLE="${AFT_ENABLE:-true}"
INIT_POLICY_PATH="${INIT_POLICY_PATH:-}"

RUN_NAME="${RUN_NAME:-aft_diffusion_mimicgen_coffee_d2_ep100_images_no_crop_aft-finetune_float}"
OUTPUT_DIR="${OUTPUT_DIR:-/PublicSSD/jhri626/outputs/${RUN_NAME}}"
WARM_START_POLICY_DIR="${WARM_START_POLICY_DIR:-/PublicSSD/jhri626/outputs/aft_warm_start_policies/${RUN_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STEPS="${STEPS:-240000}"
SAVE_FREQ="${SAVE_FREQ:-40000}"

POLICY_SOURCE_ARGS=(
  --policy.type=aft_diffusion
  --policy.n_obs_steps=2
  --policy.horizon=16
  --policy.n_action_steps=8
  --policy.resize_shape='[84,84]'
  --policy.crop_ratio=1.0
  --policy.crop_is_random=false
  --policy.use_group_norm=true
  --policy.pretrained_backbone_weights=null
  --policy.use_separate_rgb_encoder_per_camera=true
  --policy.num_train_timesteps=100
  --policy.num_inference_steps=100
)

if [[ -n "${INIT_POLICY_PATH}" ]]; then
  if [[ ! -f "${INIT_POLICY_PATH}/config.json" || ! -f "${INIT_POLICY_PATH}/model.safetensors" ]]; then
    echo "ERROR: INIT_POLICY_PATH must point to a pretrained_model dir with config.json and model.safetensors." >&2
    echo "       Got: ${INIT_POLICY_PATH}" >&2
    exit 1
  fi

  mkdir -p "${WARM_START_POLICY_DIR}"
  for src in "${INIT_POLICY_PATH}"/*; do
    name="$(basename "${src}")"
    [[ "${name}" == "config.json" ]] && continue
    ln -sfn "$(readlink -f "${src}")" "${WARM_START_POLICY_DIR}/${name}"
  done

  INIT_POLICY_PATH="${INIT_POLICY_PATH}" WARM_START_POLICY_DIR="${WARM_START_POLICY_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["INIT_POLICY_PATH"]) / "config.json"
dst = Path(os.environ["WARM_START_POLICY_DIR"]) / "config.json"
cfg = json.loads(src.read_text())
cfg["type"] = "aft_diffusion"
dst.write_text(json.dumps(cfg, indent=4) + "\n")
PY

  POLICY_SOURCE_ARGS=(--policy.path="${WARM_START_POLICY_DIR}")
fi

echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "AFT feature dir: ${AFT_FEATURE_DIR}"
echo "AFT enabled: ${AFT_ENABLE} beta=${AFT_BETA}"
if [[ -n "${INIT_POLICY_PATH}" ]]; then
  echo "Warm-start policy: ${INIT_POLICY_PATH}"
  echo "AFT policy copy: ${WARM_START_POLICY_DIR}"
fi

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/mimicgen_coffee_d2 \
  --dataset.root=/PublicSSD/jhri626/datasets/mimicgen_coffee_d2_lerobot_images \
  --dataset.video_backend=torchcodec \
  --dataset.return_uint8=true \
  --dataset.episodes="${EPISODES}" \
  "${POLICY_SOURCE_ARGS[@]}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.aft_enable="${AFT_ENABLE}" \
  --policy.aft_feature_dir="${AFT_FEATURE_DIR}" \
  --policy.aft_beta="${AFT_BETA}" \
  --policy.aft_pretrained_dim=2304 \
  --policy.aft_learn_scales=true \
  --policy.aft_kernel=linear \
  --policy.aft_token_pool=mean \
  --policy.aft_camera_reduce=concat \
  --policy.aft_camera_indices='[0,1]' \
  --policy.aft_obs_step=-1 \
  --policy.aft_prior_lr=1e-2 \
  --optimizer.type=adamw \
  --optimizer.lr=0.0001 \
  --optimizer.weight_decay=0.000001 \
  --optimizer.grad_clip_norm=10.0 \
  --optimizer.betas='[0.95,0.999]' \
  --optimizer.eps=0.00000001 \
  --scheduler.type=diffuser \
  --scheduler.name=cosine \
  --scheduler.num_warmup_steps=500 \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq=0 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --env.type=mimicgen \
  --env.task=Coffee_D2 \
  --env.fps=20 \
  --env.episode_length=400 \
  --env.camera_name=agentview,robot0_eye_in_hand \
  --env.observation_height=84 \
  --env.observation_width=84 \
  --env.state_dim=8 \
  --env.action_dim=7 \
  --env.control_freq=20 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable=true \
  --wandb.entity=vla_ft \
  --wandb.project=mimicgen_coffee_d2_no_crop \
  --wandb.disable_artifact=true \
  --wandb.log_eval_video=false
