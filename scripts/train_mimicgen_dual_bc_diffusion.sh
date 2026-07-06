#!/usr/bin/env bash
set -euo pipefail

# Train Diffusion Policy + TEACHER-FEATURE BC auxiliary loss
# (dual_bc_diffusion) on MimicGen Coffee_D2, using pre-extracted PI0 features.
#
# This is the MimicGen counterpart of train_libero_dual_bc_diffusion_local.sh:
#   L = L_bc(student) + tbc_lambda * L_bc(teacher)
#
# Expected feature shards are the per-episode safetensors produced by:
#   scripts/infer_pi0_features.sh
#
# Run:
#   conda activate lerobot
#   bash scripts/train_mimicgen_dual_bc_diffusion.sh
#
# Common overrides:
#   RUN_NAME=mimicgen_dualbc_adaT_l05 \
#   TBC_LAMBDA=0.5 TBC_ADAPTIVE_SCALE=true \
#   bash scripts/train_mimicgen_dual_bc_diffusion.sh
#
# --run NAME sets the wandb run name and output subdir. Any other --foo=bar args
# pass straight through to lerobot_train.

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

# --- parse --run (rest pass through) ---
RUN_NAME="${RUN_NAME:-mimicgen_coffee_d2_dualbc_adaF_l05}"
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_NAME="$2"; shift 2 ;;
    --run=*) RUN_NAME="${1#*=}"; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

EPISODES="${EPISODES:-$(python -c 'print("[" + ",".join(map(str, range(100))) + "]")')}"
export LEROBOT_VAL_EPISODES="${LEROBOT_VAL_EPISODES:-100,101,102,103,104,105,106,107,108,109}"
export LEROBOT_VAL_FREQ="${LEROBOT_VAL_FREQ:-1000}"
export LEROBOT_VAL_BATCHES="${LEROBOT_VAL_BATCHES:-32}"

DATASET_ROOT="${DATASET_ROOT:-/PublicSSD/jhri626/datasets/mimicgen_coffee_d2_lerobot_images}"
TBC_FEATURE_DIR="${TBC_FEATURE_DIR:-/PublicSSD/ft_vla/outputs/pi0_features_float/mimicgen}"
OUTPUT_DIR="${OUTPUT_DIR:-/PublicSSD/jhri626/outputs/${RUN_NAME}}"

TBC_ENABLE="${TBC_ENABLE:-true}"
TBC_LAMBDA="${TBC_LAMBDA:-0.5}"
TBC_PROJ_HIDDEN="${TBC_PROJ_HIDDEN:-null}"
TBC_PROJ_LR="${TBC_PROJ_LR:-null}"
TBC_ADAPTIVE_SCALE="${TBC_ADAPTIVE_SCALE:-false}"

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STEPS="${STEPS:-240000}"
SAVE_FREQ="${SAVE_FREQ:-40000}"
LOG_FREQ="${LOG_FREQ:-200}"

EVAL_FREQ="${EVAL_FREQ:-0}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-10}"
EVAL_BATCH="${EVAL_BATCH:-1}"
EVAL_ASYNC="${EVAL_ASYNC:-false}"

WANDB="${WANDB:-true}"
WANDB_ENTITY="${WANDB_ENTITY:-vla_ft}"
WANDB_PROJECT="${WANDB_PROJECT:-mimicgen_coffee_d2_no_crop}"
WANDB_ARGS=()
[[ "${WANDB}" == "true" ]] && WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}" --wandb.project="${WANDB_PROJECT}")

echo "Run name (wandb + output dir): ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Episodes: ${EPISODES}"
echo "Teacher-BC: enable=${TBC_ENABLE} lambda=${TBC_LAMBDA} adaptive_scale=${TBC_ADAPTIVE_SCALE}"
echo "Teacher feature dir: ${TBC_FEATURE_DIR}"
echo "Train: steps=${STEPS} batch_size=${BATCH_SIZE} save_freq=${SAVE_FREQ}"
[[ "${WANDB}" == "true" ]] && echo "WandB: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/mimicgen_coffee_d2 \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=torchcodec \
  --dataset.return_uint8=true \
  --dataset.episodes="${EPISODES}" \
  --policy.type=dual_bc_diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.resize_shape='[84,84]' \
  --policy.crop_ratio=1.0 \
  --policy.crop_is_random=false \
  --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=100 \
  --policy.tbc_enable="${TBC_ENABLE}" \
  --policy.tbc_feature_dir="${TBC_FEATURE_DIR}" \
  --policy.tbc_lambda="${TBC_LAMBDA}" \
  --policy.tbc_pretrained_dim=2304 \
  --policy.tbc_token_pool=mean \
  --policy.tbc_camera_reduce=concat \
  --policy.tbc_camera_indices='[0,1]' \
  --policy.tbc_obs_step=-1 \
  --policy.tbc_proj_hidden="${TBC_PROJ_HIDDEN}" \
  --policy.tbc_proj_lr="${TBC_PROJ_LR}" \
  --policy.tbc_adaptive_scale="${TBC_ADAPTIVE_SCALE}" \
  --policy.tbc_broadcast_obs_steps=true \
  --policy.tbc_share_noise=true \
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
  --log_freq="${LOG_FREQ}" \
  --eval_freq="${EVAL_FREQ}" \
  --eval.n_episodes="${EVAL_N_EPISODES}" \
  --eval.batch_size="${EVAL_BATCH}" \
  --eval.use_async_envs="${EVAL_ASYNC}" \
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
  --wandb.enable="${WANDB}" \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
  --wandb.disable_artifact=true \
  --wandb.log_eval_video=false \
  ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
