#!/usr/bin/env bash
set -euo pipefail

# FULL local training: Diffusion Policy + TEACHER-FEATURE BC auxiliary loss
# (dual_bc_diffusion) on LIBERO-10 task-0, using the pi0_base extracted features.
#
# This is the "two naive BC losses" variant (NO AFT kernel term, NO divergence
# term): the shared U-Net is trained on the same noisy actions from (a) the
# student's own vision conditioning and (b) a learnable projection of the PI0
# teacher feature, and the two BC losses are summed:
#     L = L_bc(student) + tbc_lambda * L_bc(teacher)
# See DUAL_BC_DIFFUSION_IMPLEMENTATION.md.
#
# Tailored for this machine (WSL, single RTX 5070 Ti 16GB):
#   - dataset read from local HF cache (must be v3.0; see train_libero_aft_diffusion_local.sh setup)
#   - NO --env.* flags  -> the `libero` sim package is not required; eval is off
#
# Run:
#   conda activate lerobot
#   bash scripts/train_libero_dual_bc_diffusion_local.sh --run my_run_name
#
# --run NAME : wandb run name (= --job_name) AND output subdir. (or RUN_NAME=...)
# Any other --foo=bar args pass straight through, e.g.:
#   bash scripts/train_libero_dual_bc_diffusion_local.sh --run dbc_l0.5 --policy.tbc_lambda=0.5
# Optional env overrides: TBC_LAMBDA, TBC_ENABLE, TBC_PROJ_HIDDEN, TBC_PROJ_LR,
#   TBC_ADAPTIVE_SCALE (AFT-style sigmoid(s) gate before projection),
#   BATCH_SIZE, STEPS, WANDB, EPISODES, VAL_EPISODES, EVAL, EVAL_ON_SAVE, RUN_NAME,
#   EVAL_N_EPISODES, EVAL_BATCH (eval parallel envs, default 1), EVAL_ASYNC (default false),
#   WANDB_ENTITY (default vla_ft), WANDB_PROJECT (default libero_task0),
#   TBC_FEATURE_DIR (default extracted_feature/libero10_task0_pi0base).
#
# Ablation: TBC_ENABLE=false -> plain-DP baseline (teacher term off, same code path).

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- parse --run (rest pass through) ---
RUN_NAME="${RUN_NAME:-libero10_task0_dual_bc_diffusion}"
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_NAME="$2"; shift 2 ;;
    --run=*) RUN_NAME="${1#*=}"; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

TBC_FEATURE_DIR="${TBC_FEATURE_DIR:-${REPO_ROOT}/extracted_feature/libero10_task0_pi0base}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"
TBC_LAMBDA="${TBC_LAMBDA:-1.0}"
TBC_ENABLE="${TBC_ENABLE:-true}"          # false => plain-DP baseline
TBC_PROJ_HIDDEN="${TBC_PROJ_HIDDEN:-null}" # null => single Linear; int => 1-hidden MLP
TBC_PROJ_LR="${TBC_PROJ_LR:-null}"         # null => model LR
TBC_ADAPTIVE_SCALE="${TBC_ADAPTIVE_SCALE:-false}" # true => AFT-style sigmoid(s) gate before projection
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-60000}"
WANDB="${WANDB:-true}"
EPISODES="${EPISODES:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]}"

# --- held-out validation BC loss (opt-in) ---
if [[ -n "${VAL_EPISODES:-}" ]]; then
  export LEROBOT_VAL_EPISODES="${VAL_EPISODES}"
  export LEROBOT_VAL_FREQ="${VAL_FREQ:-1000}"
  export LEROBOT_VAL_BATCHES="${VAL_BATCHES:-16}"
  echo "Validation: episodes=${LEROBOT_VAL_EPISODES} freq=${LEROBOT_VAL_FREQ} batches=${LEROBOT_VAL_BATCHES}"
fi

# --- optional in-training LIBERO success-rate eval ---
# EVAL=true          : eval on the --eval_freq schedule (default every 6000 steps).
# EVAL_ON_SAVE=true  : eval at EVERY checkpoint save (couples eval to --save_freq);
#                      sets eval_freq=0 so eval fires only on saves (no double-eval).
# Either one turns on the sim env flags below.
EVAL="${EVAL:-false}"
EVAL_ON_SAVE="${EVAL_ON_SAVE:-false}"
EVAL_ARGS=()
if [[ "${EVAL}" == "true" || "${EVAL_ON_SAVE}" == "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  if [[ "${EVAL_ON_SAVE}" == "true" ]]; then
    export LEROBOT_EVAL_ON_SAVE=true
    EVAL_FREQ_ARG=0   # eval driven by checkpoint saves (save_freq), not eval_freq
    echo "In-training eval: ON EVERY SAVE (save_freq) (MUJOCO_GL=${MUJOCO_GL})"
  else
    EVAL_FREQ_ARG="${EVAL_FREQ:-6000}"
    echo "In-training eval: ENABLED every ${EVAL_FREQ_ARG} steps (MUJOCO_GL=${MUJOCO_GL})"
  fi
  EVAL_ARGS+=(
    --eval_freq="${EVAL_FREQ_ARG}"
    --eval.n_episodes="${EVAL_N_EPISODES:-10}"
    --eval.batch_size="${EVAL_BATCH:-1}"
    --eval.use_async_envs="${EVAL_ASYNC:-false}"
    --env.type=libero
    --env.task=libero_10
    --env.task_ids='[0]'
    --env.fps=20
    --env.episode_length=500
    --env.observation_height=256
    --env.observation_width=256
    --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
  )
else
  EVAL_ARGS+=(--eval_freq=0)
fi

WANDB_ENTITY="${WANDB_ENTITY:-vla_ft}"
WANDB_PROJECT="${WANDB_PROJECT:-libero_task0}"
WANDB_ARGS=()
[[ "${WANDB}" == "true" ]] && WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}" --wandb.project="${WANDB_PROJECT}")

echo "Run name (wandb + output dir): ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Teacher-BC: enable=${TBC_ENABLE} lambda=${TBC_LAMBDA} feature_dir=${TBC_FEATURE_DIR}"
[[ "${WANDB}" == "true" ]] && echo "WandB: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"

python -m lerobot.scripts.lerobot_train \
  --policy.type=dual_bc_diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.beta_start=0.0001 \
  --policy.beta_end=0.02 \
  --policy.beta_schedule=squaredcos_cap_v2 \
  --policy.prediction_type=epsilon \
  --policy.clip_sample=true \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=null \
  --policy.use_group_norm=true \
  --policy.crop_is_random=false \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.down_dims='[512,1024,2048]' \
  --policy.kernel_size=5 \
  --policy.n_groups=8 \
  --policy.diffusion_step_embed_dim=128 \
  --policy.use_film_scale_modulation=true \
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
  --use_policy_training_preset=true \
  --optimizer.type=adamw \
  --optimizer.lr=1e-4 \
  --optimizer.betas='[0.95,0.999]' \
  --optimizer.eps=1e-8 \
  --optimizer.weight_decay=1e-6 \
  --optimizer.grad_clip_norm=10.0 \
  --scheduler.type=diffuser \
  --scheduler.name=cosine \
  --scheduler.num_warmup_steps=500 \
  --dataset.repo_id=yzembodied/libero_10_image_task_0 \
  --dataset.episodes="${EPISODES}" \
  --dataset.video_backend=pyav \
  --dataset.return_uint8=true \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers=4 \
  --prefetch_factor=2 \
  --save_freq=6000 \
  --log_freq=200 \
  --wandb.enable="${WANDB}" \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
  ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
  ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
