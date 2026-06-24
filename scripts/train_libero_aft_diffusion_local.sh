#!/usr/bin/env bash
set -euo pipefail

# FULL local training: Diffusion Policy + vision AFT on LIBERO-10 task-0,
# using the pi0_base extracted features. Tailored for this machine (WSL, single
# RTX 5070 Ti 16GB):
#   - dataset is read from the local HF cache (must be v3.0; see one-time setup below)
#   - NO --env.* flags  -> the `libero` sim package is not required; eval is off
#   - batch sized for 16GB VRAM
#
# ONE-TIME SETUP (already done on this machine):
#   1) Install extras:   pip install -e '.[training]'   (pulls dataset+diffusion deps)
#                        pip install diffusers
#   2) Convert dataset v2.1 -> v3.0 (downloads ~2GB into ~/.cache/huggingface/lerobot):
#        python -m lerobot.scripts.convert_dataset_v21_to_v30 \
#          --repo-id=yzembodied/libero_10_image_task_0 --push-to-hub=false
#
# Run:
#   conda activate lerobot          # ~/miniforge3/envs/lerobot
#   bash scripts/train_libero_aft_diffusion_local.sh --run my_run_name
#
# --run NAME  : sets the wandb run name (= --job_name) AND the output subdir.
#               (equivalently set RUN_NAME=... as an env var). Default: libero10_task0_aft_diffusion.
# Any other --foo=bar args are passed straight through to lerobot_train, e.g.:
#   bash scripts/train_libero_aft_diffusion_local.sh --run aft_beta2 --policy.aft_beta=2.0
# Optional env overrides: AFT_BETA, AFT_ENABLE, BATCH_SIZE, STEPS, WANDB, OUTPUT_DIR,
#   RUN_NAME, WANDB_ENTITY (default vla_ft), WANDB_PROJECT (default libero_task0).
#
# Ablation / analysis helpers:
#   AFT_ENABLE=false                       -> plain-DP baseline (AFT term off). See also
#                                             scripts/train_libero_baseline_diffusion.sh.
#   VAL_EPISODES="15,16,17,18,19"          -> log held-out validation (BC) loss; tune with
#                                             VAL_FREQ (1000), VAL_BATCHES (16).
#   EVAL=true                              -> in-training LIBERO success-rate eval (needs
#                                             lerobot[libero]); tune EVAL_FREQ (6000),
#                                             EVAL_N_EPISODES (20), MUJOCO_GL (egl).
# Examples:
#   VAL_EPISODES="15,16,17,18,19" EVAL=true bash scripts/train_libero_aft_diffusion_local.sh --run aft_b1
#   AFT_ENABLE=false VAL_EPISODES="15,16,17,18,19" bash scripts/train_libero_aft_diffusion_local.sh --run baseline
# These are forwarded as explicit --wandb.entity / --wandb.project flags so logging
# goes to the team regardless of shell env quirks. Examples:
#   WANDB_ENTITY=vla_ft bash scripts/train_libero_aft_diffusion_local.sh --run myrun
#   bash scripts/train_libero_aft_diffusion_local.sh --run myrun --wandb.entity=vla_ft

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- parse --run (rest of the args pass through to lerobot_train) ---
RUN_NAME="${RUN_NAME:-libero10_task0_aft_diffusion_test_longer}"
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_NAME="$2"; shift 2 ;;
    --run=*) RUN_NAME="${1#*=}"; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

AFT_FEATURE_DIR="${AFT_FEATURE_DIR:-${REPO_ROOT}/extracted_feature/libero10_task0_pi0base}"
# Each run name gets its own output dir so checkpoints don't collide.
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"
AFT_BETA="${AFT_BETA:-1.0}"
AFT_ENABLE="${AFT_ENABLE:-true}"   # set false for a plain-DP baseline (same code path, AFT term off)
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-48000}"
WANDB="${WANDB:-true}"
# Training demos. Override for fewer demos, e.g. EPISODES='[0,1,2,3,4,5,6,7,8,9]' (10 demos).
EPISODES="${EPISODES:-[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]}"

# --- (a) Held-out validation BC loss (opt-in). Set VAL_EPISODES="15,16,..." to enable.
#     Read by lerobot_train.py via these env vars; unset -> no validation.
if [[ -n "${VAL_EPISODES:-}" ]]; then
  export LEROBOT_VAL_EPISODES="${VAL_EPISODES}"
  export LEROBOT_VAL_FREQ="${VAL_FREQ:-1000}"
  export LEROBOT_VAL_BATCHES="${VAL_BATCHES:-16}"
  echo "Validation: episodes=${LEROBOT_VAL_EPISODES} freq=${LEROBOT_VAL_FREQ} batches=${LEROBOT_VAL_BATCHES}"
fi

# --- (b) OPTIONAL in-training success-rate eval in the LIBERO sim.
#     Requires lerobot[libero]; uses MUJOCO_GL for offscreen rendering.
#   EVAL=true          : eval on the --eval_freq schedule (default every 6000 steps).
#   EVAL_ON_SAVE=true  : eval at EVERY checkpoint save (couples eval to --save_freq);
#                        sets eval_freq=0 so eval fires only on saves (no double-eval).
#   Either one turns on the sim env flags below.
EVAL="${EVAL:-false}"
EVAL_ON_SAVE="${EVAL_ON_SAVE:-false}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-10}"
EVAL_ARGS=()
if [[ "${EVAL}" == "true" || "${EVAL_ON_SAVE}" == "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  if [[ "${EVAL_ON_SAVE}" == "true" ]]; then
    export LEROBOT_EVAL_ON_SAVE=true
    EVAL_FREQ_ARG=0   # eval driven by checkpoint saves (save_freq), not eval_freq
    echo "In-training eval: ON EVERY SAVE (save_freq) (MUJOCO_GL=${MUJOCO_GL} n_episodes=${EVAL_N_EPISODES})"
  else
    EVAL_FREQ_ARG="${EVAL_FREQ:-6000}"
    echo "In-training eval: ENABLED every ${EVAL_FREQ_ARG} steps (MUJOCO_GL=${MUJOCO_GL} n_episodes=${EVAL_N_EPISODES})"
  fi
  EVAL_ARGS+=(
    --eval_freq="${EVAL_FREQ_ARG}"
    --eval.n_episodes="${EVAL_N_EPISODES}"
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
  EVAL_ARGS+=(--eval_freq=0)   # default: no in-training rollout eval
fi

# Forward WANDB_ENTITY / WANDB_PROJECT as EXPLICIT lerobot flags. lerobot calls
# wandb.init(entity=cfg.wandb.entity) with a default of None, so the safest way to
# log to a team is to pass --wandb.entity explicitly (init kwargs take priority over
# any shell env). Defaults below point logging at the `vla_ft` team; override by
# exporting WANDB_ENTITY / WANDB_PROJECT or passing --wandb.entity=... after --run.
WANDB_ENTITY="${WANDB_ENTITY:-vla_ft}"
WANDB_PROJECT="${WANDB_PROJECT:-libero_task0}"
WANDB_ARGS=()
[[ "${WANDB}" == "true" ]] && WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}" --wandb.project="${WANDB_PROJECT}")

echo "Run name (wandb + output dir): ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
[[ "${WANDB}" == "true" ]] && echo "WandB: entity=${WANDB_ENTITY} project=${WANDB_PROJECT}"

python -m lerobot.scripts.lerobot_train \
  --policy.type=aft_diffusion \
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
