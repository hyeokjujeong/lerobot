#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

GLOBAL_TASK_ID="${GLOBAL_TASK_ID:-${1:-}}"
if [[ -z "${GLOBAL_TASK_ID}" ]]; then
  echo "Usage: GLOBAL_TASK_ID=03 bash scripts/train_libero_official_diffusion.sh" >&2
  echo "   or: bash scripts/train_libero_official_diffusion.sh 03" >&2
  exit 1
fi

GLOBAL_TASK_ID="$((10#${GLOBAL_TASK_ID}))"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-40}"
SUBSET_ROOT_BASE="${SUBSET_ROOT_BASE:-/PublicHDD/jhri626/datasets/libero_official_ep${EPISODES_PER_TASK}}"
OUTPUT_BASE="${OUTPUT_BASE:-/PublicSSD/jhri626/lerobot/outputs}"
WANDB_PROJECT="${WANDB_PROJECT:-libero_official_diffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-vla_ft}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

if (( GLOBAL_TASK_ID < 0 || GLOBAL_TASK_ID > 39 )); then
  echo "GLOBAL_TASK_ID must be in [0, 39], got ${GLOBAL_TASK_ID}" >&2
  exit 1
fi

if (( GLOBAL_TASK_ID < 10 )); then
  SUITE="libero_10"
  SUITE_SHORT="10"
elif (( GLOBAL_TASK_ID < 20 )); then
  SUITE="libero_goal"
  SUITE_SHORT="goal"
elif (( GLOBAL_TASK_ID < 30 )); then
  SUITE="libero_object"
  SUITE_SHORT="object"
else
  SUITE="libero_spatial"
  SUITE_SHORT="spatial"
fi

LOCAL_TASK_ID=$(( GLOBAL_TASK_ID % 10 ))
TASK_PADDED="$(printf "%02d" "${LOCAL_TASK_ID}")"
GLOBAL_PADDED="$(printf "%02d" "${GLOBAL_TASK_ID}")"
SUBSET_NAME="libero_${SUITE_SHORT}_task${TASK_PADDED}_global${GLOBAL_PADDED}_ep${EPISODES_PER_TASK}"
DATASET_ROOT="${DATASET_ROOT:-${SUBSET_ROOT_BASE}/${SUBSET_NAME}}"
RUN_NAME="${RUN_NAME:-${SUBSET_NAME}_diffusion_obs2}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  echo "Create it first with scripts/prepare_libero_official_subsets.py" >&2
  exit 1
fi

shift $(( $# > 0 ? 1 : 0 ))

python -m lerobot.scripts.lerobot_train \
  --policy.type=diffusion \
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
  --dataset.repo_id=lerobot/libero \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --dataset.return_uint8=true \
  --env.type=libero \
  --env.task="${SUITE}" \
  --env.task_ids="[${LOCAL_TASK_ID}]" \
  --env.fps=10 \
  --env.episode_length=500 \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"image2"}' \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --steps="${STEPS:-60000}" \
  --batch_size="${BATCH_SIZE:-64}" \
  --num_workers="${NUM_WORKERS:-8}" \
  --prefetch_factor="${PREFETCH_FACTOR:-2}" \
  --eval_freq="${EVAL_FREQ:-6000}" \
  --save_freq="${SAVE_FREQ:-6000}" \
  --log_freq="${LOG_FREQ:-200}" \
  --eval.n_episodes="${EVAL_N_EPISODES:-50}" \
  --eval.batch_size="${EVAL_BATCH_SIZE:-10}" \
  --eval.use_async_envs=false \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.log_eval_video=false \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.entity="${WANDB_ENTITY}" \
  "$@"
