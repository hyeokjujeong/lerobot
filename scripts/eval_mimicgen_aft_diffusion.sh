#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained AFT Diffusion Policy checkpoint in MimicGen Coffee_D2.
#
# AFT note: at inference an `aft_diffusion` policy behaves like a plain Diffusion
# Policy. PI0 feature shards are only needed during training, not eval.
#
# Usage:
#   conda activate lerobot
#   bash scripts/eval_mimicgen_aft_diffusion.sh
#   bash scripts/eval_mimicgen_aft_diffusion.sh --run aft_diffusion_mimicgen_coffee_d2_ep100_images
#   bash scripts/eval_mimicgen_aft_diffusion.sh --ckpt /path/to/checkpoints/last/pretrained_model
# Optional env: MUJOCO_GL, N_EPISODES, OUTPUTS_ROOT, EVAL_ROOT, OUTPUT_DIR.

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="${RUN_NAME:-aft_diffusion_mimicgen_coffee_d2_ep100_images_no_crop_aft-finetune_float}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/PublicSSD/jhri626/outputs}"
EVAL_ROOT="${EVAL_ROOT:-/PublicSSD/jhri626/eval}"
CKPT=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_NAME="$2"; shift 2 ;;
    --run=*) RUN_NAME="${1#*=}"; shift ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --ckpt=*) CKPT="${1#*=}"; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

CKPT="${CKPT:-${OUTPUTS_ROOT}/${RUN_NAME}/checkpoints/120000/pretrained_model}"
N_EPISODES="${N_EPISODES:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_ROOT}/aft_float/${RUN_NAME}_120000}"

if [[ ! -d "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  echo "       Train first, or pass --ckpt <path>." >&2
  exit 1
fi

echo "Checkpoint: ${CKPT}"
echo "MUJOCO_GL=${MUJOCO_GL}  n_episodes=${N_EPISODES}  output=${OUTPUT_DIR}"

python -m lerobot.scripts.lerobot_eval \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --env.type=mimicgen \
  --env.task=Coffee_D2 \
  --env.obs_type=pixels_agent_pos \
  --env.fps=20 \
  --env.observation_height=84 \
  --env.observation_width=84 \
  --env.max_parallel_tasks=1 \
  --env.camera_name=agentview,robot0_eye_in_hand \
  --env.state_dim=8 \
  --env.action_dim=7 \
  --env.control_freq=20 \
  --eval.batch_size=50 \
  --eval.n_episodes="${N_EPISODES}" \
  --eval.use_async_envs=false \
  --output_dir="${OUTPUT_DIR}" \
  ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
