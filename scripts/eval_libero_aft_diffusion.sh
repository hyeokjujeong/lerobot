#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained AFT (or plain) Diffusion Policy checkpoint in the LIBERO-10
# task-0 simulator and report success rate.
#
# AFT note: at inference an `aft_diffusion` policy behaves exactly like a plain
# Diffusion Policy (no PI0 features are needed), so this script also works for a
# checkpoint trained with --policy.type=diffusion.
#
# REQUIRES the LIBERO simulator: pip install -e '.[libero]'  (needs a C/C++
# toolchain + EGL/GL dev headers to build egl_probe; see the chat notes). Offscreen
# rendering uses MUJOCO_GL (egl by default; fall back to osmesa if egl fails).
#
# Usage:
#   conda activate lerobot
#   bash scripts/eval_libero_aft_diffusion.sh --run libero10_task0_aft_diffusion_test
#   # or point directly at a checkpoint:
#   bash scripts/eval_libero_aft_diffusion.sh --ckpt outputs/<run>/checkpoints/006000/pretrained_model
# Optional env: MUJOCO_GL (egl|osmesa), N_EPISODES, OUTPUT_DIR.

export MUJOCO_GL="${MUJOCO_GL:-egl}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_NAME="${RUN_NAME:-libero10_task0_aft_diffusion_test_longer}"
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

# Default to the run's "last" checkpoint if --ckpt was not given.
CKPT="${CKPT:-${REPO_ROOT}/outputs/${RUN_NAME}/checkpoints/last/pretrained_model}"
N_EPISODES="${N_EPISODES:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}/eval}"

if [[ ! -d "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  echo "       Train first (a checkpoint appears at save_freq / end), or pass --ckpt <path>." >&2
  exit 1
fi

echo "Checkpoint: ${CKPT}"
echo "MUJOCO_GL=${MUJOCO_GL}  n_episodes=${N_EPISODES}  output=${OUTPUT_DIR}"

python -m lerobot.scripts.lerobot_eval \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0]' \
  --env.fps=20 \
  --env.episode_length=500 \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
  --eval.batch_size=1 \
  --eval.n_episodes="${N_EPISODES}" \
  --eval.use_async_envs=false \
  --output_dir="${OUTPUT_DIR}" \
  ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
