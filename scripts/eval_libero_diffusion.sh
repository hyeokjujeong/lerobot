#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

python -m lerobot.scripts.lerobot_eval \
  --policy.path="/PublicSSD/jhri626/outputs/libero10_task0_diffusion_obs2_ep15/checkpoints/last/pretrained_model" \
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
  --eval.n_episodes=50 \
  --eval.use_async_envs=false \
  --output_dir="/PublicSSD/jhri626/eval/libero10_task0_diffusion_obs2_ep15" \

