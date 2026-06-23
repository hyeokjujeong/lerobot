#!/usr/bin/env bash
set -euo pipefail

export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

python -m lerobot.scripts.lerobot_eval \
  --policy.path=/PublicSSD/jhri626/outputs/diffusion_mimicgen_coffee_d2_ep100_images_no_crop/checkpoints/120000/pretrained_model \
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
  --eval.batch_size=25 \
  --eval.n_episodes=50 \
  --eval.use_async_envs=false \
  --output_dir=/PublicSSD/jhri626/eval/Dp/diffusion_mimicgen_coffee_d2_ep100_images_no_crop_120000
