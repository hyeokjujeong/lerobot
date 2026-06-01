#!/usr/bin/env bash
set -euo pipefail

# export PYTHONDONTWRITEBYTECODE=1
# export NUMBA_DISABLE_JIT=1

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

python -m lerobot.scripts.lerobot_eval \
  --policy.path="/PublicSSD/jhri626/outputs/diffusion_robosuite_lift/checkpoints/030000/pretrained_model" \
  --policy.device=cuda \
  --env.type=robosuite \
  --env.task=Lift \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=1 \
  --env.camera_name=agentview,robot0_eye_in_hand \
  --env.state_dim=8 \
  --env.action_dim=7 \
  --env.control_freq=10 \
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --output_dir="/PublicSSD/jhri626/eval/diffusion_robosuite_lift_030000"
