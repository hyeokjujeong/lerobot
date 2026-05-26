#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# export PYTHONDONTWRITEBYTECODE=1
# export NUMBA_DISABLE_JIT=1

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

if [[ ! -f "/PublicHDD2/jhri626/datasets/robomimic_lift_lerobot/meta/info.json" ]]; then
  python \
    "${SCRIPT_DIR}/prepare_robomimic_lift_dataset.py" \
    --repo-id "yananchen/robomimic_lift" \
    --output-root "/PublicHDD2/jhri626/datasets/robomimic_lift_lerobot"
fi

lerobot-train \
  --dataset.repo_id="yananchen/robomimic_lift" \
  --dataset.root="/PublicHDD2/jhri626/datasets/robomimic_lift_lerobot" \
  --dataset.video_backend=torchcodec \
  --dataset.return_uint8=true \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.resize_shape='[84,84]' \
  --policy.crop_ratio=0.904762 \
  --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.num_train_timesteps=100 \
  --policy.num_inference_steps=100 \
  --optimizer.type=adamw \
  --optimizer.lr=0.0001 \
  --optimizer.weight_decay=0.000001 \
  --optimizer.grad_clip_norm=10.0 \
  --optimizer.betas='[0.95,0.999]' \
  --optimizer.eps=0.00000001 \
  --scheduler.type=diffuser \
  --scheduler.name=cosine \
  --scheduler.num_warmup_steps=500 \
  --batch_size=64 \
  --num_workers=8 \
  --steps=30000 \
  --save_freq=10000 \
  --eval_freq=5000 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --env.type=robosuite \
  --env.task=Lift \
  --env.fps=10 \
  --env.episode_length=400 \
  --env.camera_name=agentview,robot0_eye_in_hand \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.state_dim=8 \
  --env.action_dim=7 \
  --env.control_freq=10 \
  --output_dir="/PublicHDD2/jhri626/outputs/diffusion_robosuite_lift" \
  --job_name=diffusion_robosuite_lift \
  --wandb.enable=true \
  --wandb.log_eval_video=false
