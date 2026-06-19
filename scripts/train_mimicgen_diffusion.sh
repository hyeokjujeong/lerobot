#!/usr/bin/env bash
set -euo pipefail

export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=egl

EPISODES="$(python -c 'print("[" + ",".join(map(str, range(100))) + "]")')"
export LEROBOT_VAL_EPISODES="${LEROBOT_VAL_EPISODES:-100,101,102,103,104,105,106,107,108,109}"
export LEROBOT_VAL_FREQ="${LEROBOT_VAL_FREQ:-1000}"
export LEROBOT_VAL_BATCHES="${LEROBOT_VAL_BATCHES:-32}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/mimicgen_coffee_d2 \
  --dataset.root=/PublicSSD/jhri626/datasets/mimicgen_coffee_d2_lerobot_images \
  --dataset.video_backend=torchcodec \
  --dataset.return_uint8=true \
  --dataset.episodes="${EPISODES}" \
  --policy.type=diffusion \
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
  --steps=240000 \
  --save_freq=40000 \
  --eval_freq=0 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
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
  --output_dir=/PublicSSD/jhri626/outputs/diffusion_mimicgen_coffee_d2_ep100_images_no_crop \
  --job_name=diffusion_mimicgen_coffee_d2_ep100_images_no_crop \
  --wandb.enable=true \
  --wandb.entity=vla_ft \
  --wandb.project=mimicgen_coffee_d2 \
  --wandb.disable_artifact=true \
  --wandb.log_eval_video=false
