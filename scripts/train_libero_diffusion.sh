#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

# NOTE: Uncomment this block if you want to train on a random subset of 10
# episodes from the 50-episode dataset. With the current sampler implementation,
# random/non-contiguous episode subsets may require the sampler fix noted in
# lerobot/src/lerobot/scripts/lerobot_train.py.
#
# EPISODES="$(
#   python - <<'PY'
# import random
# random.seed(42)
# episodes = sorted(random.sample(range(50), 10))
# print("[" + ",".join(map(str, episodes)) + "]")
# PY
# )"
#
# echo "Training with episodes: ${EPISODES}"

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
  --dataset.repo_id=yzembodied/libero_10_image_task_0 \
  --dataset.root=/PublicSSD/jhri626/datasets/libero_10_image_task_0 \
  --dataset.episodes='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]' \
  --dataset.video_backend=pyav \
  --dataset.return_uint8=true \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0]' \
  --env.fps=20 \
  --env.episode_length=500 \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
  --output_dir=/PublicSSD/jhri626/outputs/libero10_task0_diffusion_obs2_ep15 \
  --job_name=libero10_task0_diffusion_obs2_ep15 \
  --steps=24000 \
  --batch_size=32 \
  --num_workers=16 \
  --prefetch_factor=2 \
  --eval_freq=0 \
  --save_freq=6000 \
  --log_freq=500 \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --wandb.enable=true \
  --wandb.log_eval_video=false
