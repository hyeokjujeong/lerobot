#!/usr/bin/env bash
set -euo pipefail

# Train the Diffusion Policy on LIBERO-10 task-0 with vision Adaptive Feature
# Transfer (AFT) from pre-extracted PI0 `features.vision_tower` features.
#
# This is a copy of scripts/train_libero_diffusion.sh with `--policy.type=aft_diffusion`
# and the AFT-specific flags appended. The base diffusion training script is left
# unchanged. Before running, extract the PI0 features into
# lerobot/extracted_feature/<run> following pi0_feature_extraction_flow.md.
#
# AFT pooled feature dim:
#   - 2 REAL cameras (image + wrist), vision-tower dim 1152, camera_reduce=concat -> 2304
#   - set --policy.aft_camera_reduce=mean with --policy.aft_pretrained_dim=1152 to average cameras.
#
# Camera slots in features.vision_tower depend on the PI0 checkpoint used to extract:
#   - pi0_libero_base                -> 2 slots (no dummy)
#   - pi0_base + --image-key-map     -> 3 slots; slot 2 is a DUMMY missing right-wrist camera.
# aft_camera_indices='[0,1]' keeps the two real cameras (base, left_wrist) and drops the
# dummy slot. It is safe for both checkpoints (the 2-slot case simply has no index 2).

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the environment first: conda activate lerobot" >&2
  exit 1
fi

AFT_FEATURE_DIR="${AFT_FEATURE_DIR:-$(cd "$(dirname "$0")/.." && pwd)/extracted_feature/libero10_task0_pi0base}"

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
  --policy.aft_enable=true \
  --policy.aft_feature_dir="${AFT_FEATURE_DIR}" \
  --policy.aft_beta=1.0 \
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
  --output_dir=/PublicSSD/jhri626/outputs/libero10_task0_aft_diffusion_obs2_ep15 \
  --job_name=libero10_task0_aft_diffusion_obs2_ep15 \
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
