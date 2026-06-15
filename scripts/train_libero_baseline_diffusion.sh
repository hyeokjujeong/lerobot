#!/usr/bin/env bash
set -euo pipefail

# Plain-DP BASELINE for the AFT ablation: identical setup to
# scripts/train_libero_aft_diffusion_local.sh but with the AFT regularizer turned
# OFF (AFT_ENABLE=false). Same policy code path (aft_diffusion), same data, same
# hyperparameters -> the only difference vs the AFT run is the AFT loss term, so a
# success-rate / val-loss comparison isolates AFT's effect.
#
# Usage:
#   conda activate lerobot
#   bash scripts/train_libero_baseline_diffusion.sh --run baseline_b0
#   # with held-out val loss + in-training eval (same knobs as the AFT script):
#   VAL_EPISODES="15,16,17,18,19" EVAL=true bash scripts/train_libero_baseline_diffusion.sh --run baseline_b0
#
# All env overrides and passthrough args of the AFT script apply here too.

HERE="$(cd "$(dirname "$0")" && pwd)"
# Default to a distinct run name so it doesn't collide with AFT runs.
export RUN_NAME="${RUN_NAME:-libero10_task0_baseline_diffusion}"
AFT_ENABLE=false exec bash "${HERE}/train_libero_aft_diffusion_local.sh" "$@"
