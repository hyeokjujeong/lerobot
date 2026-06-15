#!/usr/bin/env bash
# Do NOT use `set -e`: we want the sweep to continue even if one run fails.
set -uo pipefail

# AFT loss-weight sweep on LIBERO-10 task-0, across feature SOURCES.
# Trains (sequentially, single GPU):
#   1) ONE plain Diffusion Policy BASELINE  (AFT off; independent of features)
#   2) For EACH feature source, an AFT run per aft_beta (the AFT loss weight)
# All runs: same data / hyperparameters / step count; held-out validation BC loss ON.
#
# Thin orchestrator: it only calls scripts/train_libero_aft_diffusion_local.sh with the
# right env vars per run. The existing scripts are NOT modified.
#
# Usage:
#   conda activate lerobot
#   nohup bash scripts/sweep_libero_aft_diffusion.sh > /tmp/aft_sweep.log 2>&1 &
#
# Override via env:
#   STEPS (60000), BETAS ("0.1 0.3 0.7 1.0"), PREFIX (sweep_libero10_task0),
#   FEATURE_DIRS (space-separated dirs; default = pi0base + pi0libero),
#   VAL_EPISODES ("15,16,17,18,19"), VAL_FREQ (1000), RUN_BASELINE (true),
#   WANDB_* (see train script).

HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="${HERE}/train_libero_aft_diffusion_local.sh"
OUTPUTS_DIR="${HERE}/../outputs"
FEAT_ROOT="${HERE}/../extracted_feature"

# ---- sweep configuration ----
STEPS="${STEPS:-60000}"
PREFIX="${PREFIX:-sweep_libero10_task0}"
BETAS="${BETAS:-0.1 0.3 0.7 1.0}"
RUN_BASELINE="${RUN_BASELINE:-true}"
# Feature sources to sweep (each gets the full beta sweep). Both have pooled dim 2304
# (2 real cameras x 1152); the store auto-detects .safetensors (pi0base) and .pt (pi0libero).
FEATURE_DIRS="${FEATURE_DIRS:-${FEAT_ROOT}/libero10_task0_pi0base ${FEAT_ROOT}/libero10_task0_pi0libero}"
# Held-out validation (BC) loss — ON by default (episodes 15-19 have no features in
# either source, so val is a clean pure-BC generalization signal for every run).
VAL_EPISODES="${VAL_EPISODES:-15,16,17,18,19}"
VAL_FREQ="${VAL_FREQ:-1000}"

# Read by the train script (env-driven).
export VAL_EPISODES VAL_FREQ STEPS

# short tag from a feature dir basename: ".../libero10_task0_pi0base" -> "pi0base"
_tag_of() { local b="${1##*/}"; echo "${b#libero10_task0_}"; }
# count shards in a dir
_n_shards() {
  shopt -s nullglob; local s=( "$1"/*.safetensors "$1"/*.pt ); shopt -u nullglob; echo "${#s[@]}"
}

# Validate sources up-front; keep only those with shards.
VALID_DIRS=()
for d in ${FEATURE_DIRS}; do
  n="$(_n_shards "$d")"
  if (( n > 0 )); then
    VALID_DIRS+=("$d"); echo "source OK: $(_tag_of "$d")  ($n shards)  $d"
  else
    echo "source SKIP (no shards): $d" >&2
  fi
done
if (( ${#VALID_DIRS[@]} == 0 )); then
  echo "ERROR: no valid feature sources found under ${FEAT_ROOT}." >&2
  exit 1
fi

# total run count for the time estimate
n_betas=$(wc -w <<< "${BETAS}")
n_aft=$(( ${#VALID_DIRS[@]} * n_betas ))
n_base=0
[[ "${RUN_BASELINE}" == "true" ]] && n_base=1
n_total=$(( n_aft + n_base ))
echo "==================== AFT BETA x SOURCE SWEEP ===================="
echo "sources    : ${#VALID_DIRS[@]}  ($(for d in "${VALID_DIRS[@]}"; do _tag_of "$d"; done | tr '\n' ' '))"
echo "betas      : ${BETAS}"
echo "baseline   : ${RUN_BASELINE} (run once)"
echo "steps/run  : ${STEPS}"
echo "validation : episodes=${VAL_EPISODES} freq=${VAL_FREQ}"
echo "TOTAL RUNS : ${n_total}  (SEQUENTIAL, single GPU; ~110 min each -> ~$(( n_total * 110 / 60 ))h)"
echo "================================================================"

run_one() {
  local name="$1" aft_enable="$2" aft_beta="$3" feat_dir="$4"
  echo
  echo ">>> [$(date '+%F %H:%M:%S')] RUN: ${name}  (aft_enable=${aft_enable} aft_beta=${aft_beta} feat=$(_tag_of "${feat_dir}"))"
  if [[ -d "${OUTPUTS_DIR}/${name}" ]]; then
    echo "    SKIP: outputs/${name} already exists (delete it or change PREFIX to re-run)."
    return 0
  fi
  AFT_ENABLE="${aft_enable}" AFT_BETA="${aft_beta}" AFT_FEATURE_DIR="${feat_dir}" \
    bash "${TRAIN}" --run "${name}" \
    || echo "    WARN: run '${name}' failed (exit $?); continuing sweep."
}

# 1) plain-DP baseline (AFT off; feature dir unused but pass a valid one)
if [[ "${RUN_BASELINE}" == "true" ]]; then
  run_one "${PREFIX}_baseline" false 0 "${VALID_DIRS[0]}"
fi

# 2) AFT sweep: per feature source, per beta
for d in "${VALID_DIRS[@]}"; do
  tag="$(_tag_of "$d")"
  for b in ${BETAS}; do
    run_one "${PREFIX}_${tag}_aft_b${b}" true "${b}" "${d}"
  done
done

echo
echo "==================== SWEEP COMPLETE ===================="
echo "Runs in ${OUTPUTS_DIR}/${PREFIX}_* ; compare val/loss (+ aft_* for AFT runs) in wandb."
echo "Eval a finished run later, e.g.:"
echo "  bash scripts/eval_libero_aft_diffusion.sh --run ${PREFIX}_pi0base_aft_b1.0"
