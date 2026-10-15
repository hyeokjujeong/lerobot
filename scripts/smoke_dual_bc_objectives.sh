#!/usr/bin/env bash
set -uo pipefail

# SMOKE: run every `dual_bc_diffusion` auxiliary objective for a handful of steps.
#
# Purpose: before committing GPU-weeks to the comparison grid, confirm for EACH arm
# that (a) it builds, (b) the teacher feature store actually matches frames
# (`tbc_matched` > 0), and (c) the auxiliary term is finite and non-zero
# (`tbc_aux_loss`). It does NOT check that any arm learns anything.
#
# All arms are driven through `train_libero_dual_bc_diffusion_local.sh`, so they take
# exactly the training code path — this smoke test cannot drift from the real runs.
#
# Run:
#   conda activate lerobot
#   bash scripts/smoke_dual_bc_objectives.sh
#
# Env overrides: OBJECTIVES, STEPS, BATCH_SIZE, EPISODES, TBC_LAMBDA, OUT_ROOT,
#   plus everything train_libero_dual_bc_diffusion_local.sh accepts
#   (TBC_FEATURE_DIR, TBC_FEATURE_KEY, TBC_PRETRAINED_DIM, DATASET_ROOT, ...).
# Extra CLI args are forwarded verbatim to the training script.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Any extra args are forwarded verbatim to the training script (e.g. --dataset.root=...).
EXTRA_ARGS=("$@")
OBJECTIVES="${OBJECTIVES:-dual_bc pullback feature kd aft none}"
STEPS="${STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPISODES="${EPISODES:-[0,1]}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/smoke_dual_bc_objectives}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "==================== SMOKE: dual_bc objectives ===================="
echo "objectives: ${OBJECTIVES}"
echo "steps=${STEPS} batch=${BATCH_SIZE} episodes=${EPISODES}"
echo "logs: ${LOG_DIR}"
echo "==================================================================="

declare -a RESULTS=()
status=0

for obj in ${OBJECTIVES}; do
  log="${LOG_DIR}/${obj}.log"
  echo ""
  echo "---- ${obj} ----"
  STEPS="${STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  EPISODES="${EPISODES}" \
  WANDB=false \
  EVAL=false \
  EVAL_ON_SAVE=false \
  RUN_NAME="smoke_${obj}" \
  OUTPUT_DIR="${OUT_ROOT}/${obj}" \
  TBC_OBJECTIVE="${obj}" \
    bash "${REPO_ROOT}/scripts/train_libero_dual_bc_diffusion_local.sh" \
    --save_checkpoint=false --log_freq=10 \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >"${log}" 2>&1
  rc=$?

  # Last logged [DualBC] line carries tbc_matched / tbc_aux_loss for this arm.
  line="$(grep -o '\[DualBC\].*' "${log}" | tail -1)"
  if [[ ${rc} -ne 0 ]]; then
    RESULTS+=("  FAIL ${obj}: exited ${rc} -> ${log}")
    status=1
  elif [[ "${obj}" == "none" ]]; then
    # The control arm deliberately emits no teacher metrics.
    if [[ -n "${line}" ]]; then
      RESULTS+=("  FAIL ${obj}: emitted teacher metrics but should not -> ${log}")
      status=1
    else
      RESULTS+=("  OK   ${obj}: ran as plain DP (no teacher term), as expected")
    fi
  elif [[ -z "${line}" ]]; then
    RESULTS+=("  FAIL ${obj}: no [DualBC] metrics logged -> ${log}")
    status=1
  else
    matched="$(sed -n 's/.*tbc_matched:\([0-9]*\).*/\1/p' <<<"${line}")"
    aux="$(sed -n 's/.*tbc_aux_loss:\([0-9.eE+-]*\).*/\1/p' <<<"${line}")"
    if [[ -z "${matched}" || "${matched}" -eq 0 ]]; then
      RESULTS+=("  FAIL ${obj}: tbc_matched=0 — the feature store keyed no frame -> ${log}")
      status=1
    elif [[ -z "${aux}" ]] || awk "BEGIN{exit !(${aux:-0} == 0)}"; then
      RESULTS+=("  FAIL ${obj}: tbc_aux_loss is zero/absent — the term never fired -> ${log}")
      status=1
    else
      RESULTS+=("  OK   ${obj}: matched=${matched} aux_loss=${aux}")
    fi
  fi
  echo "${RESULTS[-1]}"
done

echo ""
echo "==================== SUMMARY ===================="
printf '%s\n' "${RESULTS[@]}"
echo "================================================="
[[ ${status} -eq 0 ]] && echo "ALL ARMS OK" || echo "SOME ARMS FAILED (see logs above)"
exit ${status}
