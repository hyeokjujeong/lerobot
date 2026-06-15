#!/usr/bin/env bash
# Continue the sweep-eval even if one eval fails.
set -uo pipefail

# Evaluate ALL models from the AFT sweep in the LIBERO-10 task-0 simulator.
# For each run, evaluates the "6000" and "last" checkpoints, 50 episodes each,
# and collects pc_success into a summary CSV.
#
# Run this AFTER the sweep training has finished (so the GPU is free).
#
# Usage:
#   conda activate lerobot
#   nohup bash scripts/eval_all_sweep.sh > /tmp/aft_eval_all.log 2>&1 &
#
# Override via env:
#   PREFIX (sweep_libero10_task0)      : which runs to eval (outputs/<PREFIX>*)
#   N_EPISODES (50)
#   CKPTS ("006000 last")              : which checkpoints per run
#   EVAL_BATCH (1)  ASYNC (false)      : raise EVAL_BATCH (e.g. 5) to try to speed up
#                                        (sync, in-process; less tested for batch>1 here)
#   MUJOCO_GL (egl)

export MUJOCO_GL="${MUJOCO_GL:-egl}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTPUTS_DIR="$(cd "${HERE}/.." && pwd)/outputs"

PREFIX="${PREFIX:-sweep_libero10_task0}"
N_EPISODES="${N_EPISODES:-50}"
CKPTS="${CKPTS:-006000 last}"
EVAL_BATCH="${EVAL_BATCH:-1}"
ASYNC="${ASYNC:-false}"
SUMMARY="${OUTPUTS_DIR}/${PREFIX}_eval_summary.csv"
PY="${PY:-python}"

# discover run dirs that actually have a checkpoints/ folder
mapfile -t RUNS < <(for d in "${OUTPUTS_DIR}/${PREFIX}"*/; do [[ -d "${d}checkpoints" ]] && basename "${d%/}"; done)
if (( ${#RUNS[@]} == 0 )); then
  echo "ERROR: no runs matching ${OUTPUTS_DIR}/${PREFIX}* with a checkpoints/ dir." >&2
  exit 1
fi

# count planned evals
n_plan=0
for r in "${RUNS[@]}"; do for ck in ${CKPTS}; do
  [[ -d "${OUTPUTS_DIR}/${r}/checkpoints/${ck}/pretrained_model" ]] && n_plan=$((n_plan+1))
done; done

echo "==================== EVAL ALL SWEEP ===================="
echo "runs        : ${#RUNS[@]}  (${RUNS[*]})"
echo "checkpoints : ${CKPTS}"
echo "episodes    : ${N_EPISODES}  | eval_batch=${EVAL_BATCH} async=${ASYNC} MUJOCO_GL=${MUJOCO_GL}"
echo "planned eval jobs: ${n_plan}  (SEQUENTIAL; ~${N_EPISODES}x~100s each at batch 1)"
echo "summary csv : ${SUMMARY}"
echo "========================================================"

[[ -f "${SUMMARY}" ]] || echo "run,checkpoint,pc_success,n_episodes,resolved_ckpt" > "${SUMMARY}"

eval_one() {
  local run="$1" ck="$2"
  local ckdir="${OUTPUTS_DIR}/${run}/checkpoints/${ck}/pretrained_model"
  local out="${OUTPUTS_DIR}/${run}/eval_${ck}"
  local info="${out}/eval_info.json"
  if [[ ! -d "${ckdir}" ]]; then
    echo "  SKIP ${run}/${ck}: no checkpoint at ${ckdir}"; return 0
  fi
  if [[ -f "${info}" ]]; then
    echo "  CACHED ${run}/${ck}: ${info} already exists (delete to re-eval)"
  else
    echo ">>> [$(date '+%F %H:%M:%S')] EVAL ${run} / ${ck}  (${N_EPISODES} ep)"
    "${PY}" -m lerobot.scripts.lerobot_eval \
      --policy.path="${ckdir}" \
      --policy.device=cuda \
      --env.type=libero \
      --env.task=libero_10 \
      --env.task_ids='[0]' \
      --env.fps=20 \
      --env.episode_length=500 \
      --env.observation_height=256 \
      --env.observation_width=256 \
      --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
      --eval.n_episodes="${N_EPISODES}" \
      --eval.batch_size="${EVAL_BATCH}" \
      --eval.use_async_envs="${ASYNC}" \
      --output_dir="${out}" \
      || { echo "  WARN: eval ${run}/${ck} failed; continuing."; return 0; }
  fi
  # append pc_success to the summary
  local pc resolved
  pc="$("${PY}" -c "import json;print(json.load(open('${info}'))['overall']['pc_success'])" 2>/dev/null || echo NA)"
  resolved="$(readlink -f "${OUTPUTS_DIR}/${run}/checkpoints/${ck}" 2>/dev/null || echo "${ck}")"
  echo "${run},${ck},${pc},${N_EPISODES},$(basename "${resolved}")" >> "${SUMMARY}"
  echo "    pc_success=${pc}%"
}

for run in "${RUNS[@]}"; do
  for ck in ${CKPTS}; do
    eval_one "${run}" "${ck}"
  done
done

echo
echo "==================== EVAL ALL DONE ===================="
echo "Summary (${SUMMARY}):"
column -t -s, "${SUMMARY}"
