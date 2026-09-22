#!/usr/bin/env bash
# Overnight curriculum pipeline.
#
#   stage 1  basic            (trained from scratch)
#   stage 2  smart            <- stage 1
#        then the chain forks, because parameter sharing only becomes
#        meaningful once there is more than one defender:
#   stage 3  w_obs            <- stage 2     (shared | unshared)
#   stage 4  w_multi_obs      <- stage 3     (shared | unshared)
#   stage 5  multi_obs_hetero <- stage 4     (shared | unshared)
#
# Each stage partially transfers weights from its parent (see
# train_curriculum.py): everything except the input projections carries over.
#
# Usage: ./run_curriculum.sh [ENTROPY] [FRAMES]

set -uo pipefail

ENTROPY="${1:-0.01}"
FRAMES="${2:-1000000}"
ROOT="/home/azmainy/Azmain/Projects/CRADL LAB/benchmarl-target-defense"
LOGS="$ROOT/outputs/curriculum/logs"
mkdir -p "$LOGS"

cd "$ROOT"
source /home/azmainy/miniconda3/etc/profile.d/conda.sh
conda activate marl

ckpt_of () {  # read the final checkpoint path recorded by a finished stage
  python - "$1" <<'PY'
import json,sys
print(json.load(open(f"outputs/curriculum/{sys.argv[1]}/stage_result.json"))["final_checkpoint"])
PY
}

run_stage () {  # tag task parent share
  local tag="$1" task="$2" parent="$3" share="$4"
  echo "[$(date +%H:%M:%S)] START $tag  (task=$task share=$share parent=${parent:-none})"
  local args=(--task "$task" --tag "$tag" --frames "$FRAMES" --entropy "$ENTROPY" --share-params "$share")
  [ -n "$parent" ] && args+=(--parent "$parent")
  python -u train_curriculum.py "${args[@]}" > "$LOGS/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FAILED $tag (exit $rc) - see $LOGS/$tag.log"
    return $rc
  fi
  echo "[$(date +%H:%M:%S)] DONE  $tag"
}

echo "=== curriculum start: entropy=$ENTROPY frames/stage=$FRAMES ==="

# ---- shared trunk -------------------------------------------------------
run_stage s1_basic TARGET_DEFENSE_BASIC "" true || exit 1
CKPT1="$(ckpt_of s1_basic)"

run_stage s2_smart TARGET_DEFENSE_SMART "$CKPT1" true || exit 1
CKPT2="$(ckpt_of s2_smart)"

echo "=== trunk complete; forking shared / unshared branches in parallel ==="

# ---- branch: shared parameters -----------------------------------------
(
  run_stage s3_wobs_shared  TARGET_DEFENSE_SMART_W_OBS            "$CKPT2" true || exit 1
  run_stage s4_multi_shared TARGET_DEFENSE_SMART_W_MULTI_OBS      "$(ckpt_of s3_wobs_shared)"  true || exit 1
  run_stage s5_hetero_shared TARGET_DEFENSE_SMART_MULTI_OBS_HETERO "$(ckpt_of s4_multi_shared)" true || exit 1
  echo "=== SHARED branch complete ==="
) > "$LOGS/branch_shared.log" 2>&1 &
PID_SHARED=$!

# ---- branch: per-agent (unshared) parameters ---------------------------
(
  run_stage s3_wobs_unshared  TARGET_DEFENSE_SMART_W_OBS            "$CKPT2" false || exit 1
  run_stage s4_multi_unshared TARGET_DEFENSE_SMART_W_MULTI_OBS      "$(ckpt_of s3_wobs_unshared)"  false || exit 1
  run_stage s5_hetero_unshared TARGET_DEFENSE_SMART_MULTI_OBS_HETERO "$(ckpt_of s4_multi_unshared)" false || exit 1
  echo "=== UNSHARED branch complete ==="
) > "$LOGS/branch_unshared.log" 2>&1 &
PID_UNSHARED=$!

wait $PID_SHARED;   RC_S=$?
wait $PID_UNSHARED; RC_U=$?

echo "=== curriculum finished (shared rc=$RC_S, unshared rc=$RC_U) ==="
date
