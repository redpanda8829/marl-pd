#!/usr/bin/env bash
# Wait for the stage-1 entropy A/B to finish, pick the winner on evidence,
# then launch the full curriculum with that entropy value.
#
# Winner = higher mean TRAINING reward over the last 10 logged iterations
# (training reward is logged every iteration, so it is far less noisy than the
# 10-episode evaluation reward). The entropy trend of each arm is recorded
# alongside it for the write-up.

set -uo pipefail
ROOT="/home/azmainy/Azmain/Projects/CRADL LAB/benchmarl-target-defense"
SCRATCH="/tmp/claude-1000/-home-azmainy-Azmain-Projects-CRADL-LAB-benchmarl-target-defense/db3d8cf2-86e5-42d3-98de-0d93ad90af21/scratchpad"
cd "$ROOT"
source /home/azmainy/miniconda3/etc/profile.d/conda.sh
conda activate marl

echo "[$(date +%H:%M:%S)] waiting for entropy A/B to finish..."
while pgrep -f "train_curriculum.py --task TARGET_DEFENSE_BASIC --tag ab_ent" > /dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] A/B finished"

WINNER=$(python - <<'PY'
import json, glob
from pathlib import Path

def stats(tag):
    sc = glob.glob(f"outputs/curriculum/{tag}/*/*/scalars")
    if not sc:
        return None
    sc = Path(sc[0])
    def curve(n):
        f = sc / n
        if not f.exists():
            return []
        return [float(l.split(",")[1]) for l in f.read_text().strip().splitlines() if l.strip()]
    tr, ent = curve("collection_reward_episode_reward_mean.csv"), curve("train_defender_entropy.csv")
    if not tr:
        return None
    return {"final_reward_mean10": sum(tr[-10:]) / len(tr[-10:]),
            "final_entropy": ent[-1] if ent else None,
            "entropy_start": ent[0] if ent else None,
            "n_iters": len(tr)}

a, b = stats("ab_ent001"), stats("ab_ent000")
report = {"entropy_0.01": a, "entropy_0.0": b}

if a and b:
    winner = "0.01" if a["final_reward_mean10"] >= b["final_reward_mean10"] else "0.0"
elif a:
    winner = "0.01"
else:
    winner = "0.0"
report["winner"] = winner
Path("outputs/curriculum/entropy_ab_result.json").write_text(json.dumps(report, indent=2))
print(winner)
PY
)

echo "[$(date +%H:%M:%S)] entropy A/B winner: $WINNER"
cat outputs/curriculum/entropy_ab_result.json

echo "[$(date +%H:%M:%S)] launching curriculum with entropy=$WINNER, 500000 frames/stage"
./run_curriculum.sh "$WINNER" 500000
echo "[$(date +%H:%M:%S)] pipeline script returned"
