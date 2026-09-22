#!/usr/bin/env bash
# ============================================================================
# Deploy this project's custom scenarios into the installed libraries.
#
# A plain `pip install vmas benchmarl` gives you none of the target-defense
# environments. VMAS loads scenarios by walking its own vmas/scenarios/
# directory, BenchMARL reads task configs from benchmarl/conf/task/vmas/ and
# looks tasks up in a hardcoded VmasTask enum. This script wires all three up.
#
# Run it from the repository root, with the `marl` env active:
#   conda activate marl && ./deploy_scenarios.sh
#
# Re-run it whenever you edit environments/ or yaml_configs/ - the libraries
# execute their OWN copies under site-packages, not the ones in this repo.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

for required in environments yaml_configs apollonius_solver.py; do
  if [ ! -e "${REPO_ROOT}/${required}" ]; then
    echo "ERROR: '${required}' not found next to this script." >&2
    echo "This script must live in, and be run from, the repository root." >&2
    exit 1
  fi
done

SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
VMAS_SCENARIOS="${SITE_PACKAGES}/vmas/scenarios"
BENCHMARL_ENVS="${SITE_PACKAGES}/benchmarl/environments/vmas"
BENCHMARL_CONF="${SITE_PACKAGES}/benchmarl/conf/task/vmas"

for d in "${VMAS_SCENARIOS}" "${BENCHMARL_ENVS}" "${BENCHMARL_CONF}"; do
  [ -d "$d" ] || {
    echo "ERROR: expected directory missing: $d" >&2
    echo "Is the 'marl' environment active? Run setup_environment.sh first." >&2
    exit 1
  }
done

say "Copying scenarios into ${SITE_PACKAGES}"
cp -v "${REPO_ROOT}"/environments/*.py "${VMAS_SCENARIOS}/"      # what VMAS actually executes
cp -v "${REPO_ROOT}"/environments/*.py "${BENCHMARL_ENVS}/"      # for TaskConfig type-checking
cp -v "${REPO_ROOT}"/yaml_configs/*.yaml "${BENCHMARL_CONF}/"    # task configs

say "Registering tasks in BenchMARL's VmasTask enum"
python - "${BENCHMARL_ENVS}/common.py" <<'PY'
import re, sys
path = sys.argv[1]
src = open(path).read()

tasks = [
    "TARGET_DEFENSE_BASIC",
    "TARGET_DEFENSE_SMART",
    "TARGET_DEFENSE_SMART_W_OBS",
    "TARGET_DEFENSE_SMART_W_MULTI_OBS",
    "TARGET_DEFENSE_SMART_MULTI_OBS_HETERO",
    "TARGET_DEFENSE_PATROL",
    "TARGET_DEFENSE_PATROL_MA",
    "TARGET_DEFENSE_PATROL_END2END",
]
missing = [t for t in tasks if not re.search(rf"^\s*{t}\s*=", src, re.M)]
if not missing:
    print("  all tasks already registered")
    raise SystemExit(0)

# append the new members after the last existing "NAME = None" enum entry
matches = list(re.finditer(r"^(\s*)([A-Z][A-Z0-9_]*)\s*=\s*None\s*$", src, re.M))
if not matches:
    raise SystemExit("ERROR: could not locate the VmasTask enum members")
last = matches[-1]
indent = last.group(1)
block = "".join(f"{indent}{t} = None\n" for t in missing)
src = src[: last.end() + 1] + block + src[last.end() + 1 :]
open(path, "w").write(src)
print("  registered:", ", ".join(missing))
PY

say "Verifying the scenarios load"
cd "${REPO_ROOT}"
python - <<'PY'
import sys
sys.path.insert(0, ".")          # apollonius_solver.py lives at the repo root
import vmas

try:
    from apollonius_solver import solve_apollonius_optimization  # noqa: F401
    print("  apollonius_solver: importable (cvxpy solver active)")
except Exception as e:
    print(f"  WARNING apollonius_solver not importable: {e}")
    print("  -> scenarios fall back to a y-coordinate reward approximation")

from benchmarl.environments import VmasTask
ok = True
for name in ["TARGET_DEFENSE_BASIC", "TARGET_DEFENSE_SMART",
             "TARGET_DEFENSE_SMART_W_OBS", "TARGET_DEFENSE_SMART_W_MULTI_OBS",
             "TARGET_DEFENSE_SMART_MULTI_OBS_HETERO"]:
    try:
        cfg = VmasTask[name].get_from_yaml().config
        env = vmas.make_env(scenario=name.lower(), num_envs=2, device="cpu",
                            continuous_actions=True, **cfg)
        obs = env.reset()
        print(f"  OK  {name:38s} obs_dim={obs[0].shape[1]}")
    except Exception as e:
        ok = False
        print(f"  FAIL {name:38s} {type(e).__name__}: {e}")
sys.exit(0 if ok else 1)
PY

say "Scenarios deployed."
cat <<'EOF'

Run everything from the repository root (apollonius_solver.py is imported as a
top-level module, so the repo root must be the working directory):

  ./run_curriculum.sh 0.0 500000          # full 5-stage curriculum
  python train_with_video.py task=vmas/target_defense_smart_multi_obs_hetero \
      algorithm=mappo experiment.loggers=[csv] experiment.max_n_frames=500000
  python analyze_curriculum.py            # aggregate results
EOF
