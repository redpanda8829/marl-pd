#!/usr/bin/env bash
# ============================================================================
# Package setup for the BenchMARL target-defense project.
#
#   1. installs Miniconda (skipped if conda is already available)
#   2. creates the `marl` conda environment (python 3.10)
#   3. installs every required package at the versions this project was run on
#
# This script is self-contained: it can be copied to a bare machine on its own
# and run from anywhere. It does not read or need the project repository.
#
# Usage:
#   ./setup_environment.sh              # CPU build of torch
#   ./setup_environment.sh cu121        # CUDA 12.1 build instead
#
# The CPU build is the default because the experiment config runs on CPU
# (sampling_device=cpu, train_device=cpu) - a CUDA build buys nothing unless
# you also change those.
#
# NOTE: this installs the libraries only. The custom target-defense scenarios
# are NOT part of any pip package - once the repo is on this machine, run
# ./deploy_scenarios.sh from the repo root to copy them into site-packages and
# register them with BenchMARL.
# ============================================================================
set -euo pipefail


TORCH_VARIANT="${1:-cpu}"
ENV_NAME="marl"
PY_VERSION="3.10"
MINICONDA_DIR="${HOME}/miniconda3"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- 1. Miniconda -----------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  say "conda already present: $(command -v conda)"
  CONDA_BASE="$(conda info --base)"
elif [ -d "${MINICONDA_DIR}" ]; then
  say "Miniconda found at ${MINICONDA_DIR}"
  CONDA_BASE="${MINICONDA_DIR}"
else
  say "Installing Miniconda to ${MINICONDA_DIR}"
  case "$(uname -m)" in
    x86_64)          MC_ARCH="x86_64" ;;
    aarch64|arm64)   MC_ARCH="aarch64" ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  MC_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MC_ARCH}.sh"
  TMP_SH="$(mktemp /tmp/miniconda_XXXX.sh)"
  curl -fsSL "${MC_URL}" -o "${TMP_SH}"
  bash "${TMP_SH}" -b -p "${MINICONDA_DIR}"
  rm -f "${TMP_SH}"
  CONDA_BASE="${MINICONDA_DIR}"
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Newer conda refuses to touch the default channels non-interactively until
# their Terms of Service are accepted. Harmless no-op if already accepted or
# if this conda predates the ToS requirement.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

# --- 2. environment ---------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  say "conda env '${ENV_NAME}' already exists - reusing it"
else
  say "Creating conda env '${ENV_NAME}' (python ${PY_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi
conda activate "${ENV_NAME}"

# --- 3. packages ------------------------------------------------------------
say "Installing PyTorch (${TORCH_VARIANT})"
python -m pip install --upgrade pip
# Pinned to the versions this project trained on. If a CUDA variant does not
# host these exact wheels, drop the "==" pins and let pip pick the latest.
python -m pip install "torch==2.13.0" "torchvision==0.28.0" \
  --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"

say "Installing the RL stack"
# Versions this project was developed and trained against.
python -m pip install \
  "tensordict==0.11.0" \
  "torchrl==0.11.1" \
  "benchmarl==1.5.2" \
  "vmas==1.5.2"

say "Installing supporting libraries"
# - hydra/omegaconf : BenchMARL task + experiment configs
# - cvxpy           : the Apollonius pursuit-evasion solver (apollonius_solver.py)
# - pyglet <2.0     : VMAS renders with the pyglet 1.x API; pyglet 2 breaks it
# - av              : the mp4 backend. The CSV logger writes video through
#                     torchvision.io.write_video, which needs PyAV; without it
#                     video_format="mp4" fails and you only get .pt tensors
# - matplotlib      : visualizations/plot_apollonius_circles.py
# - numpy/scipy     : scenario math
python -m pip install \
  "hydra-core>=1.3,<1.4" \
  "omegaconf>=2.3,<2.4" \
  "cvxpy>=1.4" \
  "pyglet<2.0" \
  "av>=13,<14" \
  "matplotlib" \
  "numpy>=1.24" \
  "scipy>=1.10" \
  "tqdm"

# --- 4. verify --------------------------------------------------------------
say "Verifying the installation"
python - <<'PY'
import torch, torchrl, tensordict, vmas, benchmarl
print(f"  torch      {torch.__version__}")
print(f"  torchrl    {torchrl.__version__}")
print(f"  tensordict {tensordict.__version__}")
print(f"  vmas       {vmas.__version__}")
print(f"  benchmarl  {benchmarl.__version__}")
import cvxpy, pyglet, matplotlib, scipy, av      # noqa: F401
print(f"  cvxpy {cvxpy.__version__} | pyglet {pyglet.version} | "
      f"scipy {scipy.__version__} | av {av.__version__} (mp4 export)")
import site
print(f"\n  site-packages: {site.getsitepackages()[0]}")
PY

say "Done."
cat <<EOF

If this is the shell that just installed Miniconda for the first time, conda
is not yet on PATH here - either start a new terminal, or run:

  source ${CONDA_BASE}/etc/profile.d/conda.sh
  conda activate ${ENV_NAME}

From then on 'conda activate ${ENV_NAME}' works directly in any new shell.
EOF
cat <<'EOF'

The libraries are installed, but the target-defense scenarios are not - they
ship with this project, not with pip. VMAS loads scenarios by walking its own
vmas/scenarios/ directory, and BenchMARL looks tasks up in a hardcoded enum.

Once the repository is on this machine:

  cd <repo>
  conda activate marl
  ./deploy_scenarios.sh

That copies environments/*.py and yaml_configs/*.yaml into site-packages and
registers the TARGET_DEFENSE_* tasks. Re-run it after editing either directory -
the libraries execute their OWN copies, not the ones in the repo.
EOF
