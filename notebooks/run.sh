#!/bin/bash
#SBATCH -J MYGPUJOB
#SBATCH -o MYGPUJOB.o%j
#SBATCH -e MYGPUJOB.e%j
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH -t 48:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

echo "========================================"
echo "  Job info"
echo "========================================"
echo "Hostname:  $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not set}"

# ── Conda ──────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-nn}
cd -P "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"

CONDA_EXE="$(command -v conda)"
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── CUDA module ────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module load cuda12.6/toolkit 2>/dev/null || module load cuda11.8/toolkit 2>/dev/null || true
fi

# ──────────────────────────────────────────────────────
# FIX: Prepend pip-installed nvidia lib dirs to LD_LIBRARY_PATH
# ──────────────────────────────────────────────────────
SITE_PKGS="$CONDA_PREFIX/lib/python3.10/site-packages"
NVIDIA_LD_PATHS=""
for pkg_dir in "$SITE_PKGS"/nvidia/*/lib; do
  [ -d "$pkg_dir" ] && NVIDIA_LD_PATHS="${pkg_dir}:${NVIDIA_LD_PATHS}"
done
if [ -n "$NVIDIA_LD_PATHS" ]; then
  export LD_LIBRARY_PATH="${NVIDIA_LD_PATHS}${LD_LIBRARY_PATH:-}"
  echo "Prepended pip nvidia lib dirs to LD_LIBRARY_PATH"
else
  echo "WARNING: No pip nvidia lib dirs found under $SITE_PKGS/nvidia/"
fi

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# ── Environment ────────────────────────────────────────
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ──────────────────────────────────────────────────────
# Option (2): Install BlackJAX from main (must expose blackjax.nss)
# ──────────────────────────────────────────────────────
echo "========================================"
echo "  Python package setup"
echo "========================================"

# (Optional) keep this, but make it deterministic w.r.t. JAX stack:
python -m pip install --upgrade --no-deps jaxoplanet

# Replace PyPI blackjax with main-branch blackjax; do NOT touch deps (jax/jaxlib/etc)
python -m pip uninstall -y blackjax >/dev/null 2>&1 || true
python -m pip install --upgrade --no-deps "git+https://github.com/blackjax-devs/blackjax.git@main"

# Hard fail if NSS still isn't there (prints to stdout/stderr so you can see it in MYGPUJOB.*)
python - <<'PY'
import blackjax
print("blackjax version:", getattr(blackjax, "__version__", "unknown"))
print("blackjax path:", blackjax.__file__)
has_nss = hasattr(blackjax, "nss")
print("has blackjax.nss:", has_nss)
if not has_nss:
    raise SystemExit("ERROR: blackjax.nss is missing after installing from GitHub main.")
# Optional: also ensure ns utils import works (common failure if install is incomplete)
from blackjax.ns.utils import finalise, sample, log_weights, ess
print("NSS imports OK.")
PY

# ── GPU check ──────────────────────────────────────────
echo "========================================"
echo "  GPU check"
echo "========================================"
nvidia-smi

# ── JAX sanity check ──────────────────────────────────
echo ""
echo "========================================"
echo "  JAX GPU check"
echo "========================================"
python - <<'PY'
import jax
print(f"jax: {jax.__version__}, backend: {jax.default_backend()}, devices: {jax.devices()}")
if not any(d.platform in ("gpu", "cuda") for d in jax.devices()):
    raise SystemExit("ERROR: JAX did not initialize a GPU backend.")
import jax.numpy as jnp
x = jnp.ones((100, 100))
print(f"GPU compute test OK: {jnp.dot(x, x).devices()}")
PY

# ── Run ────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Running nss.py"
echo "========================================"

srun python -u nss.py