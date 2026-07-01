#!/bin/bash
#SBATCH -J SWAMP_SMC
#SBATCH -o SWAMP_SMC.o%j
#SBATCH -e SWAMP_SMC.e%j
#SBATCH -p gpu
# Sized for the actual ~1-2 h GPU-bound run (+ setup buffer) so the job BACKFILLS
# into short gaps instead of waiting for a 48 h opening. The GPU is the only hard
# requirement; raise -t/--mem if you scale the run way up.
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# =============================================================================
# Full GPU retrieval: differentiable SWAMP -> phase-curve -> BlackJAX adaptive
# tempered SMC (gradient-informed MALA kernel). Submit with `sbatch run.sh` on
# the JPL edge GPU cluster. The 'gpu' preset runs a large 512-particle swarm
# (vmapped -> the whole swarm advances at once on the A100/H100), 20-day spin-up,
# heteroscedastic photon noise, float64 -> smooth posteriors.
#
# Key correctness choices (from JAX GPU best-practice research):
#  * Do NOT `module load cuda` and do NOT set LD_LIBRARY_PATH. The `jax[cuda12]`
#    wheel bundles its own CUDA/cuDNN and finds them via rpath; a stray
#    LD_LIBRARY_PATH or system CUDA is the #1 cause of "No GPU found" / CPU fallback.
#  * Verify the backend is actually 'gpu' and ABORT (non-zero exit) otherwise, so
#    SLURM marks the job FAILED instead of silently burning hours on CPU.
#  * Single process (no srun fan-out / MPI): one GPU, one Python process.
#
# Configurable via env:
#   CONDA_ENV          conda env to activate            (default: MY_SWAMP)
#   JAX_VERSION        jax/jaxlib version to install     (default: 0.6.2, matches
#                      the jaxoplanet 0.1.0 the code is validated against)
#   SWAMP_RETRIEVAL_PRESET     fast | gpu | prod         (default: gpu)
#   SWAMP_RETRIEVAL_USE_X64    0 | 1                      (default: from preset)
#   SWAMP_RETRIEVAL_OVERRIDES  JSON of Config overrides   (e.g. tune N/steps)
#   SWAMP_SKIP_INSTALL 1 to skip the pip install step (env already set up)
#   SWAMP_SKIP_PLOTS   1 to skip figure generation after the run
# =============================================================================
set -euo pipefail

echo "======================================================"
echo "  Job info:  host=$(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}  $(date)"
echo "======================================================"

# ── Resolve layout: this script is MY_SWAMP/retrieval/scripts/run.sh ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # retrieval/scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                 # MY_SWAMP
cd "$SCRIPT_DIR"   # run_smc.py writes to retrieval/data/ and figures to retrieval/plots/

# ── Conda env (auto-created if missing) ────────────────
# Create the env from conda-forge with --override-channels. This AVOIDS the
# anaconda.com 'defaults' channel, which now returns HTTP 429 "TERMS OF SERVICE
# RATE LIMIT EXCEEDED" on many institutional clusters (incl. JPL). pip deps
# (jax[cuda12], jaxoplanet, blackjax) come from PyPI and are unaffected.
# Tip: pre-create the env once on the head node with the same command to keep
# this out of the job's wall-clock:
#   conda create -y -n swamp -c conda-forge --override-channels python=3.12 numpy scipy matplotlib
CONDA_ENV="${CONDA_ENV:-swamp}"
PYVER="${PYVER:-3.12}"
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Creating conda env '$CONDA_ENV' (python=$PYVER + numpy/scipy/matplotlib) from conda-forge..."
    conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels "python=$PYVER" numpy scipy matplotlib
  fi
  conda activate "$CONDA_ENV"
  echo "conda env: $CONDA_ENV  ($(which python))"
fi

# ── CRITICAL: let the jax[cuda12] wheel find its OWN bundled CUDA. ──
# A leftover LD_LIBRARY_PATH (or `module load cuda`) shadows the wheel's libs and
# silently drops JAX to CPU. Clear it. (Only keep system CUDA if you deliberately
# installed jax[cuda12-local] against it.)
unset LD_LIBRARY_PATH || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID    # do NOT touch CUDA_VISIBLE_DEVICES (SLURM sets it)

# ── Environment ────────────────────────────────────────
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"   # working-tree my_swamp
export MPLBACKEND=Agg
# One long job that owns the GPU: preallocate to avoid fragmentation through the
# vmapped scan; raise MEM_FRACTION if you OOM at steady state, lower if at startup.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_triton_gemm_any=true}"
export SWAMP_RETRIEVAL_PRESET="${SWAMP_RETRIEVAL_PRESET:-gpu}"

# ── Dependencies ───────────────────────────────────────
JAX_VERSION="${JAX_VERSION:-0.6.2}"
if [ "${SWAMP_SKIP_INSTALL:-0}" != "1" ]; then
  echo "------ installing GPU JAX + deps (jax[cuda12]==$JAX_VERSION) ------"
  python -m pip install --upgrade "jax[cuda12]==${JAX_VERSION}"
  # jaxoplanet's own requirements (equinox, jax, jaxlib) are all UNPINNED, so
  # installing it normally (not --no-deps) cannot touch the jax[cuda12] pin
  # above -- pip leaves an already-satisfied unpinned requirement alone. On a
  # genuinely fresh env, --no-deps here silently skips equinox (a hard
  # jaxoplanet dependency) and breaks the import.
  python -m pip install --upgrade jaxoplanet
  # Upper-bound blackjax: 1.4+ requires jax>=0.9.0/jaxlib>=0.9.0 (1.2.x/1.3.x
  # only need jax/jaxlib>=0.4.16, satisfied by the 0.6.2 pin above). An
  # unbounded install here silently drags jax/jaxlib up to satisfy blackjax
  # while jax-cuda12-plugin (installed above, pinned to JAX_VERSION) is left
  # behind -- the two then mismatch and JAX silently falls back to CPU.
  python -m pip install --upgrade "blackjax>=1.2,<1.4"
fi
# Fail loudly NOW if the version combo is broken, not mid-run.
python - <<'PY'
import jaxoplanet
from jaxoplanet.starry.light_curves import light_curve          # noqa: F401
from jaxoplanet.starry.surface import Surface                   # noqa: F401
import blackjax
from blackjax.smc import adaptive_tempered, resampling          # noqa: F401
print(f"jaxoplanet {jaxoplanet.__version__} | blackjax {getattr(blackjax,'__version__','?')} : imports OK")
PY

# ── GPU / JAX backend verification (ABORT if not GPU under SLURM) ──
echo "------ JAX backend check ------"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "(no nvidia-smi visible)"
python - <<'PY'
import os, sys, jax
be = jax.default_backend()
print(f"jax {jax.__version__} | backend={be} | devices={jax.devices()}")
on_cluster = bool(os.environ.get("SLURM_JOB_ID"))
if be != "gpu":
    msg = (f"FATAL: JAX backend is '{be}', expected 'gpu'. Check: NVIDIA driver >= 525, "
           "LD_LIBRARY_PATH unset, jax[cuda12] installed, a GPU is allocated.")
    if on_cluster:
        sys.exit(msg)            # non-zero -> SLURM marks FAILED, no wasted CPU hours
    print("WARNING (local run): " + msg)
PY

# ── Run the retrieval (single process; outputs -> retrieval/data/) ──
echo "======================================================"
echo "  run_smc.py  (preset=$SWAMP_RETRIEVAL_PRESET)"
echo "======================================================"
python -u run_smc.py

# ── Figures + summary (non-fatal) -> retrieval/plots/, retrieval/data/ ──
if [ "${SWAMP_SKIP_PLOTS:-0}" != "1" ]; then
  echo "------ generating figures + summary ------"
  python -u plot_smc.py       || echo "WARN: plot_smc.py failed (non-fatal)"
  python -u make_dashboard.py || echo "WARN: make_dashboard.py failed (non-fatal)"
  python -u summarize_run.py  || echo "WARN: summarize_run.py failed (non-fatal)"
fi
echo "DONE: outputs in $REPO_ROOT/retrieval/data , figures in $REPO_ROOT/retrieval/plots"
