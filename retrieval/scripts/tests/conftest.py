"""Pytest setup for the retrieval test suite.

These tests exercise ``retrieval/pipeline.py`` (the differentiable SWAMP ->
phase-curve retrieval). They run in float32 (fast) by default; the env var must
be set BEFORE jax / my_swamp import, so we set it here at module top and add the
package + retrieval dirs to ``sys.path`` (mirroring my_swamp's own conftest).

Run from the repo root in the project conda env (jaxoplanet + blackjax):

    conda run -n MY_SWAMP python -m pytest retrieval/tests -q
"""

import os
import sys
from pathlib import Path

# Float32 fast mode for the suite. Must precede any jax/my_swamp import.
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "0")
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = _HERE.parent.parent            # MY_SWAMP/retrieval/scripts (has pipeline.py)
_REPO_ROOT = _SCRIPTS_DIR.parent.parent       # MY_SWAMP
for p in (str(_REPO_ROOT / "src"), str(_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pipeline as P  # noqa: E402


def pytest_configure(config):
    """Register the `slow` marker (the end-to-end SMC test); deselect with -m 'not slow'."""
    config.addinivalue_line("markers", "slow: end-to-end SMC run (minutes); deselect with -m 'not slow'")


@pytest.fixture(scope="session")
def cfg_fast():
    """A small, fast config: 1-day spin-up, 2 timescales inferred (fast path)."""
    return P.fast_cpu_config(model_days=1.0, n_times=80,
                             smc_num_particles=12, smc_num_mcmc_steps=4,
                             smc_max_steps=14, num_samples=12,
                             mcmc_tune_particles=6, mcmc_tune_steps=4, mcmc_tune_iters=4,
                             do_ppc=False)


@pytest.fixture(scope="session")
def pipe(cfg_fast):
    """Built pipeline (shared; build is ~8 s) with synthetic observations injected."""
    pp = P.build_pipeline(cfg_fast)
    P.generate_observations(pp, seed=cfg_fast.seed)
    return pp


def u_for_theta(pipe, theta):
    """Invert theta_from_u on the grid (uniform/log10 sigmoid map) to get u-space coords."""
    theta = np.asarray(theta, dtype=float)
    lo = np.asarray(pipe.param_prior_lo, dtype=float)
    hi = np.asarray(pipe.param_prior_hi, dtype=float)
    z = np.empty_like(theta)
    for i, spec in enumerate(pipe.specs):
        if spec.prior_type == "uniform":
            z[i] = (theta[i] - lo[i]) / (hi[i] - lo[i])
        else:  # log10_uniform
            z[i] = (np.log10(theta[i]) - np.log10(lo[i])) / (np.log10(hi[i]) - np.log10(lo[i]))
    z = np.clip(z, 1e-6, 1 - 1e-6)
    return np.log(z) - np.log1p(-z)
