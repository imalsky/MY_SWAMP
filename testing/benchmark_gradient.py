#!/usr/bin/env python3
"""Benchmark reverse-mode gradient cost and vmap throughput for my_swamp.

This is the reproducible source for the paper's differentiability performance
claims: the cost of a reverse-mode gradient of a scalar loss with respect to the
physical parameters, relative to a single forward integration, and the vmap
ensemble throughput. It runs the forced hot-Jupiter regime of the parity figure.

The loss is the mean terminal geopotential, differentiated with respect to four
physical controls (tau_rad, tau_drag, DPhieq, Phibar). Reverse-mode cost is
independent of the number of parameters, so the ratio is the headline number;
absolute seconds are machine-dependent and reported with the device/version.

Usage
-----
    python testing/benchmark_gradient.py --days 10
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_swamp.model import run_model_scan_final  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Forced hot-Jupiter regime (matches the parity and sensitivity figures).
M = 42
DT = 120.0
DAY = 86400.0
FIXED = dict(
    M=M, dt=DT, Phibar=3.0e5, omega=3.2e-5, a=8.2e7, g=9.8, test=None,
    forcflag=True, diffflag=True, modalflag=True, alpha=0.01, expflag=False,
    K6=1.24e33, diagnostics=False, jit_scan=True,
)
# Parameter vector: [tau_rad, tau_drag, DPhieq, Phibar].
THETA0 = jnp.asarray([10.0 * 3600.0, 6.0 * 3600.0, 1.0e6, 3.0e5])


def _median_time(fn, args, warmup, repeats) -> float:
    """Median wall-clock of fn(*args), blocking on the result, after warmup."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16, help="vmap ensemble size.")
    args = parser.parse_args()

    tmax = int(round(args.days * DAY / DT)) + 1

    def loss(theta):
        """Mean terminal geopotential as a scalar function of four parameters."""
        taurad, taudrag, dphieq, phibar = theta[0], theta[1], theta[2], theta[3]
        kw = dict(FIXED)
        kw["Phibar"] = phibar
        out = run_model_scan_final(taurad=taurad, taudrag=taudrag, DPhieq=dphieq,
                                   tmax=tmax, **kw)
        return jnp.mean(out["last_state"].Phi_curr)

    # Warm the geometry cache eagerly (concrete M) so build_static is not traced
    # under jit -- otherwise the spectral basis is built on tracers and fails.
    jax.block_until_ready(loss(THETA0))

    fwd = jax.jit(loss)
    vg = jax.jit(jax.value_and_grad(loss))

    print(f"device={jax.devices()[0].platform} jax={jax.__version__} "
          f"machine={platform.processor() or platform.machine()}")
    print(f"regime=forced_hot_jupiter M={M} dt={DT} days={args.days} tmax={tmax} "
          f"n_params={THETA0.shape[0]} x64={jax.config.read('jax_enable_x64')}")

    t_fwd = _median_time(fwd, (THETA0,), args.warmup, args.repeats)
    t_grad = _median_time(vg, (THETA0,), args.warmup, args.repeats)
    print(f"forward_median_s={t_fwd:.3f}")
    print(f"grad_median_s={t_grad:.3f}")
    print(f"grad_over_forward={t_grad / t_fwd:.2f}")
    print(f"fd_forward_runs_equiv={THETA0.shape[0] + 1}  "
          f"(one-sided finite difference over {THETA0.shape[0]} params)")

    # vmap ensemble throughput: many parameter sets in one compiled call.
    batch = jnp.broadcast_to(THETA0, (args.batch, THETA0.shape[0]))
    vbatch = jax.jit(jax.vmap(loss))
    t_vmap = _median_time(vbatch, (batch,), args.warmup, args.repeats)
    print(f"vmap_batch={args.batch} vmap_median_s={t_vmap:.3f} "
          f"throughput_sims_per_s={args.batch / t_vmap:.2f} "
          f"speedup_vs_serial={(args.batch * t_fwd) / t_vmap:.1f}x")


if __name__ == "__main__":
    main()
