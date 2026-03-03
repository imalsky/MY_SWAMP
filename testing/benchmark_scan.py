#!/usr/bin/env python3
"""Deterministic benchmark harness for my_swamp scan execution."""

from __future__ import annotations

import argparse
import statistics
import time
import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from my_swamp.backend_preflight import backend_info_lines, preflight_backend  # noqa: E402
from my_swamp.model import run_model_scan_final  # noqa: E402


def _bool_arg(x: str) -> bool:
    v = str(x).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x!r}")


def _run_once(args: argparse.Namespace):
    res = run_model_scan_final(
        M=args.M,
        dt=args.dt,
        tmax=args.tmax,
        Phibar=args.Phibar,
        omega=args.omega,
        a=args.a,
        test=args.test,
        g=args.g,
        forcflag=args.forcflag,
        taurad=args.taurad,
        taudrag=args.taudrag,
        DPhieq=args.DPhieq,
        diffflag=args.diffflag,
        modalflag=args.modalflag,
        alpha=args.alpha,
        expflag=args.expflag,
        K6=args.K6,
        diagnostics=args.diagnostics,
        remat_step=args.remat_step,
        jit_scan=True,
        donate_state=args.donate_state,
    )
    # Block on terminal map to include full runtime.
    _ = res["last_state"].Phi_curr.block_until_ready()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=str, default="cpu", help="Requested JAX backend (cpu/gpu/tpu).")
    parser.add_argument("--require-gpu", action="store_true", help="Fail fast when GPU backend is unavailable.")
    parser.add_argument("--require-x64", action="store_true", help="Fail fast unless jax_enable_x64=True.")

    parser.add_argument("--M", type=int, default=42)
    parser.add_argument("--dt", type=float, default=30.0)
    parser.add_argument("--tmax", type=int, default=300)
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--Phibar", type=float, default=3.0e3)
    parser.add_argument("--omega", type=float, default=7.2921159e-5)
    parser.add_argument("--a", type=float, default=6.37122e6)
    parser.add_argument("--g", type=float, default=9.8)
    parser.add_argument("--forcflag", type=_bool_arg, default=False)
    parser.add_argument("--diffflag", type=_bool_arg, default=False)
    parser.add_argument("--modalflag", type=_bool_arg, default=True)
    parser.add_argument("--expflag", type=_bool_arg, default=False)
    parser.add_argument("--diagnostics", type=_bool_arg, default=False)
    parser.add_argument("--remat-step", type=_bool_arg, default=False)
    parser.add_argument("--donate-state", type=_bool_arg, default=False)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--taurad", type=float, default=86400.0)
    parser.add_argument("--taudrag", type=float, default=86400.0)
    parser.add_argument("--DPhieq", type=float, default=4.0e6)
    parser.add_argument("--K6", type=float, default=1.24e33)

    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    args = parser.parse_args()

    if args.tmax < 3:
        raise ValueError("--tmax must be >= 3.")

    info = preflight_backend(args.backend, require_gpu=args.require_gpu)
    for line in backend_info_lines(info):
        print(f"backend.{line}")

    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    if args.require_x64 and not x64_enabled:
        raise RuntimeError("Requested --require-x64, but jax_enable_x64 is False.")

    print(f"benchmark.M={args.M}")
    print(f"benchmark.dt={args.dt}")
    print(f"benchmark.tmax={args.tmax}")
    print(f"benchmark.n_steps={args.tmax - 2}")
    print(f"benchmark.dtype_mode={'float64' if x64_enabled else 'float32'}")
    print(
        "benchmark.flags="
        f"forc:{args.forcflag},diff:{args.diffflag},modal:{args.modalflag},"
        f"exp:{args.expflag},diagnostics:{args.diagnostics},remat:{args.remat_step}"
    )
    print(f"benchmark.warmup_runs={args.warmup_runs}")
    print(f"benchmark.timed_runs={args.timed_runs}")

    compile_t0 = time.perf_counter()
    _run_once(args)
    compile_s = time.perf_counter() - compile_t0

    # Optional extra warmups after compile.
    for _ in range(max(0, int(args.warmup_runs) - 1)):
        _run_once(args)

    runtimes = []
    for _ in range(int(args.timed_runs)):
        t0 = time.perf_counter()
        _run_once(args)
        runtimes.append(time.perf_counter() - t0)

    rt_mean = statistics.mean(runtimes)
    rt_median = statistics.median(runtimes)
    rt_min = min(runtimes)
    rt_max = max(runtimes)
    per_step_ms = (rt_median / max(1, args.tmax - 2)) * 1000.0

    print(f"result.compile_seconds={compile_s:.6f}")
    print(f"result.runtime_mean_seconds={rt_mean:.6f}")
    print(f"result.runtime_median_seconds={rt_median:.6f}")
    print(f"result.runtime_min_seconds={rt_min:.6f}")
    print(f"result.runtime_max_seconds={rt_max:.6f}")
    print(f"result.per_step_median_ms={per_step_ms:.6f}")


if __name__ == "__main__":
    main()
