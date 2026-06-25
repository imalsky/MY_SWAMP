#!/usr/bin/env python3
# ruff: noqa: E402
"""Run a user-invoked long-run parity comparison against legacy SWAMPE.

This script is intentionally not part of the pytest suite because a useful
integration horizon (for example 100 days) can take several minutes.

Default behavior
----------------
- Uses the repository's "forced default" parameter set.
- Runs a 100-day integration on CPU in float64 mode.
- Executes both legacy `SWAMPE` and `my_swamp`.
- Writes comparison fields plus error summaries to `testing/long_run_parity_outputs/`.

Example
-------
    python testing/compare_long_run_parity.py --days 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = ROOT.parents[0] / "SWAMPE"
DEFAULT_OUT_DIR = ROOT / "testing" / "long_run_parity_outputs" / "forced_default_100d"


# Keep parity runs deterministic and aligned with the repository contract.
# These environment settings must execute before importing JAX / pyplot.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


import numpy as np
import scipy.special as sp
import matplotlib
from jax import config as jax_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt


jax_config.update("jax_enable_x64", True)


def _ensure_lpmn_compat() -> None:
    """Patch SciPy 1.17+ for legacy SWAMPE's `sp.lpmn` dependency."""
    if hasattr(sp, "lpmn"):
        return

    def _lpmn_compat(M: int, N: int, x: float):
        """Compatibility shim for `scipy.special.lpmn`."""
        M = int(M)
        N = int(N)
        P = np.zeros((M + 1, N + 1), dtype=np.float64)
        dP = np.zeros((M + 1, N + 1), dtype=np.float64)
        P[0, 0] = 1.0

        s = math.sqrt(max(0.0, 1.0 - x * x))
        for m in range(1, M + 1):
            if m <= N:
                P[m, m] = -(2 * m - 1) * s * P[m - 1, m - 1]

        for m in range(0, min(M, N - 1) + 1):
            P[m, m + 1] = (2 * m + 1) * x * P[m, m]

        for m in range(0, M + 1):
            for n in range(m + 2, N + 1):
                P[m, n] = ((2 * n - 1) * x * P[m, n - 1] - (n + m - 1) * P[m, n - 2]) / (n - m)

        denom = x * x - 1.0
        if denom == 0.0:
            denom = np.finfo(np.float64).eps

        for m in range(0, M + 1):
            if m <= N:
                dP[m, m] = 0.0 if m == 0 else (m * x * P[m, m]) / denom
            for n in range(m + 1, N + 1):
                dP[m, n] = (n * x * P[m, n] - (n + m) * P[m, n - 1]) / denom
        return P, dP

    sp.lpmn = _lpmn_compat


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the long-run parity comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=10.0, help="Integration horizon in physical days.")
    parser.add_argument("--dt", type=float, default=120.0, help="Timestep in seconds.")
    parser.add_argument("--M", type=int, default=42)
    parser.add_argument("--Phibar", type=float, default=3.0e5)
    parser.add_argument("--omega", type=float, default=3.2e-5)
    parser.add_argument("--a", type=float, default=8.2e7)
    parser.add_argument("--g", type=float, default=9.8)
    parser.add_argument("--taurad", type=float, default=10.0 * 3600.0)
    parser.add_argument("--taudrag", type=float, default=6.0 * 3600.0)
    parser.add_argument("--DPhieq", type=float, default=1.0e6)
    parser.add_argument("--K6", type=float, default=1.24e33)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--test", type=int, default=0, help="0 means forced mode; 1 or 2 select idealized tests.")
    parser.add_argument("--forcflag", type=int, default=1, choices=(0, 1))
    parser.add_argument("--diffflag", type=int, default=1, choices=(0, 1))
    parser.add_argument("--modalflag", type=int, default=1, choices=(0, 1))
    parser.add_argument("--expflag", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--rel-floor-frac",
        type=float,
        default=1.0e-12,
        help="Elementwise fractional-error denominator floor as a fraction of each field's max |ref|.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for summary JSON and comparison field NPZ output.",
    )
    return parser.parse_args()


def _derive_tmax(days: float, dt: float) -> tuple[int, int, float]:
    """Convert a duration and timestep into solver step counts."""
    if days <= 0.0:
        raise ValueError("--days must be positive.")
    if dt <= 0.0:
        raise ValueError("--dt must be positive.")

    target_seconds = days * 86400.0
    n_steps = int(round(target_seconds / dt))
    if n_steps < 2:
        raise ValueError("Derived step count must be at least 2.")

    actual_days = (n_steps * dt) / 86400.0
    return n_steps + 1, n_steps, actual_days


def _field_summary(ref: np.ndarray, got: np.ndarray, rel_floor_frac: float) -> Dict[str, float]:
    """Summarize absolute and RMS differences for one compared field."""
    err = got - ref
    abs_err = np.abs(err)
    ref_abs = np.abs(ref)

    ref_max_abs = float(np.max(ref_abs))
    denom_floor = max(rel_floor_frac * max(ref_max_abs, 1.0), np.finfo(np.float64).tiny)
    frac_err = abs_err / np.maximum(ref_abs, denom_floor)

    ref_l2 = float(np.linalg.norm(ref.ravel()))
    err_l2 = float(np.linalg.norm(err.ravel()))
    l2_denom = max(ref_l2, denom_floor * math.sqrt(ref.size))

    return {
        "ref_max_abs": ref_max_abs,
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "rms_abs_error": float(np.sqrt(np.mean(abs_err * abs_err))),
        "max_fractional_error": float(np.max(frac_err)),
        "mean_fractional_error": float(np.mean(frac_err)),
        "rms_fractional_error": float(np.sqrt(np.mean(frac_err * frac_err))),
        "relative_l2_error": err_l2 / l2_denom,
    }


def _save_field_comparison_plot(
    ref_fields: Dict[str, np.ndarray],
    got_fields: Dict[str, np.ndarray],
    rel_floor_frac: float,
    actual_days: float,
    out_path: Path,
) -> None:
    """Save field comparison plot."""
    fields = ("eta", "delta", "Phi", "U", "V")

    style_path = Path(__file__).resolve().parent / "science.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
    plt.rcParams.update(
        {
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "font.size": 14,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(fields),
        ncols=3,
        figsize=(12.0, 3.0 * len(fields)),
        constrained_layout=True,
    )

    for row, field in enumerate(fields):
        ref = ref_fields[field]
        got = got_fields[field]
        ref_abs = np.abs(ref)
        ref_max_abs = float(np.max(ref_abs))
        denom_floor = max(rel_floor_frac * max(ref_max_abs, 1.0), np.finfo(np.float64).tiny)
        frac_diff = (got - ref) / np.maximum(ref_abs, denom_floor)

        value_lim = max(float(np.max(np.abs(ref))), float(np.max(np.abs(got))), np.finfo(np.float64).tiny)
        frac_lim = max(float(np.max(np.abs(frac_diff))), np.finfo(np.float64).tiny)

        ref_ax = axes[row, 0]
        got_ax = axes[row, 1]
        frac_ax = axes[row, 2]

        ref_im = ref_ax.imshow(ref, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-value_lim, vmax=value_lim)
        got_im = got_ax.imshow(got, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-value_lim, vmax=value_lim)
        frac_im = frac_ax.imshow(
            frac_diff,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-frac_lim,
            vmax=frac_lim,
        )

        ref_ax.set_ylabel(field, fontsize=22)
        for ax in (ref_ax, got_ax, frac_ax):
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(ref_im, ax=ref_ax, shrink=0.82)
        fig.colorbar(got_im, ax=got_ax, shrink=0.82)
        fig.colorbar(frac_im, ax=frac_ax, shrink=0.82)

    axes[0, 0].set_title("SWAMPE")
    axes[0, 1].set_title("SWAMPE-JAX")
    axes[0, 2].set_title("Fractional difference")
    fig.suptitle(f"SWAMPE vs SWAMPE-JAX field comparison ({actual_days:.1f} days)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _legacy_comparison_fields(custompath: str, final_step: int, dt: float) -> Dict[str, np.ndarray]:
    """Load the reference SWAMPE comparison fields for the configured run."""
    import SWAMPE.continuation as ref_cont

    ts = str(int(final_step * dt))
    return {
        "eta": np.asarray(ref_cont.read_pickle(f"eta-{ts}", custompath=custompath), dtype=np.float64),
        "delta": np.asarray(ref_cont.read_pickle(f"delta-{ts}", custompath=custompath), dtype=np.float64),
        "Phi": np.asarray(ref_cont.read_pickle(f"Phi-{ts}", custompath=custompath), dtype=np.float64),
        "U": np.asarray(ref_cont.read_pickle(f"U-{ts}", custompath=custompath), dtype=np.float64),
        "V": np.asarray(ref_cont.read_pickle(f"V-{ts}", custompath=custompath), dtype=np.float64),
    }


def _run_reference(kwargs: Dict[str, Any], final_step: int) -> tuple[Dict[str, np.ndarray], float]:
    """Run the reference NumPy SWAMPE model."""
    _ensure_lpmn_compat()
    if str(REF_ROOT) not in sys.path:
        sys.path.insert(0, str(REF_ROOT))

    import SWAMPE.model as ref_model

    with tempfile.TemporaryDirectory(prefix="swampe_long_run_", dir="/tmp") as tmp:
        custompath = f"{tmp}/"
        t0 = time.perf_counter()
        ref_model.run_model(
            plotflag=False,
            saveflag=True,
            savefreq=final_step,
            timeunits="seconds",
            custompath=custompath,
            verbose=False,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        fields = _legacy_comparison_fields(custompath, final_step=final_step, dt=float(kwargs["dt"]))
    return fields, elapsed


def _run_my_swamp(kwargs: Dict[str, Any]) -> tuple[Dict[str, np.ndarray], float]:
    """Run the MY_SWAMP model under the configured settings."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    from my_swamp.model import run_model_scan_final

    t0 = time.perf_counter()
    out = run_model_scan_final(
        diagnostics=False,
        jit_scan=True,
        **kwargs,
    )
    last = out["last_state"]
    fields = {
        "eta": np.asarray(last.eta_curr, dtype=np.float64),
        "delta": np.asarray(last.delta_curr, dtype=np.float64),
        "Phi": np.asarray(last.Phi_curr, dtype=np.float64),
        "U": np.asarray(last.U_curr, dtype=np.float64),
        "V": np.asarray(last.V_curr, dtype=np.float64),
    }
    elapsed = time.perf_counter() - t0
    return fields, elapsed


def _json_ready(value: Any) -> Any:
    """Convert NumPy and pathlib values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    """Run the long-run SWAMPE parity comparison and write summary artifacts."""
    args = _parse_args()
    tmax, final_step, actual_days = _derive_tmax(args.days, args.dt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_value = None if int(args.test) == 0 else int(args.test)

    # Test cases have no physical forcing, so forcflag must be False.
    # With forcflag=True the modified-Euler eta coefficient is 2x larger
    # (designed for when drag damping is present).  Without that damping the
    # forward-Euler amplification at high wavenumbers overwhelms the
    # hyperdiffusion filter and both codes blow up.
    forcflag = bool(args.forcflag) if test_value is None else False

    model_kwargs: Dict[str, Any] = dict(
        M=int(args.M),
        dt=float(args.dt),
        tmax=int(tmax),
        Phibar=float(args.Phibar),
        omega=float(args.omega),
        a=float(args.a),
        test=test_value,
        g=float(args.g),
        forcflag=forcflag,
        taurad=float(args.taurad),
        taudrag=float(args.taudrag),
        DPhieq=float(args.DPhieq),
        diffflag=bool(args.diffflag),
        modalflag=bool(args.modalflag),
        alpha=float(args.alpha),
        expflag=bool(args.expflag),
        K6=float(args.K6),
    )

    print("Running long-run parity comparison")
    print(f"output_dir={out_dir}")
    print(f"days_requested={args.days}")
    print(f"days_actual={actual_days}")
    print(f"dt_seconds={args.dt}")
    print(f"final_step={final_step}")
    print(f"tmax={tmax}")
    print(f"mode={'forced' if test_value is None else f'test={test_value}'}")
    print(f"forcflag={forcflag}")

    ref_fields, ref_seconds = _run_reference(model_kwargs, final_step=final_step)
    got_fields, my_seconds = _run_my_swamp(model_kwargs)

    metrics = {
        field: _field_summary(ref_fields[field], got_fields[field], rel_floor_frac=float(args.rel_floor_frac))
        for field in ("eta", "delta", "Phi", "U", "V")
    }
    plot_path = out_dir / "field_comparison.png"
    _save_field_comparison_plot(
        ref_fields=ref_fields,
        got_fields=got_fields,
        rel_floor_frac=float(args.rel_floor_frac),
        actual_days=actual_days,
        out_path=plot_path,
    )

    summary = {
        "comparison": "SWAMPE vs MY_SWAMP long-run field comparison",
        "output_dir": str(out_dir.resolve().relative_to(Path.cwd())),
        "plot_png": str(plot_path.resolve().relative_to(Path.cwd())),
        "days_requested": float(args.days),
        "days_actual": actual_days,
        "dt_seconds": float(args.dt),
        "final_step": int(final_step),
        "tmax": int(tmax),
        "runtime_seconds": {
            "swampe": ref_seconds,
            "my_swamp": my_seconds,
        },
        "params": model_kwargs,
        "metrics": metrics,
    }

    np.savez_compressed(
        out_dir / "comparison_fields.npz",
        ref_eta=ref_fields["eta"],
        ref_delta=ref_fields["delta"],
        ref_Phi=ref_fields["Phi"],
        ref_U=ref_fields["U"],
        ref_V=ref_fields["V"],
        my_eta=got_fields["eta"],
        my_delta=got_fields["delta"],
        my_Phi=got_fields["Phi"],
        my_U=got_fields["U"],
        my_V=got_fields["V"],
        abs_err_eta=np.abs(got_fields["eta"] - ref_fields["eta"]),
        abs_err_delta=np.abs(got_fields["delta"] - ref_fields["delta"]),
        abs_err_Phi=np.abs(got_fields["Phi"] - ref_fields["Phi"]),
        abs_err_U=np.abs(got_fields["U"] - ref_fields["U"]),
        abs_err_V=np.abs(got_fields["V"] - ref_fields["V"]),
    )
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("Runtime")
    print(f"  SWAMPE:   {ref_seconds:.3f} s")
    print(f"  MY_SWAMP: {my_seconds:.3f} s")
    print()
    print("Field comparison errors")
    for field in ("eta", "delta", "Phi", "U", "V"):
        m = metrics[field]
        print(
            f"  {field:5s} "
            f"rel_l2={m['relative_l2_error']:.3e} "
            f"max_frac={m['max_fractional_error']:.3e} "
            f"mean_frac={m['mean_fractional_error']:.3e} "
            f"rms_frac={m['rms_fractional_error']:.3e} "
            f"max_abs={m['max_abs_error']:.3e}"
        )

    print()
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'comparison_fields.npz'}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
