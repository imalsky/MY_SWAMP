#!/usr/bin/env python3
"""
tests/testing.py

Compare upstream SWAMPE (NumPy reference) vs MY_SWAMP (JAX port) and write plots
to repo_root/plots/.

Changes vs prior version:
- Fix quiver locations by using physical lon/lat grids (meshgrid) + matching imshow extent.
- Produce 3-panel plots (SWAMPE | JAX | relative diff) for final state only:
    * geopotential (Phi + Phibar) with U/V quivers overlaid on first two panels
    * U, V, and W wind speed magnitude (sqrt(U^2+V^2))
- Shorter code, minimal validation.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt  # noqa: E402


FIELDS = ("eta", "delta", "Phi", "U", "V")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path_for_dev() -> None:
    src = _repo_root() / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _get_run_model(pkg: str):
    for mod in ("model", "main_function", ""):
        m = importlib.import_module(pkg) if mod == "" else importlib.import_module(f"{pkg}.{mod}")
        fn = getattr(m, "run_model", None)
        if callable(fn):
            return fn
    raise AttributeError(f"Could not find run_model in {pkg!r}")


def _filter_kwargs(fn, params):
    sig = inspect.signature(fn)
    ps = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values()):
        return dict(params)
    allowed = {
        k
        for k, p in ps.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in params.items() if k in allowed}


def _final2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if np.iscomplexobj(a):
        a = np.real(a)
    return a[-1] if a.ndim == 3 else a


def _find_latest_timestamp(custompath: Path) -> str:
    ts = []
    for p in custompath.iterdir():
        m = re.match(r"^Phi-(\d+)$", p.name)
        if m:
            ts.append(int(m.group(1)))
    if not ts:
        raise FileNotFoundError(f"No Phi-<timestamp> files found in {custompath}")
    return str(max(ts))


def _load_saved_fields(custompath: Path) -> dict[str, np.ndarray]:
    ts = _find_latest_timestamp(custompath)
    out = {}
    for k in FIELDS:
        fp = custompath / f"{k}-{ts}"
        with fp.open("rb") as f:
            out[k] = np.asarray(pickle.load(f))
    return out


def _normalize_outputs(out) -> dict[str, np.ndarray] | None:
    if out is None:
        return None
    if isinstance(out, dict):
        return {k: np.asarray(out[k]) for k in FIELDS if k in out}
    if isinstance(out, tuple) and len(out) >= 5:
        eta, delta, Phi, U, V = out[:5]
        return {
            "eta": np.asarray(eta),
            "delta": np.asarray(delta),
            "Phi": np.asarray(Phi),
            "U": np.asarray(U),
            "V": np.asarray(V),
        }
    raise TypeError(f"Unsupported run_model return type: {type(out)}")


def _run_pkg(pkg: str, params: dict, outdir: Path, label: str) -> dict[str, np.ndarray]:
    if outdir.exists():
        for p in outdir.iterdir():
            if p.is_file():
                p.unlink()
    outdir.mkdir(parents=True, exist_ok=True)

    p = dict(params)
    p["custompath"] = str(outdir) + os.sep
    p["timeunits"] = "seconds"
    p["saveflag"] = True
    p["savefreq"] = 1
    p["plotflag"] = False

    fn = _get_run_model(pkg)
    kwargs = _filter_kwargs(fn, p)

    t0 = time.time()
    out_raw = fn(**kwargs)
    t1 = time.time()
    print(f"{label} runtime: {t1 - t0:.3f} s ({pkg})")

    out = _normalize_outputs(out_raw)
    return out if out is not None else _load_saved_fields(outdir)


def _three_panel(
    name: str,
    sw: np.ndarray,
    jx: np.ndarray,
    plots_dir: Path,
    *,
    extent=None,
    quiver=None,  # (Xq, Yq, Usw, Vsw, Ujx, Vjx, scale)
) -> None:
    sw = _final2d(sw)
    jx = _final2d(jx)

    vmin = float(min(np.min(sw), np.min(jx)))
    vmax = float(max(np.max(sw), np.max(jx)))

    denom = np.where(sw != 0.0, sw, np.nan)
    rel = (jx - sw) / denom
    rmax = float(np.nanmax(np.abs(rel))) if np.isfinite(rel).any() else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(sw, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title(f"{name} | SWAMPE")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(jx, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title(f"{name} | JAX")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(rel, origin="lower", aspect="auto", vmin=-rmax, vmax=rmax, extent=extent)
    axes[2].set_title(f"{name} | (JAX - SWAMPE)/SWAMPE")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if quiver is not None:
        Xq, Yq, Usw, Vsw, Ujx, Vjx, qscale = quiver
        axes[0].quiver(Xq, Yq, Usw, Vsw, pivot="mid", scale=qscale)
        axes[1].quiver(Xq, Yq, Ujx, Vjx, pivot="mid", scale=qscale)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(plots_dir / f"{name}_3panel.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=1, choices=[1, 2])
    ap.add_argument("--tmax", type=int, default=50)
    ap.add_argument("--M", type=int, default=42)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--plots_dir", type=str, default="plots")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    params = {
        "M": args.M,
        "dt": args.dt,
        "tmax": args.tmax,
        "test": args.test,
        "a": 6.37122e6,
        "omega": 7.2921159e-5,
        "g": 9.80616,
        "Phibar": 3.0e5,
        "a1": 0.05,
        "K6": 1.24e33,
        "K6Phi": 1.24e33,
        "taurad": 86400.0,
        "taudrag": 86400.0,
        "DPhieq": 4.0e6,
        "forcflag": True,
        "diffflag": True,
        "expflag": False,
        "modalflag": True,
        "alpha": 0.01,
        "plotfreq": 999999,
        "verbose": True,
    }

    numpy_pkg = os.environ.get("SWAMPE_NUMPY_PKG", "SWAMPE")
    jax_pkg = os.environ.get("MY_SWAMP_PKG", "my_swamp")

    try:
        _get_run_model(jax_pkg)
    except Exception:
        _ensure_src_on_path_for_dev()

    repo = _repo_root()
    plots_dir = (repo / args.plots_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_root = repo / "tests" / "_compare_outputs"
    out_sw = _run_pkg(numpy_pkg, params, out_root / "numpy", "SWAMPE")
    out_jx = _run_pkg(jax_pkg, params, out_root / "jax", "JAX")

    # lon/lat grids for physically-placed quivers + matching imshow extent
    X = Y = extent = Xq = Yq = None
    try:
        ic = importlib.import_module(f"{jax_pkg}.initial_conditions")
        _N, _I, _J, _dt0, lambdas, mus, _w = ic.spectral_params(args.M)
        X = np.asarray(lambdas) * 180.0 / np.pi
        Y = np.arcsin(np.asarray(mus)) * 180.0 / np.pi
        extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]
        s = max(1, int(args.stride))
        Xq, Yq = np.meshgrid(X[::s], Y[::s])
    except Exception:
        pass

    Phi_sw = _final2d(out_sw["Phi"]) + float(params["Phibar"])
    Phi_jx = _final2d(out_jx["Phi"]) + float(params["Phibar"])
    U_sw = _final2d(out_sw["U"])
    V_sw = _final2d(out_sw["V"])
    U_jx = _final2d(out_jx["U"])
    V_jx = _final2d(out_jx["V"])

    s = max(1, int(args.stride))
    if Xq is not None:
        quiv = (Xq, Yq, U_sw[::s, ::s], V_sw[::s, ::s], U_jx[::s, ::s], V_jx[::s, ::s], 600.0)
    else:
        quiv = None

    _three_panel("geopotential_Phi", Phi_sw, Phi_jx, plots_dir, extent=extent, quiver=quiv)
    _three_panel("U_wind", U_sw, U_jx, plots_dir, extent=extent)
    _three_panel("V_wind", V_sw, V_jx, plots_dir, extent=extent)

    W_sw = np.sqrt(U_sw * U_sw + V_sw * V_sw)
    W_jx = np.sqrt(U_jx * U_jx + V_jx * V_jx)
    _three_panel("W_speed", W_sw, W_jx, plots_dir, extent=extent)

    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
