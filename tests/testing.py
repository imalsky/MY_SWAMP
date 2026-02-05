#!/usr/bin/env python3
"""
tests/testing.py

Compare upstream SWAMPE (NumPy reference) vs MY_SWAMP (JAX port), and save
side-by-side plots into repo_root/plots/.

Key points:
- Upstream SWAMPE's run_model() returns None (it saves pickles). This script
  detects that and loads the latest saved fields from custompath.
- Your JAX port may return arrays directly; if it returns None, we load pickles
  the same way.
- All plots are saved (matplotlib Agg backend), no interactive windows.

Repo layout assumed:

repo/
  src/
    my_swamp/
      ...
  tests/
    testing.py  (this file)
  plots/        (created automatically)

Run:
  python tests/testing.py --test 1 --tmax 50 --M 42
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pickle
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

# Ensure non-interactive plotting and consistent backend.
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from jax import config as _jax_config

# Enable 64-bit before importing any JAX implementation.
_jax_config.update("jax_enable_x64", True)

FIELDS = ("eta", "delta", "Phi", "U", "V")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path_for_dev() -> None:
    """
    Preferred: install MY_SWAMP via `pip install -e .` so imports work.

    This fallback lets you run tests before installing by adding repo_root/src to sys.path.
    """
    src = _repo_root() / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _import_module(pkg_name: str, suffix: str) -> Any:
    return importlib.import_module(f"{pkg_name}.{suffix}")


def _get_run_model(pkg_name: str) -> Tuple[Any, str, Any]:
    """
    Locate a run_model callable for a package.

    We try common locations:
      - <pkg>.model.run_model
      - <pkg>.main_function.run_model
      - <pkg>.run_model (top-level)

    Returns: (callable, origin_string, module_object_where_found)
    """
    candidates = ("model", "main_function", "")
    last_err: Optional[Exception] = None

    for mod in candidates:
        try:
            m = importlib.import_module(pkg_name) if mod == "" else _import_module(pkg_name, mod)
        except ModuleNotFoundError as e:
            last_err = e
            continue

        if hasattr(m, "run_model") and callable(getattr(m, "run_model")):
            return getattr(m, "run_model"), f"{m.__name__}.run_model", m

    if last_err is not None:
        raise last_err
    raise AttributeError(f"Could not find a callable run_model in package {pkg_name!r}")


def _filter_kwargs_by_signature(
    fn: Any, params: Mapping[str, Any], *, label: str
) -> Tuple[Tuple[Any, ...], Dict[str, Any], Tuple[str, ...]]:
    """
    Return (args, kwargs, dropped_keys) such that fn(*args, **kwargs) is valid.

    Strict:
      - if fn has required params we don't provide => raise TypeError
      - handles positional-only params by name from params
    """
    sig = inspect.signature(fn)
    ps = sig.parameters

    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values())
    posonly = [p for p in ps.values() if p.kind == inspect.Parameter.POSITIONAL_ONLY]

    args: list[Any] = []
    kwargs: Dict[str, Any] = {}

    if posonly:
        for p in posonly:
            if p.name not in params:
                raise TypeError(f"{label}: run_model requires positional-only arg {p.name!r} missing from params.")
            args.append(params[p.name])

    if accepts_var_kw:
        kwargs = dict(params)
        dropped: Tuple[str, ...] = ()
    else:
        allowed = {
            name
            for name, p in ps.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        kwargs = {k: v for k, v in params.items() if k in allowed}
        dropped = tuple(sorted(set(params.keys()) - set(kwargs.keys()) - {p.name for p in posonly}))

    # Ensure required non-variadic params are present
    missing_required = []
    for name, p in ps.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
            continue
        if p.default is not inspect._empty:
            continue
        if name not in kwargs:
            missing_required.append(name)

    if missing_required:
        raise TypeError(
            f"{label}: run_model signature requires {missing_required}, but they are not present in params. "
            f"Provided keys: {sorted(params.keys())}"
        )

    return tuple(args), kwargs, dropped


def _normalize_outputs(out: Any, *, label: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Convert run_model return value to dict of arrays, or None if out is None.
    """
    if out is None:
        return None

    if isinstance(out, dict):
        return {k: np.asarray(v) for k, v in out.items() if k in set(FIELDS)}

    if isinstance(out, tuple) and len(out) >= 5:
        eta, delta, Phi, U, V = out[:5]
        return {
            "eta": np.asarray(eta),
            "delta": np.asarray(delta),
            "Phi": np.asarray(Phi),
            "U": np.asarray(U),
            "V": np.asarray(V),
        }

    raise TypeError(f"{label}: unsupported run_model return type: {type(out)}")


def _pickle_load(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _find_latest_timestamp(custompath: Path) -> str:
    """
    In SWAMPE, pickles are named like:
      Phi-<timestamp>, eta-<timestamp>, U-<timestamp>, ...
    with timestamp typically an integer-like string.

    We pick the max numeric timestamp that has at least Phi-<ts>.
    """
    candidates = []
    for p in custompath.iterdir():
        m = re.match(r"^Phi-(\d+)$", p.name)
        if m:
            candidates.append(int(m.group(1)))
    if not candidates:
        raise FileNotFoundError(f"No Phi-<timestamp> files found in {custompath}")
    return str(max(candidates))


def _load_saved_fields(custompath: Path, *, label: str) -> Dict[str, np.ndarray]:
    """
    Load eta/delta/Phi/U/V from pickles in custompath, choosing the latest timestamp.
    """
    ts = _find_latest_timestamp(custompath)
    out: Dict[str, np.ndarray] = {}
    for k in FIELDS:
        fp = custompath / f"{k}-{ts}"
        if not fp.exists():
            raise FileNotFoundError(f"{label}: missing expected saved file {fp}")
        out[k] = np.asarray(_pickle_load(fp))
    return out


def _prepare_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_scalar_compare(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    plots_dir: Path,
    *,
    left_title: str = "JAX",
    right_title: str = "SWAMPE (NumPy)",
) -> None:
    left = np.asarray(left)
    right = np.asarray(right)

    vmin = float(np.min([left.min(), right.min()]))
    vmax = float(np.max([left.max(), right.max()]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(left, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title(f"{name} | {left_title}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(right, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title(f"{name} | {right_title}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out = plots_dir / f"{name}_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _save_quiver_geopot_compare(
    Phi_left: np.ndarray,
    U_left: np.ndarray,
    V_left: np.ndarray,
    Phi_right: np.ndarray,
    U_right: np.ndarray,
    V_right: np.ndarray,
    plots_dir: Path,
    *,
    stride: int = 4,
    left_title: str = "JAX",
    right_title: str = "SWAMPE (NumPy)",
) -> None:
    """
    Rough analogue to SWAMPE's quiver_geopot_plot: geopotential background + wind vectors.
    """
    Phi_left = np.asarray(Phi_left)
    U_left = np.asarray(U_left)
    V_left = np.asarray(V_left)

    Phi_right = np.asarray(Phi_right)
    U_right = np.asarray(U_right)
    V_right = np.asarray(V_right)

    # Same color scaling for both Phi panels
    vmin = float(np.min([Phi_left.min(), Phi_right.min()]))
    vmax = float(np.max([Phi_left.max(), Phi_right.max()]))

    def _ds(A: np.ndarray) -> np.ndarray:
        return A[::stride, ::stride]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(Phi_left, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].quiver(_ds(U_left), _ds(V_left), angles="xy", scale_units="xy", scale=None)
    axes[0].set_title(f"Phi + winds | {left_title}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(Phi_right, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].quiver(_ds(U_right), _ds(V_right), angles="xy", scale_units="xy", scale=None)
    axes[1].set_title(f"Phi + winds | {right_title}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out = plots_dir / "Phi_winds_quiver_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _save_mean_zonal_wind_compare(
    U_left: np.ndarray,
    U_right: np.ndarray,
    mus: np.ndarray,
    plots_dir: Path,
    *,
    left_title: str = "JAX",
    right_title: str = "SWAMPE (NumPy)",
) -> None:
    """
    Zonal-mean zonal wind: mean over longitude (axis=1) vs latitude (phi).
    mus is cos(latitude) in SWAMPE conventions.
    """
    U_left = np.asarray(U_left)
    U_right = np.asarray(U_right)
    mus = np.asarray(mus)

    # Convert mus=cos(phi) to phi in degrees for plotting
    phi = np.degrees(np.arccos(np.clip(mus, -1.0, 1.0))) - 90.0  # [-90, 90]

    ubar_left = U_left.mean(axis=1)
    ubar_right = U_right.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(phi, ubar_left, label=left_title)
    ax.plot(phi, ubar_right, label=right_title)
    ax.set_xlabel("latitude (deg)")
    ax.set_ylabel("zonal-mean U")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out = plots_dir / "mean_zonal_U_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


@dataclass(frozen=True)
class PackageRun:
    pkg: str
    run_model: Any
    origin: str
    module: Any


def _run_one(
    run: PackageRun,
    params: Dict[str, Any],
    outdir: Path,
    *,
    label: str,
) -> Dict[str, np.ndarray]:
    _prepare_clean_dir(outdir)

    # Ensure custompath is a string ending with a path separator for SWAMPE's concat logic.
    params_local = dict(params)
    params_local["custompath"] = str(outdir) + os.sep
    # Encourage deterministic "latest" file naming given SWAMPE's arg-order bug:
    # with timeunits='seconds' and dt passed in the wrong slot, timestamp becomes dt*t = real seconds.
    params_local["timeunits"] = "seconds"
    params_local["saveflag"] = True
    # Turn off package plotting (we generate our own compare figures).
    params_local["plotflag"] = False

    args, kwargs, dropped = _filter_kwargs_by_signature(run.run_model, params_local, label=label)

    if dropped:
        print(f"{label}: dropped unsupported kwargs: {list(dropped)}")

    t0 = time.time()
    out_raw = run.run_model(*args, **kwargs)
    t1 = time.time()
    print(f"{label} runtime: {t1 - t0:.3f} s ({run.origin})")

    out = _normalize_outputs(out_raw, label=label)
    if out is not None:
        return out

    # Fallback: load from saved pickles (SWAMPE behavior).
    return _load_saved_fields(outdir, label=label)


def _compare(a: np.ndarray, b: np.ndarray, name: str) -> None:
    diff = a - b
    denom = max(1.0, float(np.max(np.abs(a))))
    rel = float(np.max(np.abs(diff)) / denom)
    print(
        f"{name:5s}: max|a-b|={np.max(np.abs(diff)):.3e}  "
        f"rms(a-b)={np.sqrt(np.mean(diff**2)):.3e}  max_rel={rel:.3e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=1, choices=[1, 2])
    ap.add_argument("--tmax", type=int, default=50)
    ap.add_argument("--M", type=int, default=42)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--plots_dir", type=str, default="plots")
    ap.add_argument("--stride", type=int, default=4, help="stride for quiver downsampling")
    args = ap.parse_args()

    # Canonical parameter dictionary. Each run_model gets the subset it supports.
    params: Dict[str, Any] = {
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
        # Ensure saved pickles exist even if run_model returns None:
        "savefreq": 1,
        "plotfreq": 999999,  # irrelevant because plotflag=False, but safe
        "verbose": True,
    }

    numpy_pkg = os.environ.get("SWAMPE_NUMPY_PKG", "SWAMPE")
    jax_pkg = os.environ.get("MY_SWAMP_PKG", "my_swamp")

    # JAX port: prefer installed package; fallback to repo_root/src.
    try:
        jax_fn, jax_origin, jax_mod = _get_run_model(jax_pkg)
    except (ModuleNotFoundError, AttributeError):
        _ensure_src_on_path_for_dev()
        jax_fn, jax_origin, jax_mod = _get_run_model(jax_pkg)

    numpy_fn, numpy_origin, numpy_mod = _get_run_model(numpy_pkg)

    print(f"NumPy reference import: {numpy_pkg} ({numpy_origin})")
    print(f"JAX port import:       {jax_pkg} ({jax_origin})")

    repo = _repo_root()
    plots_dir = (repo / args.plots_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_root = repo / "tests" / "_compare_outputs"
    np_outdir = out_root / "numpy"
    jx_outdir = out_root / "jax"

    out_np = _run_one(PackageRun(numpy_pkg, numpy_fn, numpy_origin, numpy_mod), params, np_outdir, label="SWAMPE")
    out_jx = _run_one(PackageRun(jax_pkg, jax_fn, jax_origin, jax_mod), params, jx_outdir, label="JAX")

    # Compare numerically
    for k in FIELDS:
        if k not in out_np or k not in out_jx:
            raise RuntimeError(f"Missing key {k!r} in outputs. Got SWAMPE={list(out_np)}, JAX={list(out_jx)}")
        _compare(out_jx[k], out_np[k], k)

    # Produce side-by-side plots
    _save_scalar_compare("Phi", out_jx["Phi"], out_np["Phi"], plots_dir)
    _save_scalar_compare("eta", out_jx["eta"], out_np["eta"], plots_dir)
    _save_scalar_compare("delta", out_jx["delta"], out_np["delta"], plots_dir)
    _save_scalar_compare("U", out_jx["U"], out_np["U"], plots_dir)
    _save_scalar_compare("V", out_jx["V"], out_np["V"], plots_dir)

    _save_quiver_geopot_compare(
        out_jx["Phi"], out_jx["U"], out_jx["V"],
        out_np["Phi"], out_np["U"], out_np["V"],
        plots_dir,
        stride=max(1, int(args.stride)),
    )

    # Latitude grid for zonal-mean plot: use the JAX port's initial_conditions if available;
    # otherwise fall back to SWAMPE's.
    mus = None
    for pkg in (jax_pkg, numpy_pkg):
        try:
            ic = _import_module(pkg, "initial_conditions")
            _N, _I, _J, _dt0, _lambdas, mus0, _w = ic.spectral_params(args.M)
            mus = np.asarray(mus0)
            break
        except Exception:
            continue

    if mus is not None:
        _save_mean_zonal_wind_compare(out_jx["U"], out_np["U"], mus, plots_dir)

    print(f"Saved plots to: {plots_dir}")
    print(f"Saved SWAMPE pickles to: {np_outdir}")
    print(f"Saved JAX pickles to: {jx_outdir}")


if __name__ == "__main__":
    main()
