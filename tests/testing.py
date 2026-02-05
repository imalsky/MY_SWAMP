#!/usr/bin/env python3
"""
JAX vs NumPy SWAMPE comparison script, adapted for src/ + tests/ layout.

This script runs both implementations with identical parameters and compares
final fields (eta, delta, Phi, U, V). It is intentionally conservative: it
tries to run using each implementation's own "legacy" run_model API if present,
and falls back to a minimal inlined loop otherwise.

Repo layout:

repo/
  src/
    (A) flat modules OR
    (B) one or more packages with SWAMPE modules
  tests/
    testing.py   (this file)

Import resolution:
- Prefer SWAMPE_NUMPY_PKG / SWAMPE_JAX_PKG env vars if set.
- Otherwise scan src/ for packages, plus a few common names.
- Finally allow flat-module imports directly from src/.

Run:
  python tests/testing.py --test 1 --tmax 50 --M 42
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> Path:
    src = _repo_root() / "src"
    if not src.is_dir():
        raise RuntimeError(f"Expected src/ directory at: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _iter_src_packages(src: Path) -> List[str]:
    pkgs: List[str] = []
    for p in sorted(src.iterdir()):
        if p.is_dir() and (p / "__init__.py").is_file():
            pkgs.append(p.name)
    return pkgs


def _name_variants(name: str) -> List[str]:
    n0 = name.strip()
    n1 = n0.replace("-", "_").replace(" ", "_")
    variants = {n0, n1, n1.lower(), n1.upper()}
    return [v for v in variants if v and v[0].isalpha()]


def _import_from_base(base: str, module: str) -> ModuleType:
    return importlib.import_module(module if base == "" else f"{base}.{module}")


def _looks_like_jax_impl(mod: ModuleType) -> bool:
    d = getattr(mod, "__dict__", {})
    if "jnp" in d or "jax" in d:
        return True
    path = (getattr(mod, "__file__", "") or "").lower()
    return "jax" in path


def _looks_like_numpy_impl(mod: ModuleType) -> bool:
    d = getattr(mod, "__dict__", {})
    if "jnp" in d or "jax" in d:
        return False
    path = (getattr(mod, "__file__", "") or "").lower()
    return "jax" not in path


def _resolve_base(kind: str, required: Sequence[str]) -> str:
    src = _ensure_src_on_path()

    if kind not in {"jax", "numpy"}:
        raise ValueError(f"Unknown kind={kind!r}")

    env_key = "SWAMPE_JAX_PKG" if kind == "jax" else "SWAMPE_NUMPY_PKG"
    env = os.environ.get(env_key)

    candidates: List[str] = []
    if env:
        candidates.append(env)

    if kind == "jax":
        candidates += ["swampe_jax", "JAX", "swampe"]
    else:
        candidates += ["SWAMPE", "swampe_numpy", "swampe_ref", "swampe"]

    candidates += _name_variants(_repo_root().name)
    candidates += _iter_src_packages(src)
    candidates.append("")  # flat-module fallback

    last_err: Exception | None = None
    for base in candidates:
        try:
            st = _import_from_base(base, required[0])
            if kind == "jax" and not _looks_like_jax_impl(st):
                continue
            if kind == "numpy" and not _looks_like_numpy_impl(st):
                continue
            for mod in required[1:]:
                _import_from_base(base, mod)
            return base
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    msg = (
        f"Could not resolve {kind} SWAMPE base package.\n"
        f"  Tried {env_key}={env!r} and candidates={candidates!r}\n"
        f"  Expected modules: {', '.join(required)}\n"
        "If your implementation lives under src/<pkg>/..., set the env var accordingly.\n"
        "If it is flat modules under src/, ensure those .py files exist in src/."
    )
    raise ModuleNotFoundError(msg) from last_err


def _import_impl(base: str) -> Dict[str, ModuleType]:
    names = [
        "initial_conditions",
        "spectral_transform",
        "time_stepping",
        "filters",
        "forcing",
        "explicit_tdiff",
        "modEuler_tdiff",
        "model",
    ]
    mods: Dict[str, ModuleType] = {}
    for n in names:
        try:
            mods[n] = _import_from_base(base, n)
        except ModuleNotFoundError:
            # Some impls may not have model.py (if it's a partial port). Keep going.
            continue
    return mods


def _call_run_model_if_present(mods: Dict[str, ModuleType], params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Prefer an implementation's own run_model entrypoint when available.

    Returns dict of final fields in physical space:
      eta, delta, Phi, U, V   each shaped (J, I)
    """
    model = mods.get("model")
    if model is None or not hasattr(model, "run_model"):
        raise AttributeError("No model.run_model found")

    run_model = getattr(model, "run_model")

    # Common SWAMPE signature. We pass only known args; extra kwargs are ignored only if wrapper supports them.
    out = run_model(
        M=params["M"],
        dt=params["dt"],
        tmax=params["tmax"],
        starttime=params["starttime"],
        test=params["test"],
        forcflag=params["forcflag"],
        diffflag=params["diffflag"],
        expflag=params["expflag"],
        modalflag=params["modalflag"],
        alpha=params["alpha"],
        K6=params["K6"],
        K6Phi=params.get("K6Phi", params["K6"]),
        taurad=params["taurad"],
        taudrag=params["taudrag"],
        DPhieq=params["DPhieq"],
        a=params["a"],
        omega=params["omega"],
        g=params["g"],
        Phibar=params["Phibar"],
        a1=params["a1"],
        use_scipy_basis=params.get("use_scipy_basis", True),
        blowup_rms=params.get("blowup_rms", 8000.0),
    )

    if isinstance(out, dict):
        # assume already in expected format
        return {k: np.asarray(v) for k, v in out.items() if k in {"eta", "delta", "Phi", "U", "V"}}

    # If the implementation returns a tuple, try a common ordering.
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


def _compare(a: np.ndarray, b: np.ndarray, name: str) -> None:
    diff = a - b
    denom = np.maximum(1.0, np.max(np.abs(a)))
    rel = np.max(np.abs(diff)) / denom
    print(
        f"{name:5s}: max|a-b|={np.max(np.abs(diff)):.3e}  "
        f"rms(a-b)={np.sqrt(np.mean(diff**2)):.3e}  max_rel={rel:.3e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=1, choices=[1, 2])
    ap.add_argument("--tmax", type=int, default=50)
    ap.add_argument("--starttime", type=int, default=2)
    ap.add_argument("--M", type=int, default=42)
    ap.add_argument("--dt", type=float, default=600.0)
    args = ap.parse_args()

    params: Dict[str, Any] = {
        "M": args.M,
        "dt": args.dt,
        "tmax": args.tmax,
        "starttime": args.starttime,
        "test": args.test,
        "a": 6.37122e6,
        "omega": 7.292e-5,
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
        "use_scipy_basis": True,
        "blowup_rms": 8000.0,
    }

    numpy_base = _resolve_base("numpy", ["spectral_transform", "initial_conditions", "time_stepping"])
    jax_base = _resolve_base("jax", ["spectral_transform", "initial_conditions", "time_stepping"])

    numpy_mods = _import_impl(numpy_base)
    jax_mods = _import_impl(jax_base)

    print(f"Resolved numpy base: {numpy_base!r}")
    print(f"Resolved jax   base: {jax_base!r}")

    t0 = time.time()
    out_np = _call_run_model_if_present(numpy_mods, params)
    t1 = time.time()
    out_jx = _call_run_model_if_present(jax_mods, params)
    t2 = time.time()

    print(f"NumPy runtime: {t1 - t0:.3f} s")
    print(f"JAX   runtime: {t2 - t1:.3f} s")

    for k in ["eta", "delta", "Phi", "U", "V"]:
        if k not in out_np or k not in out_jx:
            raise RuntimeError(f"Missing key {k!r} in outputs. Got numpy={list(out_np)}, jax={list(out_jx)}")
        _compare(out_np[k], out_jx[k], k)


if __name__ == "__main__":
    main()
