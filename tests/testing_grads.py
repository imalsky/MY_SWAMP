#!/usr/bin/env python3
from __future__ import annotations

"""
Gradient / sensitivity smoke test for the SWAMPE JAX port, adapted for src/ + tests/ layout.

This script runs a short forced-mode simulation and differentiates a simple scalar
objective with respect to (taurad, taudrag).

Run:
  python tests/testing_grads.py
  (or from tests/: python testing_grads.py)

Import resolution:
- Prefer SWAMPE_JAX_PKG if set (base package name under src/).
- Otherwise auto-detect by scanning src/ for packages, common names, and then
  falling back to flat-module imports from src/.
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import config as _jax_config

# Enable 64-bit for numerical stability / parity.
_jax_config.update("jax_enable_x64", True)


# =============================================================================
# USER CONFIG (edit these)
# =============================================================================

TMAX: int = 100
STARTTIME: int = 2
M: int = 42
DT: float = 600.0

A: float = 6.37122e6
OMEGA: float = 7.292e-5
G: float = 9.80616
PHIBAR: float = 3.0e5

TEST: Optional[int] = None          # None => forced mode; 1/2 => test cases
DPHIEQ: float = 4.0e6
A1: float = 0.05
TAURAD0: float = 86400.0
TAUDRAG0: float = 86400.0

K6: float = 1.24e33
K6PHI: float = 1.24e33

USE_SCIPY_BASIS: bool = True
FORCFLAG: bool = True
DIFFFLAG: bool = True
EXPFLAG: bool = False
MODALFLAG: bool = False
ALPHA: float = 0.01
BLOWUP_RMS: float = 8000.0

OUTDIR: Path = Path("outputs_tau_sens")
DO_FD_CHECK: bool = False
FD_FRAC: float = 1e-4
DO_PLOT: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> Path:
    src = _repo_root() / "src"
    if not src.is_dir():
        raise RuntimeError(f"Expected src/ directory at: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _iter_src_packages(src: Path) -> list[str]:
    pkgs: list[str] = []
    for p in sorted(src.iterdir()):
        if p.is_dir() and (p / "__init__.py").is_file():
            pkgs.append(p.name)
    return pkgs


def _name_variants(name: str) -> list[str]:
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


def _resolve_jax_base() -> str:
    src = _ensure_src_on_path()
    env = os.environ.get("SWAMPE_JAX_PKG")

    candidates: list[str] = []
    if env:
        candidates.append(env)

    candidates += ["swampe_jax", "JAX", "swampe"]
    candidates += _name_variants(_repo_root().name)
    candidates += _iter_src_packages(src)
    candidates.append("")  # flat-module fallback

    last_err: Exception | None = None
    for base in candidates:
        try:
            st = _import_from_base(base, "spectral_transform")
            if not _looks_like_jax_impl(st):
                continue
            # model must exist for this script
            _import_from_base(base, "model")
            return base
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    raise ModuleNotFoundError(
        "Could not import JAX SWAMPE port (need at least spectral_transform.py and model.py).\n"
        f"Tried SWAMPE_JAX_PKG={env!r} and candidates={candidates!r}"
    ) from last_err


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _maybe_plot(outdir: Path, name: str, arr: np.ndarray) -> None:
    if not DO_PLOT:
        return
    import matplotlib.pyplot as plt  # optional

    plt.figure()
    plt.imshow(arr, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(name)
    plt.tight_layout()
    plt.savefig(outdir / f"{name}.png", dpi=150)
    plt.close()


def main() -> None:
    base = _resolve_jax_base()
    model = _import_from_base(base, "model")

    # Prefer stable internal entrypoints if present (matches your earlier script).
    if not all(hasattr(model, k) for k in ("RunFlags", "build_static", "_init_state", "_step_once")):
        raise RuntimeError(
            "JAX model.py is missing one of: RunFlags, build_static, _init_state, _step_once. "
            "If your model API differs, either update this script or expose those helpers."
        )

    RunFlags = getattr(model, "RunFlags")
    build_static = getattr(model, "build_static")
    _init_state = getattr(model, "_init_state")
    _step_once = getattr(model, "_step_once")

    outdir = (_repo_root() / OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    flags = RunFlags(
        forcflag=FORCFLAG,
        diffflag=DIFFFLAG,
        expflag=EXPFLAG,
        modalflag=MODALFLAG,
        alpha=ALPHA,
        blowup_rms=BLOWUP_RMS,
        use_scipy_basis=USE_SCIPY_BASIS,
    )

    static = build_static(M=M, dt=DT, a=A, omega=OMEGA, g=G, Phibar=PHIBAR, a1=A1, K6=K6, K6Phi=K6PHI, flags=flags)

    # forced mode => test=None, but your model likely expects test as int or None.
    state0 = _init_state(static=static, starttime=STARTTIME, test=TEST, taurad=TAURAD0, taudrag=TAUDRAG0, DPhieq=DPHIEQ)

    def objective(taurad_taudrag: jnp.ndarray) -> jnp.ndarray:
        taurad, taudrag = taurad_taudrag[0], taurad_taudrag[1]
        state = _init_state(static=static, starttime=STARTTIME, test=TEST, taurad=taurad, taudrag=taudrag, DPhieq=DPHIEQ)
        # integrate
        for _ in range(TMAX):
            state, diag = _step_once(static=static, state=state, taurad=taurad, taudrag=taudrag, DPhieq=DPHIEQ)
        Phi = state.Phi  # (J, I)
        # scalar loss: RMS Phi
        return jnp.sqrt(jnp.mean(Phi**2))

    x0 = jnp.array([TAURAD0, TAUDRAG0], dtype=jnp.float64)

    t0 = time.time()
    val = objective(x0)
    g = jax.grad(objective)(x0)
    t1 = time.time()

    val_np = float(np.asarray(val))
    g_np = np.asarray(g, dtype=np.float64)

    print(f"Objective(Phi_rms) = {val_np:.6e}")
    print(f"grad w.r.t [taurad, taudrag] = [{g_np[0]:.6e}, {g_np[1]:.6e}]")
    print(f"runtime (post-compile) = {t1 - t0:.3f} s")

    _save_json(outdir / "grad_summary.json", {"objective": val_np, "grad": g_np.tolist()})

    _maybe_plot(outdir, "Phi_final", np.asarray(state0.Phi))

    if DO_FD_CHECK:
        eps = FD_FRAC
        fd = []
        for i in range(2):
            dx = np.zeros(2, dtype=np.float64)
            dx[i] = eps * float(x0[i])
            v_p = float(np.asarray(objective(jnp.array(x0 + dx))))
            v_m = float(np.asarray(objective(jnp.array(x0 - dx))))
            fd.append((v_p - v_m) / (2.0 * dx[i]))
        print(f"FD grad approx = [{fd[0]:.6e}, {fd[1]:.6e}]")

    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
