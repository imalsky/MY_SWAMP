#!/usr/bin/env python3
from __future__ import annotations

"""
testing_grads.py

Gradient / sensitivity smoke test for the MY_SWAMP (JAX) port.

Expected repo layout:

repo/
  src/
    my_swamp/
      model.py
      ...
  testing/
    testing_grads.py  (this file)

Run:
  python testing/testing_grads.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax import config as _jax_config

# Enable 64-bit for numerical stability / parity.
_jax_config.update("jax_enable_x64", True)


# =============================================================================
# USER CONFIG
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

OUTDIR: Path = Path("outputs")
DO_FD_CHECK: bool = False
FD_FRAC: float = 1e-4
DO_PLOT: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path_for_dev() -> None:
    src = _repo_root() / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    try:
        from my_swamp import model  # noqa: E402
    except ModuleNotFoundError:
        _ensure_src_on_path_for_dev()
        from my_swamp import model  # type: ignore[no-redef]  # noqa: E402

    required = ("RunFlags", "build_static", "_init_state", "_step_once")
    missing = [k for k in required if not hasattr(model, k)]
    if missing:
        raise RuntimeError(
            "my_swamp.model is missing required helpers for this script: "
            + ", ".join(missing)
            + ". Either expose these names or update testing/testing_grads.py."
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

    static = build_static(
        M=M,
        dt=DT,
        a=A,
        omega=OMEGA,
        g=G,
        Phibar=PHIBAR,
        a1=A1,
        K6=K6,
        K6Phi=K6PHI,
        flags=flags,
    )

    def objective(x: jnp.ndarray) -> jnp.ndarray:
        taurad, taudrag = x[0], x[1]
        state = _init_state(static=static, starttime=STARTTIME, test=TEST, taurad=taurad, taudrag=taudrag, DPhieq=DPHIEQ)
        for _ in range(TMAX):
            state, _diag = _step_once(static=static, state=state, taurad=taurad, taudrag=taudrag, DPhieq=DPHIEQ)
        Phi = state.Phi  # (J, I)
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

    # optional plot of baseline Phi (not a result of the optimization)
    baseline_state = _init_state(static=static, starttime=STARTTIME, test=TEST, taurad=TAURAD0, taudrag=TAUDRAG0, DPhieq=DPHIEQ)
    _maybe_plot(outdir, "Phi_baseline", np.asarray(baseline_state.Phi))

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
