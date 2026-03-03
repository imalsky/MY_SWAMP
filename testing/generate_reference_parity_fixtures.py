#!/usr/bin/env python3
# ruff: noqa: E741
"""Generate trusted SWAMPE parity fixtures for my_swamp regression tests."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import scipy.special as sp


ROOT = Path(__file__).resolve().parents[1]
REF_ROOT = ROOT.parents[0] / "SWAMPE"
OUT_DIR = ROOT / "tests" / "fixtures"


def _ensure_lpmn_compat() -> None:
    if hasattr(sp, "lpmn"):
        return

    def _lpmn_compat(M: int, N: int, x: float):
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


def _phase_curve_from_phi(phi: np.ndarray, n_phase: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    J, I = phi.shape
    lambdas = np.linspace(-np.pi, np.pi, I, endpoint=False)[None, :]
    mus = np.polynomial.legendre.leggauss(J)[0][:, None]
    area_weight = np.sqrt(np.clip(1.0 - mus * mus, 0.0, None))

    phases = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    flux = np.empty((n_phase,), dtype=np.float64)
    for k, phase in enumerate(phases):
        visible = np.maximum(np.cos(lambdas - phase), 0.0)
        flux[k] = float(np.sum(phi * visible * area_weight))

    norm = np.mean(np.abs(flux))
    if norm > 0:
        flux = flux / norm
    return phases, flux


def _read_fields(ref_cont, path: str, step: int, dt: float) -> Dict[str, np.ndarray]:
    ts = str(int(dt * step))
    return {
        "eta": np.asarray(ref_cont.read_pickle(f"eta-{ts}", custompath=path), dtype=np.float64),
        "delta": np.asarray(ref_cont.read_pickle(f"delta-{ts}", custompath=path), dtype=np.float64),
        "Phi": np.asarray(ref_cont.read_pickle(f"Phi-{ts}", custompath=path), dtype=np.float64),
        "U": np.asarray(ref_cont.read_pickle(f"U-{ts}", custompath=path), dtype=np.float64),
        "V": np.asarray(ref_cont.read_pickle(f"V-{ts}", custompath=path), dtype=np.float64),
    }


def _save_case(case_name: str, kwargs: Dict[str, float], snapshot_steps: Iterable[int]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path("/tmp/mpl").mkdir(parents=True, exist_ok=True)

    _ensure_lpmn_compat()
    sys.path.insert(0, str(REF_ROOT))

    import SWAMPE.continuation as ref_cont
    import SWAMPE.model as ref_model

    snapshot_steps = tuple(int(s) for s in snapshot_steps)
    tmax = int(kwargs["tmax"])
    final_step = tmax - 1

    with tempfile.TemporaryDirectory(prefix=f"swampe_ref_{case_name}_", dir="/tmp") as tmp:
        custompath = f"{tmp}/"
        ref_model.run_model(
            plotflag=False,
            saveflag=True,
            savefreq=1,
            timeunits="seconds",
            custompath=custompath,
            verbose=False,
            **kwargs,
        )

        final_fields = _read_fields(ref_cont, custompath, final_step, float(kwargs["dt"]))
        snap_fields = [_read_fields(ref_cont, custompath, s, float(kwargs["dt"])) for s in snapshot_steps]

        phases, flux = _phase_curve_from_phi(final_fields["Phi"], n_phase=64)

        out = {
            "snapshot_steps": np.asarray(snapshot_steps, dtype=np.int32),
            "final_eta": final_fields["eta"],
            "final_delta": final_fields["delta"],
            "final_Phi": final_fields["Phi"],
            "final_U": final_fields["U"],
            "final_V": final_fields["V"],
            "snapshot_eta": np.stack([s["eta"] for s in snap_fields], axis=0),
            "snapshot_delta": np.stack([s["delta"] for s in snap_fields], axis=0),
            "snapshot_Phi": np.stack([s["Phi"] for s in snap_fields], axis=0),
            "snapshot_U": np.stack([s["U"] for s in snap_fields], axis=0),
            "snapshot_V": np.stack([s["V"] for s in snap_fields], axis=0),
            "spinup": np.asarray(ref_cont.read_pickle("spinup-winds", custompath=custompath), dtype=np.float64),
            "geopot": np.asarray(ref_cont.read_pickle("spinup-geopot", custompath=custompath), dtype=np.float64),
            "phase_angles": phases,
            "phase_curve": flux,
        }

        for key, value in kwargs.items():
            if value is None:
                out[f"param_{key}"] = np.asarray(np.nan, dtype=np.float64)
                out[f"param_{key}_is_none"] = np.asarray(True)
            else:
                out[f"param_{key}"] = np.asarray(value)
                out[f"param_{key}_is_none"] = np.asarray(False)

        out_path = OUT_DIR / f"{case_name}.npz"
        np.savez_compressed(out_path, **out)
        print(f"Wrote fixture: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _save_case(
        "reference_case_unforced_test1",
        dict(
            M=42,
            dt=30.0,
            tmax=8,
            Phibar=3.0e3,
            omega=7.2921159e-5,
            a=6.37122e6,
            test=1,
            g=9.8,
            forcflag=False,
            taurad=86400.0,
            taudrag=86400.0,
            DPhieq=4.0e6,
            diffflag=False,
            modalflag=True,
            alpha=0.01,
            expflag=False,
            K6=1.24e33,
        ),
        snapshot_steps=(2, 4, 7),
    )

    _save_case(
        "reference_case_forced_default",
        dict(
            M=42,
            dt=1200.0,
            tmax=6,
            Phibar=3.0e5,
            omega=3.2e-5,
            a=8.2e7,
            test=None,
            g=9.8,
            forcflag=True,
            taurad=10.0 * 3600.0,
            taudrag=6.0 * 3600.0,
            DPhieq=1.0e6,
            diffflag=True,
            modalflag=True,
            alpha=0.01,
            expflag=False,
            K6=1.24e33,
        ),
        snapshot_steps=(2, 3, 5),
    )


if __name__ == "__main__":
    main()
