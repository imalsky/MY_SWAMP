# ruff: noqa: E741
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


CASES = {
    "reference_case_unforced_test1": dict(
        kwargs=dict(
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
    ),
    "reference_case_forced_default": dict(
        kwargs=dict(
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
    ),
}

ATOL = {
    "eta": 1.0e-10,
    "delta": 1.0e-10,
    "Phi": 5.0e-8,
    "U": 1.0e-9,
    "V": 1.0e-9,
}


def _phase_curve_from_phi(phi: np.ndarray, phase_angles: np.ndarray) -> np.ndarray:
    J, I = phi.shape
    lambdas = np.linspace(-np.pi, np.pi, I, endpoint=False)[None, :]
    mus = np.polynomial.legendre.leggauss(J)[0][:, None]
    area_weight = np.sqrt(np.clip(1.0 - mus * mus, 0.0, None))

    flux = np.empty_like(phase_angles, dtype=np.float64)
    for k, phase in enumerate(phase_angles):
        visible = np.maximum(np.cos(lambdas - phase), 0.0)
        flux[k] = float(np.sum(phi * visible * area_weight))

    norm = np.mean(np.abs(flux))
    if norm > 0:
        flux = flux / norm
    return flux


def _assert_x64_enabled() -> None:
    import jax

    if not bool(jax.config.read("jax_enable_x64")):
        raise AssertionError(
            "Parity regression tests require float64 mode. "
            "Set SWAMPE_JAX_ENABLE_X64=1 (and/or JAX_ENABLE_X64=1)."
        )


@pytest.mark.parity
@pytest.mark.parametrize("case_name", list(CASES.keys()))
def test_reference_parity_terminal_and_snapshots(case_name: str) -> None:
    _assert_x64_enabled()
    from my_swamp.model import run_model_scan

    ref = np.load(FIXTURE_DIR / f"{case_name}.npz")
    kwargs = dict(CASES[case_name]["kwargs"])

    out = run_model_scan(**kwargs, diagnostics=True, jit_scan=False)
    t_seq = np.asarray(out["t_seq"])
    hist = out["outs"]
    last = out["last_state"]

    got_terminal = {
        "eta": np.asarray(last.eta_curr),
        "delta": np.asarray(last.delta_curr),
        "Phi": np.asarray(last.Phi_curr),
        "U": np.asarray(last.U_curr),
        "V": np.asarray(last.V_curr),
    }
    for field, arr in got_terminal.items():
        np.testing.assert_allclose(arr, np.asarray(ref[f"final_{field}"]), rtol=0.0, atol=ATOL[field])

    snapshot_steps = np.asarray(ref["snapshot_steps"], dtype=np.int32)
    for i, step in enumerate(snapshot_steps):
        idx = int(np.where(t_seq == step)[0][0])
        for field in ("eta", "delta", "Phi", "U", "V"):
            np.testing.assert_allclose(
                np.asarray(hist[field])[idx],
                np.asarray(ref[f"snapshot_{field}"])[i],
                rtol=0.0,
                atol=ATOL[field],
            )


@pytest.mark.parity
@pytest.mark.parametrize("case_name", list(CASES.keys()))
def test_reference_retrieval_projection_parity(case_name: str) -> None:
    _assert_x64_enabled()
    from my_swamp.model import run_model_scan_final

    ref = np.load(FIXTURE_DIR / f"{case_name}.npz")
    kwargs = dict(CASES[case_name]["kwargs"])

    out = run_model_scan_final(**kwargs, diagnostics=False, jit_scan=False)
    phi = np.asarray(out["last_state"].Phi_curr, dtype=np.float64)

    phase_angles = np.asarray(ref["phase_angles"], dtype=np.float64)
    got_curve = _phase_curve_from_phi(phi, phase_angles)
    np.testing.assert_allclose(got_curve, np.asarray(ref["phase_curve"]), rtol=0.0, atol=1.0e-10)
