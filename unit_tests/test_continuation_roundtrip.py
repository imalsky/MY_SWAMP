# ruff: noqa: E741
"""Continuation save/load round-trip test.

Important: continuation deliberately saves only a single time level of
physical state (eta, delta, Phi) and re-derives winds + spectral coefficients
on resume. The leapfrog "two-level" memory is therefore lost across a
save/load boundary — this is faithful to reference SWAMPE behavior. As a
result, a save-resume run cannot bitwise reproduce a direct 2N-step
integration; the leapfrog effectively reinitializes at the resume point.

What this test instead asserts is the right invariant: a continuation resume
must reproduce *what you would get if you started a fresh run from the same
single-level physical state*. That is, the contflag-driven path through
`run_model_scan` must agree with passing the same physical fields explicitly
via `eta0_init`/`delta0_init`/`Phi0_init`.

This locks in the `continuation.{save,read}_pickle` round-trip and the
contTime / save-filename plumbing without making a false claim about leapfrog
state continuity that the save format cannot support.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest


# Use a tiny configuration so the test is fast; "seconds" timeunits keeps the
# timestamp arithmetic exact across save/load.
_BASE_KWARGS = dict(
    M=42,
    dt=600.0,
    Phibar=3.0e5,
    omega=7.292e-5,
    a=6.37122e6,
    test=None,
    g=9.8,
    forcflag=True,
    taurad=86400.0,
    taudrag=86400.0,
    DPhieq=4.0e6,
    diffflag=True,
    modalflag=True,
    alpha=0.01,
    expflag=False,
    K6=1.24e33,
    diagnostics=False,
    jit_scan=False,
)

_N = 4  # half-run length, in steps after the 2-level init


def _x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def _terminal_fields(sim_payload):
    last = sim_payload["last_state"]
    return {
        "eta": np.asarray(last.eta_curr),
        "delta": np.asarray(last.delta_curr),
        "Phi": np.asarray(last.Phi_curr),
        "U": np.asarray(last.U_curr),
        "V": np.asarray(last.V_curr),
    }


@pytest.mark.parity
def test_continuation_resume_matches_explicit_ic_restart() -> None:
    """contflag-resume must reproduce a fresh run started from the same physical state.

    This pins down the contTime / save-filename / `_diagnose_winds` plumbing
    without making a false claim about leapfrog continuity (see module docstring).
    """
    if not _x64_enabled():
        pytest.skip("Continuation parity requires x64 for tight tolerance.")

    from my_swamp.continuation import compute_timestamp, save_data
    from my_swamp.model import run_model_scan, run_model_scan_final

    with tempfile.TemporaryDirectory(prefix="swampe_cont_roundtrip_", dir="/tmp") as tmp:
        tmp_path = str(Path(tmp).resolve()) + "/"

        # 1) Run N steps with history so we have a saved snapshot to resume from.
        first = run_model_scan(**_BASE_KWARGS, tmax=2 + _N)
        last1 = first["last_state"]
        eta_save = np.asarray(last1.eta_curr)
        delta_save = np.asarray(last1.delta_curr)
        Phi_save = np.asarray(last1.Phi_curr)

        last_step = int(np.asarray(first["t_seq"])[-1])
        timestamp = compute_timestamp("seconds", last_step, _BASE_KWARGS["dt"])
        spinupdata = np.zeros((2 + _N, 2), dtype=float)
        geopotdata = np.zeros((2 + _N, 2), dtype=float)
        save_data(
            timestamp,
            eta_save,
            delta_save,
            Phi_save,
            np.asarray(last1.U_curr),
            np.asarray(last1.V_curr),
            spinupdata,
            geopotdata,
            custompath=tmp_path,
        )

        # 2a) Resume via contflag from the saved snapshot.
        resumed = run_model_scan_final(
            **_BASE_KWARGS,
            tmax=2 + 2 * _N,
            contflag=True,
            custompath=tmp_path,
            contTime=timestamp,
            timeunits="seconds",
        )
        resumed_fields = _terminal_fields(resumed)

        # 2b) Reference: build the same single-level state from explicit ICs
        # (this is what continuation conceptually does internally) and run N
        # more steps. The starttime is chosen so the same number of scan
        # iterations runs.
        starttime_resume = last_step
        explicit = run_model_scan_final(
            **_BASE_KWARGS,
            tmax=2 + 2 * _N,
            eta0_init=eta_save,
            delta0_init=delta_save,
            Phi0_init=Phi_save,
            starttime=starttime_resume,
        )
        explicit_fields = _terminal_fields(explicit)

    # contflag-resume and explicit-IC restart must match to tight tolerance.
    atol = {
        "eta": 1.0e-10,
        "delta": 1.0e-10,
        "Phi": 5.0e-8,
        "U": 1.0e-9,
        "V": 1.0e-9,
    }
    for field in ("eta", "delta", "Phi", "U", "V"):
        np.testing.assert_allclose(
            resumed_fields[field],
            explicit_fields[field],
            rtol=0.0,
            atol=atol[field],
            err_msg=f"contflag resume diverges from explicit-IC restart for field {field}",
        )
