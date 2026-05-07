# ruff: noqa: E741
"""Validation tests for `run_model_scan` boundary input checks.

Each test verifies that bad input raises the expected exception type and that
the failure happens at the run_model_scan entry, not partway through the scan.
This locks in the validation logic so future refactors can't accidentally drop
a guard.
"""
from __future__ import annotations

import numpy as np
import pytest


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


def test_tmax_too_small() -> None:
    """tmax < 2 must raise ValueError (SWAMPE uses a 2-level initialization)."""
    from my_swamp.model import run_model_scan

    with pytest.raises(ValueError, match=r"tmax"):
        run_model_scan(**_BASE_KWARGS, tmax=1)


def test_dt_non_positive() -> None:
    """dt <= 0 with a concrete Python scalar must raise ValueError."""
    from my_swamp.model import run_model_scan

    kwargs = dict(_BASE_KWARGS)
    kwargs["dt"] = 0.0
    with pytest.raises(ValueError, match=r"dt"):
        run_model_scan(**kwargs, tmax=4)


def test_starttime_after_tmax() -> None:
    """starttime > tmax must raise ValueError."""
    from my_swamp.model import run_model_scan

    with pytest.raises(ValueError, match=r"starttime"):
        run_model_scan(**_BASE_KWARGS, tmax=4, starttime=10)


def test_unsupported_test_selector_raises_via_main() -> None:
    """`main_function.main(test=99)` must raise NotImplementedError."""
    from my_swamp.main_function import main

    with pytest.raises(NotImplementedError, match=r"test=99"):
        main(
            M=42,
            dt=600.0,
            tmax=4,
            Phibar=3.0e5,
            omega=7.292e-5,
            a=6.37122e6,
            test=99,
            saveflag=False,
            plotflag=False,
            verbose=False,
        )


def test_unsupported_M_raises() -> None:
    """`spectral_params(M=50)` must raise ValueError (only 42, 63, 106 supported)."""
    from my_swamp.initial_conditions import spectral_params

    with pytest.raises(ValueError, match=r"M"):
        spectral_params(50)


def test_partial_initial_conditions_eta_only_raises() -> None:
    """Providing only eta0_init (without delta0/Phi0) must raise ValueError."""
    from my_swamp.initial_conditions import spectral_params
    from my_swamp.model import run_model_scan

    M = 42
    _N, I, J, _dt, _lambdas, _mus, _w = spectral_params(M)
    eta0 = np.zeros((J, I), dtype=np.float64)

    with pytest.raises(ValueError, match=r"eta0_init.*delta0_init.*Phi0_init"):
        run_model_scan(**_BASE_KWARGS, tmax=4, eta0_init=eta0)


def test_partial_velocity_initial_conditions_raises() -> None:
    """Providing U0_init without V0_init (or vice versa) must raise ValueError."""
    from my_swamp.initial_conditions import spectral_params
    from my_swamp.model import run_model_scan

    M = 42
    _N, I, J, _dt, _lambdas, _mus, _w = spectral_params(M)
    eta0 = np.zeros((J, I), dtype=np.float64)
    delta0 = np.zeros((J, I), dtype=np.float64)
    Phi0 = np.zeros((J, I), dtype=np.float64)
    U0 = np.zeros((J, I), dtype=np.float64)

    with pytest.raises(ValueError, match=r"U0_init.*V0_init"):
        run_model_scan(
            **_BASE_KWARGS,
            tmax=4,
            eta0_init=eta0,
            delta0_init=delta0,
            Phi0_init=Phi0,
            U0_init=U0,  # V0_init missing
        )


def test_wrong_shape_initial_condition_raises() -> None:
    """An initial-condition field with the wrong shape must raise ValueError."""
    from my_swamp.initial_conditions import spectral_params
    from my_swamp.model import run_model_scan

    M = 42
    _N, I, J, _dt, _lambdas, _mus, _w = spectral_params(M)
    eta0 = np.zeros((J, I), dtype=np.float64)
    delta0 = np.zeros((J, I), dtype=np.float64)
    Phi0_bad = np.zeros((J + 1, I), dtype=np.float64)  # wrong shape

    with pytest.raises(ValueError, match=r"Phi0_init"):
        run_model_scan(
            **_BASE_KWARGS,
            tmax=4,
            eta0_init=eta0,
            delta0_init=delta0,
            Phi0_init=Phi0_bad,
        )


def test_contflag_without_contTime_raises() -> None:
    """contflag=True without contTime must raise ValueError."""
    from my_swamp.model import run_model_scan

    with pytest.raises(ValueError, match=r"contTime"):
        run_model_scan(**_BASE_KWARGS, tmax=4, contflag=True)


def test_contTime_non_numeric_raises() -> None:
    """contflag=True with a non-numeric contTime must raise ValueError."""
    from my_swamp.model import run_model_scan

    with pytest.raises(ValueError, match=r"numeric"):
        run_model_scan(**_BASE_KWARGS, tmax=4, contflag=True, contTime="not-a-number")
