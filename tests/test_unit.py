# -*- coding: utf-8 -*-
"""
test_unit.py

Unit tests for the MY_SWAMP (JAX) port.

Expected repo layout:

repo/
  src/
    my_swamp/
      __init__.py
      spectral_transform.py
      initial_conditions.py
      time_stepping.py
      ...
  testing/
    test_unit.py   (this file)

Run:
  python testing/test_unit.py
  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from jax import config as _jax_config

# Enable 64-bit *before* importing modules that import jax.numpy.
_jax_config.update("jax_enable_x64", True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path_for_dev() -> None:
    """
    Editable install (pip install -e .) is the preferred setup.

    This fallback makes `python testing/test_unit.py` work even if you haven't
    installed the package yet.
    """
    src = _repo_root() / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# Import MY_SWAMP modules (package import name is `my_swamp`)
try:
    import my_swamp.initial_conditions as ic  # noqa: E402
    import my_swamp.spectral_transform as st  # noqa: E402
    import my_swamp.time_stepping as tstep  # noqa: E402
except ModuleNotFoundError:
    _ensure_src_on_path_for_dev()
    import my_swamp.initial_conditions as ic  # type: ignore[no-redef]  # noqa: E402
    import my_swamp.spectral_transform as st  # type: ignore[no-redef]  # noqa: E402
    import my_swamp.time_stepping as tstep  # type: ignore[no-redef]  # noqa: E402


def _np(a) -> np.ndarray:
    return np.asarray(a)


def test_init() -> None:
    N, I, J, dt, lambdas, mus, w = ic.spectral_params(42)
    assert N == 42
    assert I == 128
    assert J == 64


def test_Pmn_Hmn() -> None:
    M = 42
    N, I, J, dt, lambdas, mus, w = ic.spectral_params(M)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    mus_np = _np(mus)
    Pmn_np = _np(Pmn)
    Hmn_np = _np(Hmn)

    Pmncheck = 0.25 * np.sqrt(15.0) * (1.0 - mus_np**2)
    Hmncheck = 0.5 * np.sqrt(6.0) * (1.0 - mus_np**2)

    assert np.allclose(Pmn_np[:, 2, 2], Pmncheck, atol=1e-12)
    assert np.allclose(Hmn_np[:, 0, 1], Hmncheck, atol=1e-12)


def test_spectral_transform() -> None:
    M = 106
    omega = 3.2e-5
    N, I, J, dt, lambdas, mus, w = ic.spectral_params(M)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    mus_np = _np(mus)
    f = (2.0 * omega * mus_np)[:, None] * np.ones((1, I), dtype=np.float64)

    fm = st.fwd_fft_trunc(f, I, M)
    fmn = st.fwd_leg(fm, J, M, N, Pmn, w)
    fmn_np = _np(fmn)

    fmncheck = np.zeros((M + 1, N + 1), dtype=fmn_np.dtype)
    fmncheck[0, 1] = omega / np.sqrt(0.375)

    assert np.allclose(fmn_np, fmncheck, atol=1e-12)


def test_spectral_transform_forward_inverse() -> None:
    M = 63
    omega = 7.2921159e-5
    a = 6.37122e6
    a1 = np.pi / 2.0
    test = 1

    N, I, J, dt, lambdas, mus, w = ic.spectral_params(M)
    Phibar = 3.0e3
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    SU0, sina, cosa, etaamp, Phiamp = ic.test1_init(a, omega, a1)
    etaic0, etaic1, deltaic0, deltaic1, Phiic0, Phiic1 = ic.state_var_init(
        I, J, mus, lambdas, test, etaamp, a, sina, cosa, Phibar, Phiamp
    )
    Uic, Vic = ic.velocity_init(I, J, SU0, cosa, sina, mus, lambdas, test)

    Uicm = st.fwd_fft_trunc(Uic, I, M)
    Uicmn = st.fwd_leg(Uicm, J, M, N, Pmn, w)
    Uicmnew = st.invrs_leg(Uicmn, I, J, M, N, Pmn)
    Uicnew = st.invrs_fft(Uicmnew, I)

    assert np.allclose(_np(Uic), _np(Uicnew), atol=1e-11)


def test_wind_transform() -> None:
    M = 106
    omega = 7.2921159e-5
    a = 6.37122e6
    a1 = np.pi / 4.0
    test = 1
    dt = 30.0
    Phibar = 1.0e3

    N, I, J, _, lambdas, mus, w = ic.spectral_params(M)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    fmn = np.zeros((M + 1, N + 1), dtype=np.float64)
    fmn[0, 1] = omega / np.sqrt(0.375)

    tstepcoeffmn = tstep.tstepcoeffmn(M, N, a)
    tstepcoeff = tstep.tstepcoeff(J, M, dt, mus, a)
    mJarray = tstep.mJarray(J, M)
    marray = tstep.marray(M, N)

    SU0, sina, cosa, etaamp, Phiamp = ic.test1_init(a, omega, a1)
    etaic0, etaic1, deltaic0, deltaic1, Phiic0, Phiic1 = ic.state_var_init(
        I, J, mus, lambdas, test, etaamp, a, sina, cosa, Phibar, Phiamp
    )
    U, V = ic.velocity_init(I, J, SU0, cosa, sina, mus, lambdas, test)

    Um = st.fwd_fft_trunc(U, I, M)
    Vm = st.fwd_fft_trunc(V, I, M)

    eta, delta, etamn, deltamn = st.diagnostic_eta_delta(
        Um, Vm, fmn, I, J, M, N, Pmn, Hmn, w, tstepcoeff, mJarray, dt
    )

    Unew, Vnew = st.invrsUV(deltamn, etamn, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray)

    assert np.allclose(_np(U), _np(Unew), atol=1e-11)
    assert np.allclose(_np(V), _np(Vnew), atol=1e-11)


def test_vorticity_divergence_transform() -> None:
    M = 63
    omega = 7.2921159e-5
    a = 6.37122e6
    a1 = np.pi / 4.0
    test = 1
    dt = 30.0
    Phibar = 4.0e4

    N, I, J, _, lambdas, mus, w = ic.spectral_params(M)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    fmn = np.zeros((M + 1, N + 1), dtype=np.float64)
    fmn[0, 1] = omega / np.sqrt(0.375)

    tstepcoeffmn = tstep.tstepcoeffmn(M, N, a)
    tstepcoeff = tstep.tstepcoeff(J, M, dt, mus, a)
    mJarray = tstep.mJarray(J, M)
    marray = tstep.marray(M, N)

    SU0, sina, cosa, etaamp, Phiamp = ic.test1_init(a, omega, a1)
    etaic0, etaic1, deltaic0, deltaic1, Phiic0, Phiic1 = ic.state_var_init(
        I, J, mus, lambdas, test, etaamp, a, sina, cosa, Phibar, Phiamp
    )
    U, V = ic.velocity_init(I, J, SU0, cosa, sina, mus, lambdas, test)

    deltam = st.fwd_fft_trunc(deltaic0, I, M)
    deltamn = st.fwd_leg(deltam, J, M, N, Pmn, w)

    etam = st.fwd_fft_trunc(etaic0, I, M)
    etamn = st.fwd_leg(etam, J, M, N, Pmn, w)

    U, V = st.invrsUV(deltamn, etamn, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray)

    Um = st.fwd_fft_trunc(U, I, M)
    Vm = st.fwd_fft_trunc(V, I, M)

    etanew, deltanew, etamnnew, deltamnnew = st.diagnostic_eta_delta(
        Um, Vm, fmn, I, J, M, N, Pmn, Hmn, w, tstepcoeff, mJarray, dt
    )

    assert np.allclose(_np(etaic0), _np(etanew), atol=1e-11)
    assert np.allclose(_np(deltaic0), _np(deltanew), atol=1e-11)


def _run_as_script() -> int:
    tests = [name for name in globals() if name.startswith("test_")]
    failures = 0
    for name in sorted(tests):
        fn = globals()[name]
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
