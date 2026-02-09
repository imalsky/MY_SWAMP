# -*- coding: utf-8 -*-
"""
Unit tests for the spectral transform stack.

These are adapted from the reference SWAMPE `test_unit.py` to validate that the
JAX port preserves the spectral/physical transform identities.

Run as a script:
    python -m my_swamp.test_unit

Or under pytest (recommended):
    pytest -q
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax
from my_swamp.dtypes import float_dtype

# These tests are *numerically sensitive* because they exercise chained FFT and
# Legendre transforms.
#
# In float64 mode, we expect near machine-precision agreement.
# In float32 mode, round-off accumulates and we must use a looser tolerance.
_X64 = bool(jax.config.read("jax_enable_x64"))

if _X64:
    _ATOL = 1e-10
    _RTOL = 1e-10
else:
    # Empirically, the forward+inverse spectral stack is accurate to O(1e-5)
    # absolute / O(1e-4) relative in float32 on CPU.
    _ATOL = 1e-4
    _RTOL = 2e-4

from my_swamp import spectral_transform as st
from my_swamp import initial_conditions as ic
from my_swamp import time_stepping as tstep


def test_init() -> None:
    N, I, J, _dt, _lambdas, _mus, _w = ic.spectral_params(42)
    assert N == 42, "N should be 42"
    assert I == 128, "I should be 128"
    assert J == 64, "J should be 64"


def test_Pmn_Hmn() -> None:
    M = 42
    N, I, J, _dt, _lambdas, mus, _w = ic.spectral_params(M)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    mus_np = np.asarray(mus)
    Pmncheck = 0.25 * np.sqrt(15.0) * (1.0 - mus_np**2)
    Hmncheck = 0.5 * np.sqrt(6.0) * (1.0 - mus_np**2)

    assert np.allclose(np.asarray(Pmn)[:, 2, 2], Pmncheck, atol=_ATOL, rtol=_RTOL), "Pmn[:,2,2] mismatch"
    assert np.allclose(np.asarray(Hmn)[:, 0, 1], Hmncheck, atol=_ATOL, rtol=_RTOL), "Hmn[:,0,1] mismatch"


def test_spectral_transform() -> None:
    M = 106
    omega = 3.2e-5
    N, I, J, _dt, _lambdas, mus, w = ic.spectral_params(M)
    Pmn, _Hmn = st.PmnHmn(J, M, N, mus)

    # f(j,i) = 2*omega*mu(j) (constant in longitude)
    f = (2.0 * omega) * mus[:, None] * jnp.ones((1, I), dtype=float_dtype())

    fm = st.fwd_fft_trunc(f, I, M)
    fmn = st.fwd_leg(fm, J, M, N, Pmn, w)

    fmncheck = np.zeros((M + 1, N + 1), dtype=(np.complex128 if _X64 else np.complex64))
    fmncheck[0, 1] = omega / np.sqrt(0.375)

    assert np.allclose(np.asarray(fmn), fmncheck, atol=_ATOL, rtol=_RTOL), "fmn mismatch"


def test_spectral_transform_forward_inverse() -> None:
    M = 63
    omega = 7.2921159e-5
    a = 6.37122e6
    a1 = np.pi / 2
    test = 1
    Phibar = 3.0e3

    N, I, J, _dt, lambdas, mus, w = ic.spectral_params(M)
    Pmn, _Hmn = st.PmnHmn(J, M, N, mus)

    SU0, sina, cosa, etaamp, Phiamp = ic.test1_init(a, omega, a1)
    etaic0, _etaic1, _deltaic0, _deltaic1, Phiic0, _Phiic1 = ic.state_var_init(
        I, J, mus, lambdas, test, etaamp, a, sina, cosa, Phibar, Phiamp
    )
    Uic, _Vic = ic.velocity_init(I, J, SU0, cosa, sina, mus, lambdas, test)

    Uicm = st.fwd_fft_trunc(Uic, I, M)
    Uicmn = st.fwd_leg(Uicm, J, M, N, Pmn, w)
    Uicmnew = st.invrs_leg(Uicmn, I, J, M, N, Pmn)
    Uicnew = st.invrs_fft(Uicmnew, I)

    # Physical-space fields are complex from IFFT but should be real to round-off.
    assert np.allclose(
        np.asarray(Uic),
        np.asarray(jnp.real(Uicnew)),
        atol=_ATOL,
        rtol=_RTOL,
    ), "forward+inverse mismatch"


def test_wind_transform() -> None:
    # Wind -> (eta,delta) -> wind
    M = 106
    omega = 7.2921159e-5
    a = 6.37122e6
    a1 = np.pi / 4
    test = 1
    dt = 30.0
    Phibar = 1.0e3

    N, I, J, _dt, lambdas, mus, w = ic.spectral_params(M)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    fmn = jnp.zeros((M + 1, N + 1), dtype=float_dtype()).at[0, 1].set(omega / jnp.sqrt(0.375))

    tstepcoeffmn = tstep.tstepcoeffmn(M, N, a)
    tstepcoeff = tstep.tstepcoeff(J, M, dt, mus, a)
    mJarray = tstep.mJarray(J, M)
    marray = tstep.marray(M, N)

    SU0, sina, cosa, etaamp, Phiamp = ic.test1_init(a, omega, a1)
    etaic0, _etaic1, deltaic0, _deltaic1, Phiic0, _Phiic1 = ic.state_var_init(
        I, J, mus, lambdas, test, etaamp, a, sina, cosa, Phibar, Phiamp
    )
    U, V = ic.velocity_init(I, J, SU0, cosa, sina, mus, lambdas, test)

    Um = st.fwd_fft_trunc(U, I, M)
    Vm = st.fwd_fft_trunc(V, I, M)

    etanew, deltanew, etamnnew, deltamnnew = st.diagnostic_eta_delta(
        Um, Vm, fmn, I, J, M, N, Pmn, Hmn, w, tstepcoeff, mJarray, dt
    )

    Unew, Vnew = st.invrsUV(deltamnnew, etamnnew, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray)

    assert np.allclose(np.asarray(U), np.asarray(jnp.real(Unew)), atol=_ATOL, rtol=_RTOL), "U error too high"
    assert np.allclose(np.asarray(V), np.asarray(jnp.real(Vnew)), atol=_ATOL, rtol=_RTOL), "V error too high"


def test_vorticity_divergence_transform() -> None:
    # (eta,delta) -> wind -> (eta,delta)
    M = 63
    omega = 7.2921159e-5
    a = 6.37122e6
    a1 = np.pi / 4
    test = 1
    dt = 30.0
    Phibar = 4.0e4

    N, I, J, _dt, lambdas, mus, w = ic.spectral_params(M)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)

    fmn = jnp.zeros((M + 1, N + 1), dtype=float_dtype()).at[0, 1].set(omega / jnp.sqrt(0.375))

    tstepcoeffmn = tstep.tstepcoeffmn(M, N, a)
    tstepcoeff = tstep.tstepcoeff(J, M, dt, mus, a)
    mJarray = tstep.mJarray(J, M)
    marray = tstep.marray(M, N)

    SU0, sina, cosa, etaamp, Phiamp = ic.test1_init(a, omega, a1)
    etaic0, _etaic1, deltaic0, _deltaic1, _Phiic0, _Phiic1 = ic.state_var_init(
        I, J, mus, lambdas, test, etaamp, a, sina, cosa, Phibar, Phiamp
    )

    deltam = st.fwd_fft_trunc(deltaic0, I, M)
    deltamn = st.fwd_leg(deltam, J, M, N, Pmn, w)

    etam = st.fwd_fft_trunc(etaic0, I, M)
    etamn = st.fwd_leg(etam, J, M, N, Pmn, w)

    U, V = st.invrsUV(deltamn, etamn, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray)

    Um = st.fwd_fft_trunc(U, I, M)
    Vm = st.fwd_fft_trunc(V, I, M)

    etanew, deltanew, _etamnnew, _deltamnnew = st.diagnostic_eta_delta(
        Um, Vm, fmn, I, J, M, N, Pmn, Hmn, w, tstepcoeff, mJarray, dt
    )

    assert np.allclose(np.asarray(etaic0), np.asarray(jnp.real(etanew)), atol=_ATOL, rtol=_RTOL), "eta error too high"
    assert np.allclose(np.asarray(deltaic0), np.asarray(jnp.real(deltanew)), atol=_ATOL, rtol=_RTOL), "delta error too high"


if __name__ == "__main__":
    test_init()
    test_Pmn_Hmn()
    test_spectral_transform()
    test_spectral_transform_forward_inverse()
    test_wind_transform()
    test_vorticity_divergence_transform()
    print("All tests passed!")
