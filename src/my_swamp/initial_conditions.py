"""
Initialization routines for SWAMPE (JAX port).

This is a line-by-line faithful translation of the original SWAMPE numpy
initialization logic, but vectorized and implemented in JAX.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

from . import spectral_transform as st


def test1_init(a: float, omega: float, a1: float) -> Tuple[float, float, float, float, float]:
    """
    Initializes the parameters from Test 1 in Williamson et al. (1992),
    Advection of Cosine Bell over the Pole.

    Returns:
        SU0, sina, cosa, etaamp, Phiamp
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    a1 = jnp.asarray(a1, dtype=jnp.float64)

    SU0 = 2.0 * jnp.pi * a / (3600.0 * 24.0 * 12.0)
    sina = jnp.sin(a1)
    cosa = jnp.cos(a1)
    etaamp = 2.0 * ((SU0 / a) + omega)
    Phiamp = (SU0 * a * omega + 0.5 * SU0**2)
    return SU0, sina, cosa, etaamp, Phiamp


def spectral_params(M: int):
    """
    Generates the resolution parameters according to Table 1 and 2 from Jakob et al. (1993).

    Returns:
        N, I, J, dt, lambdas, mus, w
    """
    N = int(M)

    if M == 42:
        J = 64
        I = 128
        dt = 1200
    elif M == 63:
        J = 96
        I = 192
        dt = 900
    elif M == 106:
        J = 160
        I = 320
        dt = 600
    else:
        raise ValueError(f"Unsupported value of M={M}. Only 42, 63, and 106 are supported.")

    lambdas = st.build_lambdas(I, dtype=jnp.float64)
    mus, w = st.gauss_legendre(J, dtype=jnp.float64)
    return N, I, J, dt, lambdas, mus, w


def state_var_init(
    I: int,
    J: int,
    mus: jnp.ndarray,
    lambdas: jnp.ndarray,
    test: Optional[int],
    etaamp: float,
    *args,
):
    """
    Initializes state variables (eta, delta, Phi) in physical space.

    Returns:
        etaic0, etaic1, deltaic0, deltaic1, Phiic0, Phiic1
    """
    I = int(I)
    J = int(J)

    mu = jnp.asarray(mus, dtype=jnp.float64)[:, None]          # (J,1)
    lam = jnp.asarray(lambdas, dtype=jnp.float64)[None, :]     # (1,I)
    sqrt_1m = jnp.sqrt(jnp.maximum(0.0, 1.0 - mu**2))

    etaamp = jnp.asarray(etaamp, dtype=jnp.float64)

    deltaic0 = jnp.zeros((J, I), dtype=jnp.float64)
    Phiic0 = jnp.zeros((J, I), dtype=jnp.float64)

    if test is not None:
        if len(args) != 5:
            raise ValueError("For test!=None, expected args=(a,sina,cosa,Phibar,Phiamp).")
        a, sina, cosa, Phibar, Phiamp = args
        a = jnp.asarray(a, dtype=jnp.float64)
        sina = jnp.asarray(sina, dtype=jnp.float64)
        cosa = jnp.asarray(cosa, dtype=jnp.float64)
        Phibar = jnp.asarray(Phibar, dtype=jnp.float64)
        Phiamp = jnp.asarray(Phiamp, dtype=jnp.float64)

    if test == 1:
        # Test 1: cosine bell bump in geopotential, vorticity set by solid-body rotation tilt
        latlonarg = -jnp.cos(lam) * sqrt_1m * sina + mu * cosa
        etaic0 = etaamp * latlonarg

        bumpr = a / 3.0
        mucenter = 0.0
        lambdacenter = 3.0 * jnp.pi / 2.0

        # With mucenter=0, the expression simplifies but we keep the original form.
        dist_arg = mucenter * mu + jnp.cos(jnp.arcsin(mucenter)) * jnp.cos(jnp.arcsin(mu)) * jnp.cos(lam - lambdacenter)
        dist_arg = jnp.clip(dist_arg, -1.0, 1.0)
        dist = a * jnp.arccos(dist_arg)

        bump = (Phibar / 2.0) * (1.0 + jnp.cos(jnp.pi * dist / bumpr))
        Phiic0 = jnp.where(dist < bumpr, bump, 0.0)

    elif test == 2:
        # Test 2: balanced zonal flow (Williamson Test 2 as in stswm)
        latlonarg = -jnp.cos(lam) * sqrt_1m * sina + mu * cosa
        etaic0 = etaamp * latlonarg
        Phiic0 = (Phibar - Phiamp) * (latlonarg**2)

    else:
        # Default: eta depends only on mu (sina=0, cosa=1)
        etaic0 = etaamp * jnp.broadcast_to(mu, (J, I))
    etaic1 = etaic0
    deltaic1 = deltaic0
    Phiic1 = Phiic0
    return etaic0, etaic1, deltaic0, deltaic1, Phiic0, Phiic1


def velocity_init(
    I: int,
    J: int,
    SU0: float,
    cosa: float,
    sina: float,
    mus: jnp.ndarray,
    lambdas: jnp.ndarray,
    test: Optional[int],
):
    """
    Initializes the wind components U (zonal) and V (meridional) in physical space.

    Returns:
        Uic, Vic (both shape (J,I))
    """
    I = int(I)
    J = int(J)

    mu = jnp.asarray(mus, dtype=jnp.float64)[:, None]
    lam = jnp.asarray(lambdas, dtype=jnp.float64)[None, :]
    sqrt_1m = jnp.sqrt(jnp.maximum(0.0, 1.0 - mu**2))

    SU0 = jnp.asarray(SU0, dtype=jnp.float64)
    cosa = jnp.asarray(cosa, dtype=jnp.float64)
    sina = jnp.asarray(sina, dtype=jnp.float64)

    if test == 1:
        Uic = SU0 * (sqrt_1m * cosa + mu * jnp.cos(lam) * sina) * sqrt_1m
        Vic = -SU0 * jnp.sin(lam) * sina * sqrt_1m
    elif test == 2:
        Uic = SU0 * (sqrt_1m * cosa + jnp.cos(lam) * mu * sina)
        Vic = -SU0 * (jnp.sin(lam) * sina)
    else:
        Uic = jnp.zeros((J, I), dtype=jnp.float64)
        Vic = jnp.zeros((J, I), dtype=jnp.float64)

    return Uic, Vic


def ABCDE_init(
    Uic: jnp.ndarray,
    Vic: jnp.ndarray,
    etaic0: jnp.ndarray,
    Phiic0: jnp.ndarray,
    mus: jnp.ndarray,
    I: int,
    J: int,
):
    """
    Initializes the auxiliary nonlinear components:
        A=U*eta, B=V*eta, C=U*Phi, D=V*Phi,
        E=(U^2+V^2)/(2*(1-mu^2)).
    """
    I = int(I)
    J = int(J)

    mu = jnp.asarray(mus, dtype=jnp.float64)[:, None]
    denom = 2.0 * (1.0 - mu**2)

    Aic = Uic * etaic0
    Bic = Vic * etaic0
    Cic = Uic * Phiic0
    Dic = Vic * Phiic0
    Eic = (Uic * Uic + Vic * Vic) / denom

    return Aic, Bic, Cic, Dic, Eic


def coriolismn(M: int, omega: float) -> jnp.ndarray:
    """
    Initializes the Coriolis parameter in spectral space.
    """
    M = int(M)
    omega = jnp.asarray(omega, dtype=jnp.float64)
    fmn = jnp.zeros((M + 1, M + 1), dtype=jnp.float64)
    fmn = fmn.at[0, 1].set(omega / jnp.sqrt(0.375))
    return fmn
