"""
This module contains the initialization functions.
Matches the original SWAMPE numpy implementation exactly.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as onp
import jax.numpy as jnp


def test1_init(a: float, omega: float, a1: float) -> Tuple[float, float, float, float, float]:
    """
    Initializes the parameters from Test 1 in Williamson et al. (1992),
    Advection of Cosine Bell over the Pole.
    
    Parameters
    ----------
    a : float
        Planetary radius, in meters.
    omega : float
        Planetary rotation rate, in radians per second.
    a1 : float
        Angle of advection, in radians.

    Returns
    -------
    SU0 : float
        Amplitude parameter from Test 1 in Williamson et al. (1992)
    sina : float
        sine of the angle of advection.
    cosa : float
        cosine of the angle of advection.
    etaamp : float
        Amplitude of absolute vorticity.
    Phiamp : float
        Amplitude of geopotential.
    """
    # Parameters for Test 1 in Williamson et al. (1992)
    SU0 = 2.0 * onp.pi * a / (3600.0 * 24 * 12)
    sina = onp.sin(a1)  # sine of the angle of advection
    cosa = onp.cos(a1)  # cosine of the angle of advection
    etaamp = 2.0 * ((SU0 / a) + omega)  # relative vorticity amplitude
    Phiamp = SU0 * a * omega + 0.5 * SU0**2  # geopotential height amplitude
    
    return float(SU0), float(sina), float(cosa), float(etaamp), float(Phiamp)


def spectral_params(M: int):
    """
    Generates the resolution parameters according to Table 1 and 2 from Jakob et al. (1993).
    Note the timesteps are appropriate for Earth-like forcing. More strongly forced planets 
    will need shorter timesteps.

    Parameters
    ----------
    M : int
        spectral resolution

    Returns
    -------
    N : int
        the highest degree of the Legendre functions for m = 0
    I : int
        number of longitudes
    J : int
        number of Gaussian latitudes
    dt : float
        timestep length, in seconds
    lambdas : array of float
        evenly spaced longitudes of length I
    mus : array of float
        Gaussian latitudes of length J
    w : array of float
        Gaussian weights of length J
    """
    N = M
    
    # Set dimensions according to Jakob and Hack (1993), Tables 1, 2, and 3
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
        raise ValueError(f'Unsupported value of M={M}. Only 42, 63, and 106 are supported')
    from . import spectral_transform as st

    lambdas = st.build_lambdas(I)
    mus = st.build_mus(J)
    w = st.build_w(J)
    return N, I, J, dt, lambdas, mus, w


def state_var_init(
    I: int,
    J: int,
    mus: jnp.ndarray,
    lambdas: jnp.ndarray,
    test: Optional[int],
    etaamp: float,
    *args
):
    """
    Initializes state variables.

    Parameters
    ----------
    I : int
        number of longitudes.
    J : int
        number of latitudes.
    mus : array of float
        Array of Gaussian latitudes of length J.
    lambdas : array of float
        Uniformly spaced longitudes of length I.
    test : int or None
        The number of the regime being tested from Williamson et al. (1992)
    etaamp : float
        Amplitude of absolute vorticity.
    *args : 
        Additional initialization parameters for tests from Williamson et al. (1992):
        a, sina, cosa, Phibar, Phiamp

    Returns
    -------
    etaic0 : array (J, I)
        Initial condition for absolute vorticity
    etaic1 : array (J, I)
        Second initial condition for absolute vorticity
    deltaic0 : array (J, I)
        Initial condition for divergence
    deltaic1 : array (J, I)
        Second initial condition for divergence
    Phiic0 : array (J, I)
        Initial condition for geopotential
    Phiic1 : array (J, I)
        Second initial condition for geopotential
    """
    etaic0 = jnp.zeros((J, I))
    Phiic0 = jnp.zeros((J, I))
    deltaic0 = jnp.zeros((J, I))
    
    # Parse args if test is not None
    if test is not None:
        if len(args) >= 5:
            a, sina, cosa, Phibar, Phiamp = args[:5]
        else:
            raise ValueError("test=1 or test=2 requires args: a, sina, cosa, Phibar, Phiamp")
    
    if test == 1:
        # Williamson Test 1 - Advection of cosine bell
        bumpr = a / 3  # radius of the bump
        mucenter = 0
        lambdacenter = 3 * onp.pi / 2
        
        # Build arrays for vectorized computation
        lam = lambdas[None, :]  # (1, I)
        mu = mus[:, None]       # (J, 1)
        
        # eta initialization
        etaic0 = etaamp * (-jnp.cos(lam) * jnp.sqrt(1 - mu**2) * sina + mu * cosa)
        
        # Phi initialization with bump
        # dist = a * arccos(mucenter*mu + cos(arcsin(mucenter))*cos(arcsin(mu))*cos(lambda - lambdacenter))
        # Since mucenter = 0, this simplifies
        coslat = jnp.sqrt(1 - mu**2)
        dist = a * jnp.arccos(coslat * jnp.cos(lam - lambdacenter))
        
        # Where dist < bumpr, add the bump
        inbump = dist < bumpr
        bump = (Phibar / 2) * (1 + jnp.cos(onp.pi * dist / bumpr))
        Phiic0 = jnp.where(inbump, bump, 0.0)
        
    elif test == 2:
        # Williamson Test 2 - Steady state nonlinear zonal geostrophic flow
        lam = lambdas[None, :]  # (1, I)
        mu = mus[:, None]       # (J, 1)
        
        latlonarg = -jnp.cos(lam) * jnp.sqrt(1 - mu**2) * sina + mu * cosa
        etaic0 = etaamp * latlonarg
        Phiic0 = (Phibar - Phiamp) * latlonarg**2
        
    else:
        # Default initialization (test=None)
        lam = lambdas[None, :]
        mu = mus[:, None]
        # sina=0, cosa=1 case
        etaic0 = etaamp * (-jnp.cos(lam) * jnp.sqrt(1 - mu**2) * 0 + mu * 1)
    
    etaic1 = etaic0  # need two time steps to initialize
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
    Initializes the zonal and meridional components of the wind vector field.

    Parameters
    ----------
    I : int
        number of longitudes.
    J : int
        number of latitudes.
    SU0 : float
        Amplitude parameter from Test 1 in Williamson et al. (1992)
    cosa : float
        cosine of the angle of advection.
    sina : float
        sine of the angle of advection.
    mus : array of float
        Array of Gaussian latitudes of length J
    lambdas : array of float
        Array of uniformly spaced longitudes of length I.
    test : int or None
        when applicable, number of test from Williamson et al. (1992).

    Returns
    -------
    Uic : array (J, I)
        the initial condition for the zonal velocity component
    Vic : array (J, I)
        the initial condition for the meridional velocity component
    """
    lam = lambdas[None, :]  # (1, I)
    mu = mus[:, None]       # (J, 1)
    coslat = jnp.sqrt(jnp.maximum(0.0, 1 - mu**2))  # cos(arcsin(mu))
    
    if test == 1:
        # Test 1: includes extra coslat factor
        Uic = SU0 * (coslat * cosa + mu * jnp.cos(lam) * sina) * coslat
        Vic = -SU0 * jnp.sin(lam) * sina * coslat
        
    elif test == 2:
        # Test 2: no extra coslat factor
        Uic = SU0 * (coslat * cosa + jnp.cos(lam) * mu * sina)
        Vic = -SU0 * (jnp.sin(lam) * sina)
        
    else:
        # Default: zero winds
        Uic = jnp.zeros((J, I))
        Vic = jnp.zeros((J, I))
    
    return Uic, Vic


def ABCDE_init(
    Uic: jnp.ndarray,
    Vic: jnp.ndarray,
    etaic0: jnp.ndarray,
    Phiic0: jnp.ndarray,
    mus: jnp.ndarray,
    I: int,
    J: int
):
    """
    Initializes the auxiliary nonlinear components.
    
    Parameters
    ----------
    Uic : array (J, I)
        zonal velocity component
    Vic : array (J, I)
        meridional velocity component
    etaic0 : array (J, I)
        initial eta
    Phiic0 : array (J, I)
        initial Phi
    mus : array (J,)
        Gaussian latitudes
    I : int
        number of longitudes
    J : int
        number of latitudes

    Returns
    -------
    Aic, Bic, Cic, Dic, Eic : arrays (J, I)
        Nonlinear components
    """
    mu = mus[:, None]
    
    Aic = Uic * etaic0  # A = U*eta
    Bic = Vic * etaic0  # B = V*eta
    Cic = Uic * Phiic0  # C = U*Phi
    Dic = Vic * Phiic0  # D = V*Phi
    
    # E = (U^2 + V^2) / (2*(1-mu^2))
    denom = 2 * (1 - mu**2)
    denom = jnp.where(denom == 0, 1e-30, denom)  # avoid division by zero
    Eic = (Uic**2 + Vic**2) / denom
    
    return Aic, Bic, Cic, Dic, Eic


def coriolismn(M: int, omega: float) -> jnp.ndarray:
    """
    Initializes the Coriolis parameter in spectral space.
    
    Parameters
    ----------
    M : int
        Spectral dimension.
    omega : float
        Planetary rotation rate, in radians per second.

    Returns
    -------
    fmn : array (M+1, M+1)
        The Coriolis parameter in spectral space.
    """
    fmn = jnp.zeros((M + 1, M + 1))
    # Matches original formula: omega/sqrt(0.375)
    fmn = fmn.at[0, 1].set(omega / onp.sqrt(0.375))
    
    return fmn
