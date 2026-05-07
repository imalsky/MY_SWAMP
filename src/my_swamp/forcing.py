# ruff: noqa: E741
"""
This module contains the functions used for the evaluation of stellar forcing (insolation).
Matches the original SWAMPE numpy implementation exactly.
"""
from __future__ import annotations

import jax.numpy as jnp
from typing import Tuple

from .dtypes import Scalar


def Phieqfun(
    Phibar: Scalar,
    DPhieq: Scalar,
    lambdas: jnp.ndarray,
    mus: jnp.ndarray,
    I: int,
    J: int,
    g: Scalar,
) -> jnp.ndarray:
    """Evaluates the equilibrium geopotential from Perez-Becker and Showman (2013).

    Parameters
    ----------
    Phibar : Scalar
        Reference (mean) geopotential in SI units.
    DPhieq : Scalar
        Day-night equilibrium geopotential contrast.
    lambdas : jnp.ndarray
        Longitudes in radians with shape ``(I,)``.
    mus : jnp.ndarray
        Sine of Gaussian latitudes with shape ``(J,)``.
    I : int
        Number of longitude grid points.
    J : int
        Number of Gaussian latitude grid points.
    g : float
        Surface gravity (unused in the current formulation but kept for
        API compatibility with the original SWAMPE).

    Returns
    -------
    jnp.ndarray
        Equilibrium geopotential field with shape ``(J, I)``.
    """
    lam = lambdas[None, :]  # (1, I)
    mu = mus[:, None]       # (J, 1)
    
    # Initialize to flat nightside geopotential
    PhieqMat = jnp.full((J, I), Phibar)
    
    # Only force the dayside: -pi/2 < lambda < pi/2 (strict inequality, matching SWAMPE)
    dayside = (lambdas > -jnp.pi / 2) & (lambdas < jnp.pi / 2)  # (I,)
    daymask = dayside[None, :]  # (1, I)
    
    # Add forcing term on dayside
    term = DPhieq * jnp.cos(lam) * jnp.sqrt(1 - mu**2)
    PhieqMat = jnp.where(daymask, PhieqMat + term, PhieqMat)
    
    return PhieqMat


def Qfun(
    Phieq: jnp.ndarray,
    Phi: jnp.ndarray,
    Phibar: Scalar,
    taurad: Scalar,
) -> jnp.ndarray:
    """Evaluates the radiative forcing on the geopotential.

    Q = (Phieq - (Phi + Phibar)) / taurad, following Perez-Becker and
    Showman (2013).

    Parameters
    ----------
    Phieq : jnp.ndarray
        Equilibrium geopotential field with shape ``(J, I)``.
    Phi : jnp.ndarray
        Current geopotential perturbation with shape ``(J, I)``.
    Phibar : float
        Reference (mean) geopotential.
    taurad : float
        Radiative relaxation timescale in seconds.

    Returns
    -------
    jnp.ndarray
        Radiative forcing field with shape ``(J, I)``.
    """
    Q = (1 / taurad) * (Phieq - (Phi + Phibar))
    return Q


def Rfun(
    U: jnp.ndarray,
    V: jnp.ndarray,
    Q: jnp.ndarray,
    Phi: jnp.ndarray,
    Phibar: Scalar,
    taudrag: Scalar,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluates the velocity forcing from Perez-Becker and Showman (2013).

    Negative Q values are clamped to zero (mass-loss prevention).  When
    ``taudrag == -1``, Rayleigh drag is disabled.

    Parameters
    ----------
    U : jnp.ndarray
        Zonal wind field with shape ``(J, I)``.
    V : jnp.ndarray
        Meridional wind field with shape ``(J, I)``.
    Q : jnp.ndarray
        Radiative forcing field with shape ``(J, I)``.
    Phi : jnp.ndarray
        Current geopotential perturbation with shape ``(J, I)``.
    Phibar : float
        Reference (mean) geopotential.
    taudrag : float
        Drag timescale in seconds.  ``-1`` disables Rayleigh drag.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        ``(F, G)`` velocity forcing fields for zonal and meridional
        directions, each with shape ``(J, I)``.
    """
    # Clone Q and zero out negative values (mass loss prevention)
    Qclone = jnp.where(Q < 0, 0.0, Q)

    # Compute Ru, Rv. Guard against transient Phi values driving Phi+Phibar
    # to zero. This is a small departure from reference SWAMPE (which divides
    # without a guard), but only affects pathological transients; whenever
    # Qclone is non-zero in well-posed runs, Phi+Phibar is far from zero.
    phi_total = Phi + Phibar
    phi_total_safe = jnp.where(jnp.abs(phi_total) > 0, phi_total, jnp.finfo(phi_total.dtype).tiny)
    Ru = -U * Qclone / phi_total_safe
    Rv = -V * Qclone / phi_total_safe
    
    # Handle taudrag == -1 case (no Rayleigh drag) without Python branching.
    taudrag_arr = jnp.asarray(taudrag)
    no_drag = taudrag_arr == -1
    taudrag_eff = jnp.where(no_drag, 1.0, taudrag_arr)

    F_drag = Ru - (U / taudrag_eff)
    G_drag = Rv - (V / taudrag_eff)

    F = jnp.where(no_drag, Ru, F_drag)
    G = jnp.where(no_drag, Rv, G_drag)

    return F, G
