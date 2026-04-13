# ruff: noqa: E741
"""
This module contains the functions used for the evaluation of stellar forcing (insolation).
Matches the original SWAMPE numpy implementation exactly.
"""
from __future__ import annotations

import jax.numpy as jnp


def Phieqfun(
    Phibar: float,
    DPhieq: float,
    lambdas: jnp.ndarray,
    mus: jnp.ndarray,
    I: int,
    J: int,
    g: float
) -> jnp.ndarray:
    """Evaluates the equilibrium geopotential from Perez-Becker and Showman (2013).

    Parameters
    ----------
    Phibar : float
        Reference (mean) geopotential in SI units.
    DPhieq : float
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
    Phibar: float,
    taurad: float
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
    Phibar: float,
    taudrag: float
):
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
    
    # Compute Ru, Rv
    phi_total = Phi + Phibar
    Ru = -U * Qclone / phi_total
    Rv = -V * Qclone / phi_total
    
    # Handle taudrag == -1 case (no Rayleigh drag) without Python branching.
    taudrag_arr = jnp.asarray(taudrag)
    no_drag = taudrag_arr == -1
    taudrag_eff = jnp.where(no_drag, 1.0, taudrag_arr)

    F_drag = Ru - (U / taudrag_eff)
    G_drag = Rv - (V / taudrag_eff)

    F = jnp.where(no_drag, Ru, F_drag)
    G = jnp.where(no_drag, Rv, G_drag)

    return F, G
