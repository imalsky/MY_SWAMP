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
    """
    Evaluates the equilibrium geopotential from Perez-Becker and Showman (2013).
    """
    lam = lambdas[None, :]  # (1, I)
    mu = mus[:, None]       # (J, 1)
    
    # Initialize to flat nightside geopotential
    PhieqMat = jnp.full((J, I), Phibar)
    
    # Only force the dayside: -pi/2 < lambda < pi/2
    # CORRECTED: Original uses strict inequality
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
    """
    Evaluates the radiative forcing on the geopotential.
    Q corresponds to the forcing from Perez-Becker and Showman (2013).
    """
    Q = (1 / taurad) * (Phieq - (Phi + Phibar))
    return Q


def Qfun_with_rampup(
    Phieq: jnp.ndarray,
    Phi: jnp.ndarray,
    Phibar: float,
    taurad: float,
    t: int,
    dt: float
) -> jnp.ndarray:
    """
    Evaluates the radiative forcing on the geopotential, but slowly ramps up 
    the forcing to improve stability for short radiative timescales.
    """
    # slowly ramp up over 15 hours
    time_elapsed = t * dt
    ramp_time = 15 * 3600  # 15 hours in seconds
    
    factor = jnp.where(time_elapsed < ramp_time, time_elapsed / ramp_time, 1.0)
    
    Q = factor * (1 / taurad) * (Phieq - (Phi + Phibar))
    return Q


def Rfun(
    U: jnp.ndarray,
    V: jnp.ndarray,
    Q: jnp.ndarray,
    Phi: jnp.ndarray,
    Phibar: float,
    taudrag: float
):
    """
    Evaluates the velocity forcing in Perez-Becker and Showman.
    
    Includes Q<0 handling and taudrag==-1 case from original.
    """
    # Clone Q and zero out negative values (mass loss prevention)
    Qclone = jnp.where(Q < 0, 0.0, Q)
    
    # Compute Ru, Rv
    phi_total = Phi + Phibar
    # Avoid division by zero
    denom = jnp.where(phi_total == 0.0, 1e-30, phi_total)
    
    Ru = -U * Qclone / denom
    Rv = -V * Qclone / denom
    
    # Reset to 0 if losing mass (Q < 0)
    Ru = jnp.where(Q < 0, 0.0, Ru)
    Rv = jnp.where(Q < 0, 0.0, Rv)
    
    # Handle taudrag == -1 case (no Rayleigh drag) without Python branching.
    taudrag_arr = jnp.asarray(taudrag)
    no_drag = taudrag_arr == -1
    taudrag_eff = jnp.where(no_drag, 1.0, taudrag_arr)

    F_drag = Ru - (U / taudrag_eff)
    G_drag = Rv - (V / taudrag_eff)

    F = jnp.where(no_drag, Ru, F_drag)
    G = jnp.where(no_drag, Rv, G_drag)

    return F, G


def Rfun_jax(
    U: jnp.ndarray,
    V: jnp.ndarray,
    Q: jnp.ndarray,
    Phi: jnp.ndarray,
    Phibar: float,
    taudrag: float,
    no_drag: bool = False
):
    """
    JAX-compatible version of Rfun that handles taudrag condition with explicit flag.
    
    Use no_drag=True to emulate taudrag==-1 behavior.
    """
    # Clone Q and zero out negative values
    Qclone = jnp.where(Q < 0, 0.0, Q)
    
    # Compute Ru, Rv
    phi_total = Phi + Phibar
    denom = jnp.where(phi_total == 0.0, 1e-30, phi_total)
    
    Ru = -U * Qclone / denom
    Rv = -V * Qclone / denom
    
    # Reset to 0 if losing mass
    Ru = jnp.where(Q < 0, 0.0, Ru)
    Rv = jnp.where(Q < 0, 0.0, Rv)
    
    # Apply drag if not disabled
    if no_drag:
        F = Ru
        G = Rv
    else:
        F = Ru - (U / taudrag)
        G = Rv - (V / taudrag)
    
    return F, G
