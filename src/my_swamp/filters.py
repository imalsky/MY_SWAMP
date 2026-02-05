"""
This module contains the functions associated with filters needed for numerical stability.
Matches the original SWAMPE numpy implementation.
"""
from __future__ import annotations

import jax.numpy as jnp


def modal_splitting(Xidataslice: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Applies the modal splitting filter from Hack and Jakob (1992).
    
    Xidataslice: shape (3, J, I) where axis 0 is time level [t-1, t, t+1]
    Returns: filtered data at time t, shape (J, I)
    """
    newxi = Xidataslice[1, :, :] + alpha * (
        Xidataslice[0, :, :] - 2 * Xidataslice[1, :, :] + Xidataslice[2, :, :]
    )
    return newxi


def diffusion(Ximn: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the diffusion filter described in Gelb and Gleeson (eq. 12).
    
    Ximn: spectral coefficients, shape (M+1, N+1)
    sigma: filter coefficients, shape (M+1, N+1)
    Returns: filtered spectral coefficients
    """
    return Ximn * sigma


def sigma(M: int, N: int, K4: float, a: float, dt: float) -> jnp.ndarray:
    """
    Computes the coefficient for the fourth degree diffusion filter 
    described in Gelb and Gleeson (eq. 12) for vorticity and divergence.
    
    CORRECTED: Uses original implicit filter formulation.
    """
    nvec = jnp.arange(N + 1, dtype=jnp.float64)
    ncoeff = (nvec * nvec / a**2) * ((nvec + 1) * (nvec + 1) / a**2)
    factor1 = 4 / a**4
    factor2 = 2 * dt * K4
    
    sigmacoeff = 1 + factor2 * (ncoeff - factor1)
    sigmas = 1 / sigmacoeff
    
    # Broadcast to (M+1, N+1)
    return jnp.tile(sigmas[None, :], (M + 1, 1))


def sigmaPhi(M: int, N: int, K4: float, a: float, dt: float) -> jnp.ndarray:
    """
    Computes the coefficient for the fourth degree diffusion filter 
    described in Gelb and Gleeson (eq. 12) for geopotential.
    
    Uses original implicit filter formulation (no factor1 subtraction).
    """
    nvec = jnp.arange(N + 1, dtype=jnp.float64)
    ncoeff = (nvec * nvec / a**2) * ((nvec + 1) * (nvec + 1) / a**2)
    factor2 = 2 * dt * K4
    
    sigmacoeff = 1 + factor2 * ncoeff
    sigmas = 1 / sigmacoeff
    
    return jnp.tile(sigmas[None, :], (M + 1, 1))


def sigma6(M: int, N: int, K6: float, a: float, dt: float) -> jnp.ndarray:
    """
    Computes the coefficient for the sixth degree diffusion filter 
    for vorticity and divergence.
    
    Uses original implicit filter formulation.
    """
    nvec = jnp.arange(N + 1, dtype=jnp.float64)
    
    # n^3 * (n+1)^3 / a^6
    ncoeff = ((nvec * nvec * nvec) / a**3) * (((nvec + 1) * (nvec + 1) * (nvec + 1)) / a**3)
    factor1 = 8 / a**6
    factor2 = 2 * dt * K6
    
    sigmacoeff = 1 + factor2 * (ncoeff - factor1)
    sigmas = 1 / sigmacoeff
    
    return jnp.tile(sigmas[None, :], (M + 1, 1))


def sigma6Phi(M: int, N: int, K6: float, a: float, dt: float) -> jnp.ndarray:
    """
    Computes the coefficient for the sixth degree diffusion filter for geopotential.
    
    Uses original implicit filter formulation (no factor1 subtraction).
    """
    nvec = jnp.arange(N + 1, dtype=jnp.float64)
    
    ncoeff = ((nvec * nvec * nvec) / a**3) * (((nvec + 1) * (nvec + 1) * (nvec + 1)) / a**3)
    factor2 = 2 * dt * K6
    
    sigmacoeff = 1 + factor2 * ncoeff
    sigmas = 1 / sigmacoeff
    
    return jnp.tile(sigmas[None, :], (M + 1, 1))
