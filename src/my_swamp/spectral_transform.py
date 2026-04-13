# -*- coding: utf-8 -*-
# ruff: noqa: E741
"""my_swamp.spectral_transform

Spectral transform utilities for SWAMPE.

Parity intent
-------------
This module is written to match the reference NumPy/SciPy SWAMPE implementation
as closely as possible, while keeping the time-critical transforms in JAX so
they can be JIT-compiled, GPU-accelerated, and differentiated.

Static precomputation
---------------------
The Gaussian quadrature nodes/weights and the associated Legendre basis are
*static* with respect to the time integration. They are computed once in Python
and then stored as JAX arrays.

For compatibility across SciPy versions:

- Gaussian quadrature nodes/weights:
    * Prefer ``scipy.special.roots_legendre`` when available.
    * Fallback to ``numpy.polynomial.legendre.leggauss`` otherwise.

- Associated Legendre polynomials and derivatives:
    * Prefer ``scipy.special.assoc_legendre_p_all`` when available.
    * Fallback to ``scipy.special.lpmn`` when needed.
    * Fallback to a recurrence-based implementation that matches SciPy's
      ``lpmn`` (including the Condon–Shortley phase) up to floating round-off.

The runtime transforms (FFT and Legendre matrix multiplications) are implemented
with JAX so they can run on GPU and be differentiated.
"""

from __future__ import annotations

from typing import Tuple

import math

import numpy as np

import jax.numpy as jnp

from .dtypes import float_dtype


try:
    import scipy.special as sp
except ImportError:  # pragma: no cover
    sp = None


def build_lambdas(I: int, dtype=None) -> jnp.ndarray:
    """Return uniformly spaced longitudes in [-pi, pi).
    
    Parameters
    ----------
    I : int
        Number of longitude points.
    dtype : optional
        Floating dtype for the returned array. If omitted, uses
        :func:`my_swamp.dtypes.float_dtype`.
    
    Notes
    -----
    The reference SWAMPE code constructs longitudes as a uniform grid in
    ``[-pi, pi)`` with ``endpoint=False``. We keep the same convention here.
    
    Returns
    -------
    jnp.ndarray
    """
    I = int(I)
    if dtype is None:
        dtype = float_dtype()
    return jnp.linspace(-jnp.pi, jnp.pi, num=I, endpoint=False, dtype=dtype)


def gauss_legendre(J: int, dtype=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Gaussian quadrature nodes/weights (mus, w) for order J.
    
    The reference SWAMPE uses ``scipy.special.roots_legendre(J)``.
    For portability, this implementation falls back to
    ``numpy.polynomial.legendre.leggauss(J)`` if SciPy is unavailable.
    
    Parameters
    ----------
    J : int
        Number of Gaussian latitude nodes.
    dtype : Any
        Floating dtype for the returned arrays. If omitted, uses
        :func:`my_swamp.dtypes.float_dtype`.
    
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Tuple ``(mus, w)`` containing Gaussian latitudes and quadrature weights,
        each with shape ``(J,)``.
    """
    if dtype is None:
        dtype = float_dtype()

    J = int(J)

    if sp is not None and hasattr(sp, "roots_legendre"):
        mus_np, w_np = sp.roots_legendre(J)
    else:
        # NumPy fallback (no SciPy dependency)
        from numpy.polynomial.legendre import leggauss

        mus_np, w_np = leggauss(J)

    return jnp.asarray(mus_np, dtype=dtype), jnp.asarray(w_np, dtype=dtype)


def _scaling_table(M: int, N: int) -> np.ndarray:
    """Scaling table matching reference SWAMPE's factorial-based normalization.
    
    Parameters
    ----------
    M : int
        Maximum zonal wavenumber.
    N : int
        Maximum total degree.
    
    Returns
    -------
    np.ndarray
        Real scaling factors with shape ``(M+1, N+1)`` used to normalize the
        associated Legendre basis in the same way as the reference SWAMPE code.
    """
    M = int(M)
    N = int(N)
    scale = np.zeros((M + 1, N + 1), dtype=np.float64)
    for m in range(M + 1):
        for n in range(N + 1):
            if n < m:
                scale[m, n] = 0.0
            else:
                # Reference code:
                #   sqrt((((2*n)+1)*factorial(n-m)) / (2*factorial(n+m)))
                numer = ((2 * n) + 1) * math.factorial(n - m)
                denom = 2 * math.factorial(n + m)
                scale[m, n] = math.sqrt(numer / denom)
    return scale


def _lpmn_fallback(M: int, N: int, x: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback for ``scipy.special.lpmn`` using a stable recurrence.

    Notes
    -----
    - Includes the Condon–Shortley phase (``(-1)^m``), matching SciPy.
    - The derivative is with respect to ``x`` (not latitude).
    
    Parameters
    ----------
    M : int
    N : int
    x : float

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple ``(P, dP)`` of NumPy arrays with shape ``(M+1, N+1)`` matching
        SciPy's ``lpmn`` ordering, where the first index is order ``m`` and the
        second index is degree ``n``.
    """
    M = int(M)
    N = int(N)

    P = np.zeros((M + 1, N + 1), dtype=np.float64)
    dP = np.zeros((M + 1, N + 1), dtype=np.float64)

    P[0, 0] = 1.0

    if M == 0 and N == 0:
        return P, dP

    # sqrt(1 - x^2) is well-defined for Gauss–Legendre nodes (|x|<1).
    s = math.sqrt(max(0.0, 1.0 - x * x))

    # Diagonal terms P_m^m via a stable recurrence:
    # P_m^m(x) = -(2m-1) * sqrt(1-x^2) * P_{m-1}^{m-1}(x)
    for m in range(1, M + 1):
        if m <= N:
            P[m, m] = -(2 * m - 1) * s * P[m - 1, m - 1]

    # First off-diagonal: P_m^{m+1}(x) = (2m+1) x P_m^m(x)
    for m in range(0, min(M, N - 1) + 1):
        P[m, m + 1] = (2 * m + 1) * x * P[m, m]

    # Upward recurrence in n (degree):
    # P_m^n(x) = ((2n-1)x P_m^{n-1}(x) - (n+m-1) P_m^{n-2}(x)) / (n-m)
    for m in range(0, M + 1):
        for n in range(m + 2, N + 1):
            P[m, n] = ((2 * n - 1) * x * P[m, n - 1] - (n + m - 1) * P[m, n - 2]) / (n - m)

    # Derivatives using:
    # d/dx P_m^n(x) = (n x P_m^n(x) - (n+m) P_m^{n-1}(x)) / (x^2 - 1)
    denom = (x * x) - 1.0
    if denom == 0.0:
        denom = np.finfo(np.float64).eps

    for m in range(0, M + 1):
        if m <= N:
            if m == 0:
                dP[m, m] = 0.0
            else:
                dP[m, m] = (m * x * P[m, m]) / denom

        for n in range(m + 1, N + 1):
            dP[m, n] = (n * x * P[m, n] - (n + m) * P[m, n - 1]) / denom

    return P, dP


def PmnHmn(J: int, M: int, N: int, mus: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute associated Legendre polynomials and derivatives at Gaussian latitudes.

    Implementation notes
    --------------------
    The reference SWAMPE uses SciPy's ``special.lpmn`` and then applies:
    
    - a factorial-based normalization (see `_scaling_table`)
    - a sign flip for odd m (which cancels SciPy's Condon–Shortley phase)
    
    For portability across SciPy versions, we:
    - use SciPy's ``assoc_legendre_p_all`` when present (preferred modern API)
    - otherwise use SciPy's ``lpmn`` when present
    - otherwise fall back to `_lpmn_fallback`, which matches SciPy's output up to
      floating round-off.
    
    Parameters
    ----------
    J : int
    M : int
    N : int
    mus : jnp.ndarray

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Tuple ``(Pmn, Hmn)`` of basis arrays with shape ``(J, M+1, N+1)``
        matching the reference SWAMPE normalization.
    """
    J = int(J)
    M = int(M)
    N = int(N)

    mus_np = np.asarray(mus, dtype=np.float64)

    Pmntemp = np.zeros((J, M + 1, N + 1), dtype=np.float64)
    Hmntemp = np.zeros((J, M + 1, N + 1), dtype=np.float64)

    use_assoc_p_all = (sp is not None) and hasattr(sp, "assoc_legendre_p_all")
    use_scipy_lpmn = (not use_assoc_p_all) and (sp is not None) and hasattr(sp, "lpmn")

    for j in range(J):
        mu = float(mus_np[j])
        if use_assoc_p_all:
            # assoc_legendre_p_all returns arrays indexed by [derivative_order, n, m]
            # for non-negative m at columns [0..M].
            assoc = np.asarray(sp.assoc_legendre_p_all(N, M, mu, diff_n=1), dtype=np.float64)
            P_j = np.swapaxes(assoc[0, : (N + 1), : (M + 1)], 0, 1)
            dP_j = np.swapaxes(assoc[1, : (N + 1), : (M + 1)], 0, 1)
        elif use_scipy_lpmn:
            # SciPy returns (Pmn, dP/dmu) with shape (M+1, N+1)
            P_j, dP_j = sp.lpmn(M, N, mu)
        else:
            P_j, dP_j = _lpmn_fallback(M, N, mu)

        Pmntemp[j, :, :] = P_j
        # Reference SWAMPE uses (1 - mu^2) * dP/dmu
        Hmntemp[j, :, :] = (1.0 - mu * mu) * dP_j

    scale = _scaling_table(M, N)  # (M+1, N+1)
    Pmn = Pmntemp * scale[None, :, :]
    Hmn = Hmntemp * scale[None, :, :]

    # Reference SWAMPE flips the sign for odd m and n>0.
    odd_m = (np.arange(M + 1) % 2) == 1
    Pmn[:, odd_m, 1:] *= -1.0
    Hmn[:, odd_m, 1:] *= -1.0

    return jnp.asarray(Pmn, dtype=float_dtype()), jnp.asarray(Hmn, dtype=float_dtype())


def fwd_fft_trunc(data: jnp.ndarray, I: int, M: int) -> jnp.ndarray:
    """Fourier transform along longitude, truncating to m=0..M.
    
    Parameters
    ----------
    data : jnp.ndarray
    I : int
    M : int
    
    Returns
    -------
    jnp.ndarray
    """
    I = int(I)
    M = int(M)
    datahat = jnp.fft.fft(data / I, n=I, axis=1)
    return datahat[:, : (M + 1)]


def fwd_fft_trunc_batch(data: jnp.ndarray, I: int, M: int) -> jnp.ndarray:
    """Batched Fourier transform along longitude, truncating to m=0..M.
    
    Parameters
    ----------
    data : jnp.ndarray
        Batched longitude fields with trailing shape ``(..., J, I)``.
    I : int
        Total longitude count used by the FFT.
    M : int
        Maximum retained zonal wavenumber.
    
    Returns
    -------
    jnp.ndarray
    """
    I = int(I)
    M = int(M)
    datahat = jnp.fft.fft(data / I, n=I, axis=-1)
    return datahat[..., : (M + 1)]


def invrs_fft(approxXim: jnp.ndarray, I: int) -> jnp.ndarray:
    """Inverse Fourier transform along longitude (expects full I coefficients).
    
    Parameters
    ----------
    approxXim : jnp.ndarray
        Fourier coefficients with longitude axis ``M+1`` or ``I`` already laid
        out for inverse transformation.
    I : int
        Number of longitude points in physical space.
    
    Returns
    -------
    jnp.ndarray
        Complex-valued longitude-space field with shape ``(..., I)``.
    """
    I = int(I)
    return jnp.fft.ifft(I * approxXim, n=I, axis=1)


def weighted_legendre_basis(Pmn: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Precompute w-weighted Legendre basis used by forward transforms.
    
    Parameters
    ----------
    Pmn : jnp.ndarray
        Associated Legendre basis with shape ``(J, M+1, N+1)``.
    w : jnp.ndarray
        Gaussian quadrature weights with shape ``(J,)``.
    
    Returns
    -------
    jnp.ndarray
        Weighted basis ``w[:, None, None] * Pmn`` with the same shape as
        ``Pmn``.
    """
    return jnp.asarray(w)[:, None, None] * Pmn


def fwd_leg_w(data: jnp.ndarray, Pmnw: jnp.ndarray) -> jnp.ndarray:
    """Forward Legendre transform with preweighted basis.
    
    Parameters
    ----------
    data : jnp.ndarray
        Latitude-by-wavenumber field with shape ``(J, M+1)``.
    Pmnw : jnp.ndarray
        Weighted Legendre basis produced by :func:`weighted_legendre_basis`.
    
    Returns
    -------
    jnp.ndarray
        Spectral coefficients with shape ``(M+1, N+1)``.
    """
    return jnp.einsum("jm,jmn->mn", data, Pmnw)


def fwd_leg_w_batch(data: jnp.ndarray, Pmnw: jnp.ndarray) -> jnp.ndarray:
    """Batched forward Legendre transform with preweighted basis.
    
    Parameters
    ----------
    data : jnp.ndarray
        Batched Fourier fields with shape ``(K, J, M+1)``.
    Pmnw : jnp.ndarray
        Preweighted Legendre basis with shape ``(J, M+1, N+1)``.
    
    Returns
    -------
    jnp.ndarray
    """
    return jnp.einsum("kjm,jmn->kmn", data, Pmnw)


def fwd_leg(
    data: jnp.ndarray,
    J: int,
    M: int,
    N: int,
    Pmn: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    """Forward Legendre transform.
    
    Parameters
    ----------
    data : jnp.ndarray
        Fourier coefficients as a function of latitude with shape ``(J, M+1)``.
    J : int
        Number of Gaussian latitudes.
    M : int
        Maximum retained zonal wavenumber.
    N : int
        Maximum total wavenumber.
    Pmn : jnp.ndarray
        Associated Legendre basis with shape ``(J, M+1, N+1)``.
    w : jnp.ndarray
        Gauss–Legendre quadrature weights with shape ``(J,)``.
    
    Returns
    -------
    jnp.ndarray
    """
    # Reference implementation: out[m,n] = sum_j w[j] * data[j,m] * Pmn[j,m,n]
    return fwd_leg_w(data, weighted_legendre_basis(Pmn, w))


def invrs_leg(
    legcoeff: jnp.ndarray,
    I: int,
    J: int,
    M: int,
    N: int,
    Pmn: jnp.ndarray,
) -> jnp.ndarray:
    """Inverse Legendre transform.
    
    Returns Fourier coefficients approxXim of shape (J, I), with the last M
    columns containing the negative-m modes (-M..-1).
    
    Parameters
    ----------
    legcoeff : jnp.ndarray
    I : int
    J : int
    M : int
    N : int
    Pmn : jnp.ndarray

    Returns
    -------
    jnp.ndarray
        Longitude Fourier coefficients with shape ``(J, I)`` whose final
        ``M`` columns contain the reconstructed negative-``m`` modes.
    """
    I = int(I)
    J = int(J)
    M = int(M)

    # Positive m (0..M)
    pos = jnp.einsum("jmn,mn->jm", Pmn, legcoeff)  # (J, M+1)

    approxXim = jnp.zeros((J, I), dtype=legcoeff.dtype)
    approxXim = approxXim.at[:, : (M + 1)].set(pos)

    # Negative m (-M..-1) from conjugate symmetry, matching reference layout.
    if M > 0:
        neg = jnp.einsum("jmn,mn->jm", Pmn[:, 1:, :], jnp.conj(legcoeff[1:, :]))  # (J, M)
        approxXim = approxXim.at[:, I - M : I].set(neg[:, ::-1])

    return approxXim


def invrsUV(
    deltamn: jnp.ndarray,
    etamn: jnp.ndarray,
    fmn: jnp.ndarray,
    I: int,
    J: int,
    M: int,
    N: int,
    Pmn: jnp.ndarray,
    Hmn: jnp.ndarray,
    tstepcoeffmn: jnp.ndarray,
    marray: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute U,V from spectral divergence/vorticity (diagnostic).
    
    Parameters
    ----------
    deltamn : jnp.ndarray
    etamn : jnp.ndarray
    fmn : jnp.ndarray
    I : int
    J : int
    M : int
    N : int
    Pmn : jnp.ndarray
    Hmn : jnp.ndarray
    tstepcoeffmn : jnp.ndarray
    marray : jnp.ndarray
    
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
    """
    deltamn = deltamn.at[:, 0].set(0.0)
    etamn = etamn.at[:, 0].set(0.0)

    eta_minus_f = etamn - fmn
    delta_scaled = deltamn * tstepcoeffmn
    eta_scaled = eta_minus_f * tstepcoeffmn

    newUm1 = invrs_leg(1j * (marray * delta_scaled), I, J, M, N, Pmn)
    newUm2 = invrs_leg(eta_scaled, I, J, M, N, Hmn)

    newVm1 = invrs_leg(1j * (marray * eta_scaled), I, J, M, N, Pmn)
    newVm2 = invrs_leg(delta_scaled, I, J, M, N, Hmn)

    Unew = -invrs_fft(newUm1 - newUm2, I)
    Vnew = -invrs_fft(newVm1 + newVm2, I)
    return Unew, Vnew


def invrsUV_with_coeffs(
    deltamn: jnp.ndarray,
    etamn: jnp.ndarray,
    fmn: jnp.ndarray,
    I: int,
    J: int,
    M: int,
    N: int,
    Pmn: jnp.ndarray,
    Hmn: jnp.ndarray,
    tstepcoeffmn: jnp.ndarray,
    marray: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute U,V from spectral divergence/vorticity and also return Fourier coeffs.

    This is a drop-in variant of :func:`invrsUV` that avoids a redundant
    physical→spectral FFT in the caller by returning the positive-m Fourier
    coefficients of U and V (shape ``(J, M+1)``). These coefficients are
    already available immediately before the inverse FFT to physical space.

    Parameters
    ----------
    deltamn : jnp.ndarray
    etamn : jnp.ndarray
    fmn : jnp.ndarray
    I : int
    J : int
    M : int
    N : int
    Pmn : jnp.ndarray
    Hmn : jnp.ndarray
    tstepcoeffmn : jnp.ndarray
    marray : jnp.ndarray

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Tuple ``(Unew, Vnew, Um_trunc, Vm_trunc)`` containing physical-space
        winds with shape ``(J, I)`` and truncated positive-``m`` Fourier
        coefficients with shape ``(J, M+1)``.
    """

    deltamn = deltamn.at[:, 0].set(0.0)
    etamn = etamn.at[:, 0].set(0.0)

    eta_minus_f = etamn - fmn
    delta_scaled = deltamn * tstepcoeffmn
    eta_scaled = eta_minus_f * tstepcoeffmn

    newUm1 = invrs_leg(1j * (marray * delta_scaled), I, J, M, N, Pmn)
    newUm2 = invrs_leg(eta_scaled, I, J, M, N, Hmn)
    Um_full = -(newUm1 - newUm2)
    Unew = invrs_fft(Um_full, I)
    Um_trunc = Um_full[:, : (M + 1)]

    newVm1 = invrs_leg(1j * (marray * eta_scaled), I, J, M, N, Pmn)
    newVm2 = invrs_leg(delta_scaled, I, J, M, N, Hmn)
    Vm_full = -(newVm1 + newVm2)
    Vnew = invrs_fft(Vm_full, I)
    Vm_trunc = Vm_full[:, : (M + 1)]

    return Unew, Vnew, Um_trunc, Vm_trunc


def diagnostic_eta_delta(
    Um: jnp.ndarray,
    Vm: jnp.ndarray,
    fmn: jnp.ndarray,
    I: int,
    J: int,
    M: int,
    N: int,
    Pmn: jnp.ndarray,
    Hmn: jnp.ndarray,
    w: jnp.ndarray,
    tstepcoeff: jnp.ndarray,
    mJarray: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (eta, delta) from Fourier wind coefficients (diagnostic).
    
    Parameters
    ----------
    Um : jnp.ndarray
    Vm : jnp.ndarray
    fmn : jnp.ndarray
    I : int
    J : int
    M : int
    N : int
    Pmn : jnp.ndarray
    Hmn : jnp.ndarray
    w : jnp.ndarray
    tstepcoeff : jnp.ndarray
    mJarray : jnp.ndarray
    dt : float
    
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    """
    dt_j = jnp.asarray(dt, dtype=float_dtype())
    coeff = tstepcoeff / (2.0 * dt_j)

    zetamn = fwd_leg(coeff * (1j) * mJarray * Vm, J, M, N, Pmn, w) + fwd_leg(coeff * Um, J, M, N, Hmn, w)
    etamn = zetamn + fmn

    deltamn = fwd_leg(coeff * (1j) * mJarray * Um, J, M, N, Pmn, w) - fwd_leg(coeff * Vm, J, M, N, Hmn, w)

    newdeltam = invrs_leg(deltamn, I, J, M, N, Pmn)
    newdelta = invrs_fft(newdeltam, I)

    newetam = invrs_leg(etamn, I, J, M, N, Pmn)
    neweta = invrs_fft(newetam, I)

    return neweta, newdelta, etamn, deltamn
