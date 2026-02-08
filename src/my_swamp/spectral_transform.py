# -*- coding: utf-8 -*-
"""my_swamp.spectral_transform

Spectral transform utilities for SWAMPE.

This module matches the reference NumPy/SciPy SWAMPE implementation as closely
as possible:

* Gaussian quadrature nodes/weights are computed via SciPy
  (`scipy.special.roots_legendre`).
* Associated Legendre polynomials and their derivatives are computed via SciPy
  (`scipy.special.lpmn`) and then scaled exactly like the reference code.

The time-critical transforms (FFT and Legendre matrix multiplications) are
implemented with JAX so they can run on GPU and be differentiated.
"""

from __future__ import annotations

from typing import Tuple

import math

import numpy as np

import jax.numpy as jnp

from .dtypes import float_dtype


try:
    import scipy.special as sp
except Exception as exc:  # pragma: no cover
    sp = None
    _SCIPY_IMPORT_ERROR = exc


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
    """
    I = int(I)
    if dtype is None:
        dtype = float_dtype()
    return jnp.linspace(-jnp.pi, jnp.pi, num=I, endpoint=False, dtype=dtype)


def gauss_legendre(J: int, dtype=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Gaussian quadrature nodes/weights (mus, w) for order J.

    The reference SWAMPE uses `scipy.special.roots_legendre(J)`.
    """
    if sp is None:  # pragma: no cover
        raise ImportError(
            "SciPy is required for gauss_legendre (roots_legendre) to match SWAMPE."  # noqa: E501
        ) from _SCIPY_IMPORT_ERROR

    if dtype is None:
        dtype = float_dtype()

    mus_np, w_np = sp.roots_legendre(int(J))
    return jnp.asarray(mus_np, dtype=dtype), jnp.asarray(w_np, dtype=dtype)


def _scaling_table(M: int, N: int) -> np.ndarray:
    """Scaling table matching reference SWAMPE's factorial-based normalization."""
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


def PmnHmn(J: int, M: int, N: int, mus: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute associated Legendre polynomials and derivatives at Gaussian latitudes.

    Returns
    -------
    Pmn, Hmn : jnp.ndarray
        Arrays of shape (J, M+1, N+1) matching reference SWAMPE.
    """
    if sp is None:  # pragma: no cover
        raise ImportError(
            "SciPy is required for PmnHmn (lpmn) to match SWAMPE."  # noqa: E501
        ) from _SCIPY_IMPORT_ERROR

    J = int(J)
    M = int(M)
    N = int(N)

    mus_np = np.asarray(mus, dtype=np.float64)

    Pmntemp = np.zeros((J, M + 1, N + 1), dtype=np.float64)
    Hmntemp = np.zeros((J, M + 1, N + 1), dtype=np.float64)

    for j in range(J):
        # SciPy returns (Pmn, dP/dmu) with shape (M+1, N+1)
        P_j, dP_j = sp.lpmn(M, N, float(mus_np[j]))
        Pmntemp[j, :, :] = P_j
        # Reference SWAMPE uses (1 - mu^2) * dP/dmu
        Hmntemp[j, :, :] = (1.0 - mus_np[j] ** 2) * dP_j

    scale = _scaling_table(M, N)  # (M+1, N+1)
    Pmn = Pmntemp * scale[None, :, :]
    Hmn = Hmntemp * scale[None, :, :]

    # Reference SWAMPE flips the sign for odd m and n>0.
    odd_m = (np.arange(M + 1) % 2) == 1
    Pmn[:, odd_m, 1:] *= -1.0
    Hmn[:, odd_m, 1:] *= -1.0

    return jnp.asarray(Pmn, dtype=float_dtype()), jnp.asarray(Hmn, dtype=float_dtype())


def fwd_fft_trunc(data: jnp.ndarray, I: int, M: int) -> jnp.ndarray:
    """Fourier transform along longitude, truncating to m=0..M."""
    I = int(I)
    M = int(M)
    datahat = jnp.fft.fft(data / I, n=I, axis=1)
    return datahat[:, : (M + 1)]


def invrs_fft(approxXim: jnp.ndarray, I: int) -> jnp.ndarray:
    """Inverse Fourier transform along longitude (expects full I coefficients)."""
    I = int(I)
    return jnp.fft.ifft(I * approxXim, n=I, axis=1)


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
    data : (J, M+1) complex
        Fourier coefficients as a function of latitude.
    Pmn : (J, M+1, N+1) real
        Associated Legendre basis.
    w : (J,) real
        Gauss–Legendre weights.
    """
    # Reference implementation: out[m,n] = sum_j w[j] * data[j,m] * Pmn[j,m,n]
    return jnp.einsum("j,jm,jmn->mn", w, data, Pmn)


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
    """Compute U,V from spectral divergence/vorticity (diagnostic)."""
    deltamn = deltamn.at[:, 0].set(0.0)
    etamn = etamn.at[:, 0].set(0.0)

    newUm1 = invrs_leg(1j * (marray * deltamn) * tstepcoeffmn, I, J, M, N, Pmn)
    newUm2 = invrs_leg((etamn - fmn) * tstepcoeffmn, I, J, M, N, Hmn)

    newVm1 = invrs_leg(1j * (marray * (etamn - fmn)) * tstepcoeffmn, I, J, M, N, Pmn)
    newVm2 = invrs_leg(deltamn * tstepcoeffmn, I, J, M, N, Hmn)

    Unew = -invrs_fft(newUm1 - newUm2, I)
    Vnew = -invrs_fft(newVm1 + newVm2, I)
    return Unew, Vnew


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
    """Compute (eta, delta) from Fourier wind coefficients (diagnostic)."""
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
