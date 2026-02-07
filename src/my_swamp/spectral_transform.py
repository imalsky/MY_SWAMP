"""
Spectral transform utilities for SWAMPE (JAX port).

This module mirrors the original SWAMPE `spectral_transform.py` API, but the
core operations are implemented in JAX so they are JIT-compilable and
differentiable.

Conventions (matches the reference numpy SWAMPE):
  - Longitudes are on an evenly spaced grid in [-pi, pi).
  - Gaussian latitudes (mu = sin(phi)) and weights use Gauss–Legendre quadrature.
  - Associated Legendre functions are *scaled* by:
        sqrt((2n+1)/2 * (n-m)!/(n+m)!)
  - SciPy's `lpmn` includes the Condon–Shortley phase; the reference SWAMPE
    removes it by flipping odd m. This port matches that behavior.
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional

import hashlib
import math

import numpy as np

import jax

import jax.numpy as jnp
from .dtypes import float_dtype
from jax.scipy.special import gammaln

# -----------------------------------------------------------------------------
# Small caches for static objects (quadrature + basis)
# -----------------------------------------------------------------------------
# NOTE: Pmn/Hmn are uniquely determined by (J,M,N,dtype) for canonical Gaussian
# nodes returned by `gauss_legendre(J)`. If you pass custom `mus`, disable caching.
_GL_CACHE: Dict[Tuple[int, str], Tuple[jnp.ndarray, jnp.ndarray]] = {}
# Include a hash of `mus` in the cache key to avoid returning an incorrect basis
# when users pass custom quadrature nodes.
_PMN_CACHE: Dict[Tuple[int, int, int, str, str], Tuple[jnp.ndarray, jnp.ndarray]] = {}


def _is_tracer(x) -> bool:
    """Return True if `x` is a JAX tracer.

    This module caches precomputed quadrature/basis arrays in module-level dicts
    for performance.

    If the builder functions are called *inside* a `jax.jit` trace (common under
    NumPyro's HMC), the computed values can be tracers. Caching tracers can lead
    to subtle failures (e.g., returning invalid objects outside the trace) and/or
    memory leaks.

    We therefore only cache *concrete* arrays.
    """

    return isinstance(x, jax.core.Tracer)


def build_lambdas(I: int, *, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """Evenly spaced longitudes in [-pi, pi) of length I (legacy convention)."""
    dtype = float_dtype() if dtype is None else dtype
    return jnp.linspace(-jnp.pi, jnp.pi, num=int(I), endpoint=False, dtype=dtype)


def gauss_legendre(J: int, *, dtype: Optional[jnp.dtype] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return Gauss–Legendre nodes and weights on [-1, 1] (J-point quadrature).

    SciPy-free replacement for `scipy.special.roots_legendre(J)`.
    Uses Golub–Welsch (Jacobi matrix eigen-decomposition).
    """
    dtype = float_dtype() if dtype is None else dtype
    J = int(J)
    key = (J, jnp.dtype(dtype).name)
    cached = _GL_CACHE.get(key)
    if cached is not None:
        mus, w = cached
        if not (_is_tracer(mus) or _is_tracer(w)):
            return mus, w
        # Defensive: drop any previously cached tracer values.
        _GL_CACHE.pop(key, None)

    # Fast path (recommended): compute nodes/weights on host with NumPy.
    # This avoids many tiny JAX dispatches during initialization, and is also
    # friendlier when the default JAX platform is GPU.
    try:
        mus_np, w_np = np.polynomial.legendre.leggauss(J)
        mus = jnp.asarray(mus_np, dtype=dtype)
        w = jnp.asarray(w_np, dtype=dtype)
    except Exception:
        # Fallback: SciPy-free Golub–Welsch in JAX.
        i = jnp.arange(1, J, dtype=dtype)
        beta = i / jnp.sqrt(4.0 * i * i - 1.0)

        # Dense symmetric Jacobi matrix.
        Jmat = jnp.diag(beta, 1) + jnp.diag(beta, -1)

        eigvals, eigvecs = jnp.linalg.eigh(Jmat)  # ascending
        mus = eigvals.astype(dtype)
        w = (2.0 * (eigvecs[0, :] ** 2)).astype(dtype)

    # Only cache concrete arrays; never cache tracers produced inside a JIT trace.
    if not (_is_tracer(mus) or _is_tracer(w)):
        _GL_CACHE[key] = (mus, w)
    return mus, w


def roots_legendre(J: int, *, dtype: Optional[jnp.dtype] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy-compatible alias for Gauss–Legendre nodes/weights."""
    return gauss_legendre(J, dtype=dtype)


def build_mus(J: int, *, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """Legacy helper: return Gauss–Legendre nodes."""
    mus, _ = gauss_legendre(J, dtype=dtype)
    return mus


def build_w(J: int, *, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """Legacy helper: return Gauss–Legendre weights."""
    _, w = gauss_legendre(J, dtype=dtype)
    return w


def _scaling_table(M: int, N: int, *, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """Legacy scaling: sqrt((2n+1)/2 * (n-m)!/(n+m)!)."""
    dtype = float_dtype() if dtype is None else dtype
    m = jnp.arange(M + 1, dtype=dtype)[:, None]
    n = jnp.arange(N + 1, dtype=dtype)[None, :]
    valid = n >= m

    log_ratio = jnp.where(
        valid,
        gammaln(n - m + 1.0) - gammaln(n + m + 1.0),
        -jnp.inf,
    )
    scale = jnp.where(
        valid,
        jnp.sqrt(((2.0 * n + 1.0) / 2.0) * jnp.exp(log_ratio)),
        0.0,
    )
    return scale.astype(dtype)


def _mus_digest(mus_np: np.ndarray) -> str:
    """Stable short hash for cache keys.

    The cache avoids recomputing expensive bases during repeated runs, but we
    must not return a basis computed on different quadrature nodes.
    """

    mus_np = np.ascontiguousarray(mus_np)
    return hashlib.sha1(mus_np.tobytes()).hexdigest()[:16]


def _scaling_table_numpy(M: int, N: int, *, dtype: np.dtype) -> np.ndarray:
    """NumPy version of the SWAMPE scaling table.

    This is used as an initialization fast-path outside of JIT tracing.
    Sizes are small (<= 107x107), so explicit Python loops are acceptable and
    avoid a hard SciPy dependency.
    """

    scale = np.zeros((M + 1, N + 1), dtype=dtype)
    for m in range(M + 1):
        for n in range(m, N + 1):
            # sqrt((2n+1)/2 * (n-m)!/(n+m)!)
            log_ratio = math.lgamma(n - m + 1.0) - math.lgamma(n + m + 1.0)
            scale[m, n] = math.sqrt(((2.0 * n + 1.0) / 2.0) * math.exp(log_ratio))
    return scale


def _PmnHmn_numpy(J: int, M: int, N: int, mus_np: np.ndarray, *, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (Pmn,Hmn) on host with NumPy.

    This mirrors the JAX recurrence exactly, but avoids thousands of tiny JAX
    dispatches during model initialization.
    """

    x = np.asarray(mus_np, dtype=dtype)
    if x.ndim != 1 or x.shape[0] != J:
        raise ValueError(f"mus must have shape ({J},), got {x.shape}")

    # Unscaled P^m_n(x) including CS phase, stored as (m,n,j).
    P = np.zeros((M + 1, N + 1, J), dtype=dtype)
    P[0, 0, :] = 1.0

    # m=0 Legendre polynomials.
    if N >= 1:
        P[0, 1, :] = x
    for n in range(2, N + 1):
        P[0, n, :] = ((2 * n - 1) * x * P[0, n - 1, :] - (n - 1) * P[0, n - 2, :]) / n

    sqrt_1mx2 = np.sqrt(np.maximum(0.0, 1.0 - x * x))

    # m>=1 via standard recurrences.
    for m in range(1, M + 1):
        P[m, m, :] = -(2 * m - 1) * sqrt_1mx2 * P[m - 1, m - 1, :]
        if m < N:
            P[m, m + 1, :] = (2 * m + 1) * x * P[m, m, :]
        for n in range(m + 2, N + 1):
            P[m, n, :] = ((2 * n - 1) * x * P[m, n - 1, :] - (n + m - 1) * P[m, n - 2, :]) / (n - m)

    # H = (1-x^2) dP/dx via identity:
    #   (1-x^2) dP^m_n/dx = (n+m) P^m_{n-1} - n x P^m_n
    H = np.zeros_like(P)
    H[0, 0, :] = 0.0
    for n in range(1, N + 1):
        H[0, n, :] = n * P[0, n - 1, :] - n * x * P[0, n, :]

    for m in range(1, M + 1):
        H[m, m, :] = -m * x * P[m, m, :]
        for n in range(m + 1, N + 1):
            H[m, n, :] = (n + m) * P[m, n - 1, :] - n * x * P[m, n, :]

    # Remove CS phase (match SWAMPE): flip odd m.
    for m in range(1, M + 1, 2):
        P[m, :, :] *= -1.0
        H[m, :, :] *= -1.0

    # Apply SWAMPE scaling.
    scale = _scaling_table_numpy(M, N, dtype=dtype)
    P *= scale[:, :, None]
    H *= scale[:, :, None]

    Pmn = np.transpose(P, (2, 0, 1))
    Hmn = np.transpose(H, (2, 0, 1))
    return Pmn, Hmn


def PmnHmn(
    J: int,
    M: int,
    N: int,
    mus: jnp.ndarray,
    *,
    dtype: Optional[jnp.dtype] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute scaled associated Legendre basis Pmn and Hmn.

    Outputs:
      Pmn: (J, M+1, N+1)
      Hmn: (J, M+1, N+1) where H = (1 - mu^2) dP/dmu with the same scaling.

    Conventions follow the original SWAMPE code:
      - Start from the standard recurrence that includes Condon–Shortley phase,
        then flip odd m to match SWAMPE (i.e., remove the CS phase).
      - Apply the SWAMPE scaling factor sqrt((2n+1)/2 * (n-m)!/(n+m)!).
    """
    dtype = float_dtype() if dtype is None else dtype
    J = int(J)
    M = int(M)
    N = int(N)

    # If mus is concrete, compute a digest so caching is safe even for custom nodes.
    # If mus is a tracer (called from inside jax.jit), skip caching and use the
    # pure-JAX recurrence.
    mus_is_tracer = _is_tracer(mus)
    digest = ""
    if not mus_is_tracer:
        digest = _mus_digest(np.asarray(mus, dtype=np.float64))
        key = (J, M, N, jnp.dtype(dtype).name, digest)
        cached = _PMN_CACHE.get(key)
        if cached is not None:
            Pmn, Hmn = cached
            if not (_is_tracer(Pmn) or _is_tracer(Hmn)):
                return Pmn, Hmn
            _PMN_CACHE.pop(key, None)

        # Host fast-path: compute basis with NumPy recurrences, then transfer.
        dtype_np = np.dtype(jnp.dtype(dtype).name)
        Pmn_np, Hmn_np = _PmnHmn_numpy(J, M, N, np.asarray(mus), dtype=dtype_np)
        Pmn = jnp.asarray(Pmn_np, dtype=dtype)
        Hmn = jnp.asarray(Hmn_np, dtype=dtype)

        # Only cache concrete arrays; never cache tracers produced inside a JIT trace.
        if not (_is_tracer(Pmn) or _is_tracer(Hmn)):
            _PMN_CACHE[key] = (Pmn, Hmn)
        return Pmn, Hmn

    # Tracer path: stay in JAX (differentiable + JIT traceable).
    x = jnp.asarray(mus, dtype=dtype)
    if x.ndim != 1 or x.shape[0] != J:
        raise ValueError(f"mus must have shape ({J},), got {x.shape}")

    # Unscaled P^m_n(x) including CS phase, stored as (m,n,j).
    P = jnp.zeros((M + 1, N + 1, J), dtype=dtype)
    P = P.at[0, 0, :].set(1.0)

    # m=0 Legendre polynomials.
    if N >= 1:
        P = P.at[0, 1, :].set(x)
    for n in range(2, N + 1):
        P0n = ((2 * n - 1) * x * P[0, n - 1, :] - (n - 1) * P[0, n - 2, :]) / n
        P = P.at[0, n, :].set(P0n)

    sqrt_1mx2 = jnp.sqrt(jnp.maximum(0.0, 1.0 - x * x))

    # m>=1 via standard recurrences.
    for m in range(1, M + 1):
        P_mm = -(2 * m - 1) * sqrt_1mx2 * P[m - 1, m - 1, :]
        P = P.at[m, m, :].set(P_mm)

        if m < N:
            P = P.at[m, m + 1, :].set((2 * m + 1) * x * P_mm)

        for n in range(m + 2, N + 1):
            P_mn = (
                (2 * n - 1) * x * P[m, n - 1, :] - (n + m - 1) * P[m, n - 2, :]
            ) / (n - m)
            P = P.at[m, n, :].set(P_mn)

    # H = (1-x^2) dP/dx computed from the stable identity:
    #   (x^2 - 1) dP^m_n/dx = n x P^m_n - (n+m) P^m_{n-1}
    # so
    #   (1-x^2) dP^m_n/dx = (n+m) P^m_{n-1} - n x P^m_n
    H = jnp.zeros_like(P)

    # m=0: avoid n-1=-1 at n=0
    H = H.at[0, 0, :].set(0.0)
    for n in range(1, N + 1):
        H0n = n * P[0, n - 1, :] - n * x * P[0, n, :]
        H = H.at[0, n, :].set(H0n)

    for m in range(1, M + 1):
        # n=m term (P[m, m-1] is 0.0 by construction)
        H = H.at[m, m, :].set(-m * x * P[m, m, :])
        for n in range(m + 1, N + 1):
            H_mn = (n + m) * P[m, n - 1, :] - n * x * P[m, n, :]
            H = H.at[m, n, :].set(H_mn)

    # Remove CS phase (match SWAMPE): flip odd m.
    for m in range(1, M + 1, 2):
        P = P.at[m, :, :].multiply(-1.0)
        H = H.at[m, :, :].multiply(-1.0)

    # Apply SWAMPE scaling.
    scale = _scaling_table(M, N, dtype=dtype)
    P = (P * scale[:, :, None]).astype(dtype)
    H = (H * scale[:, :, None]).astype(dtype)

    Pmn = jnp.transpose(P, (2, 0, 1))
    Hmn = jnp.transpose(H, (2, 0, 1))

    # Tracer path: never cache, because the output can be a traced value.
    return Pmn, Hmn


def fwd_fft_trunc(data: jnp.ndarray, I: int, M: int) -> jnp.ndarray:
    """Forward FFT in longitude with truncation to m=0..M.

    Args:
        data: (J, I) real/complex field in physical space.
    Returns:
        (J, M+1) complex Fourier coefficients.
    """
    I = int(I)
    M = int(M)
    coeff = jnp.fft.fft(data, axis=1) / float(I)
    return coeff[:, : (M + 1)]


def invrs_fft(approxXim: jnp.ndarray, I: int) -> jnp.ndarray:
    """Inverse FFT in longitude (complex physical field)."""
    I = int(I)
    return jnp.fft.ifft(float(I) * approxXim, axis=1)


def fwd_leg(data: jnp.ndarray, J: int, M: int, N: int, Pmn: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Forward Legendre transform: truncated Fourier coeffs -> (m,n) spectral coefficients.

    Args:
        data: (J, M+1) complex Fourier coefficients.
        w: (J,) Gauss–Legendre weights.
        Pmn: (J, M+1, N+1) basis.
    Returns:
        (M+1, N+1) complex spectral coefficients.
    """
    # legcoeff[m,n] = sum_j w[j] * data[j,m] * Pmn[j,m,n]
    return jnp.einsum("j,jm,jmn->mn", w, data, Pmn)


def invrs_leg(legcoeff: jnp.ndarray, I: int, J: int, M: int, N: int, Pmn: jnp.ndarray) -> jnp.ndarray:
    """Inverse Legendre transform: (m,n) -> full Fourier coeffs (J,I).

    Layout matches SWAMPE:
      - positive modes m=0..M placed in columns 0..M
      - negative modes m=-M..-1 placed in columns I-M .. I-1
    """
    I = int(I)
    J = int(J)
    M = int(M)

    # Positive m (0..M)
    pos = jnp.einsum("jmn,mn->jm", Pmn, legcoeff)

    # Negative m (-M..-1) from conjugate symmetry of real fields.
    neg_m = jnp.einsum("jmn,mn->jm", Pmn[:, 1:, :], jnp.conj(legcoeff[1:, :]))
    neg_rev = neg_m[:, ::-1]  # order columns as -M..-1

    out_dtype = jnp.result_type(legcoeff, jnp.complex64)
    approxXim = jnp.zeros((J, I), dtype=out_dtype)
    approxXim = approxXim.at[:, 0 : M + 1].set(pos)
    approxXim = approxXim.at[:, I - M : I].set(neg_rev)
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
    """Compute vorticity and divergence from wind Fourier coefficients (diagnostic).

    Matches SWAMPE's `diagnostic_eta_delta`:
      - Input Um/Vm are Fourier coefficients (J, M+1).
      - Returns (eta, delta) in physical space and (etamn, deltamn) in spectral space.
    """
    I = int(I)
    J = int(J)
    M = int(M)
    N = int(N)

    dt = jnp.asarray(dt, dtype=float_dtype())
    coeff = tstepcoeff / (2.0 * dt)

    zetamn = fwd_leg(coeff * (1j) * mJarray * Vm, J, M, N, Pmn, w) + fwd_leg(coeff * Um, J, M, N, Hmn, w)
    etamn = zetamn + fmn

    deltamn = fwd_leg(coeff * (1j) * mJarray * Um, J, M, N, Pmn, w) - fwd_leg(coeff * Vm, J, M, N, Hmn, w)

    newdeltam = invrs_leg(deltamn, I, J, M, N, Pmn)
    newdelta = invrs_fft(newdeltam, I)

    newetam = invrs_leg(etamn, I, J, M, N, Pmn)
    neweta = invrs_fft(newetam, I)

    return neweta, newdelta, etamn, deltamn
