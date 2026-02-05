"""
Spectral transform module for JAX-based SWAMPE.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as onp
import jax.numpy as jnp


# ---------------------------- BASIS BUILDING ----------------------------

def _scaling_term(n: int, m: int) -> float:
    """Same scaling used in the legacy SciPy-based PmnHmn."""
    if n < m:
        return 0.0
    # sqrt((2n+1)/2 * (n-m)!/(n+m)!)
    # use log-gamma for numerical stability
    lg = math.lgamma
    log_ratio = lg(n - m + 1) - lg(n + m + 1)
    return math.sqrt((2.0 * n + 1.0) / 2.0 * math.exp(log_ratio))


def _double_factorial_odd(n: int) -> int:
    """Compute (2m-1)!! for odd n."""
    out = 1
    k = n
    while k > 1:
        out *= k
        k -= 2
    return out


def compute_legendre_basis(
    mus: jnp.ndarray,
    M: int,
    N: int,
    *,
    dtype=jnp.float64,
    use_scipy: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Pmn and Hmn (associated Legendre polynomials and their H-derivatives).

    Output shapes match the legacy code:
      Pmn: (J, M+1, N+1)
      Hmn: (J, M+1, N+1)

    Notes on end-to-end differentiability:
      - These depend only on the fixed grid (mus) and truncation (M,N).
      - They are typically treated as constants; gradients rarely need to flow through them.
      - The returned arrays are JAX arrays so the rest of the code is fully JAX.

    If use_scipy=True and SciPy is available, uses scipy.special.lpmn to match the legacy results.
    Otherwise uses a standard three-term recurrence (unnormalized) plus an analytic derivative formula.
    """
    mu_np = onp.asarray(mus, dtype=onp.float64)
    J = mu_np.shape[0]

    if use_scipy:
        try:
            import scipy.special as sp
        except ImportError as e:
            raise ImportError(
                "SciPy not available, but use_scipy=True. Install scipy or set use_scipy=False."
            ) from e

        Pmn = onp.zeros((J, M + 1, N + 1), dtype=onp.float64)
        Hmn = onp.zeros((J, M + 1, N + 1), dtype=onp.float64)
        for j in range(J):
            P, dP = sp.lpmn(M, N, float(mu_np[j]))
            Pmn[j, :, :] = P[:, :]
            # H = (1 - mu^2) dP/dmu
            Hmn[j, :, :] = (1.0 - mu_np[j] ** 2) * dP[:, :]

        # match legacy sign convention
        for m in range(1, M + 1, 2):
            Pmn[:, m, :] *= -1.0
            Hmn[:, m, :] *= -1.0

        # scaling per (m,n)
        for m in range(M + 1):
            for n in range(m, N + 1):
                s = _scaling_term(n, m)
                Pmn[:, m, n] *= s
                Hmn[:, m, n] *= s

        return jnp.asarray(Pmn, dtype=dtype), jnp.asarray(Hmn, dtype=dtype)

    # -------- recurrence fallback (no SciPy) --------
    Pmn = onp.zeros((J, M + 1, N + 1), dtype=onp.float64)
    Hmn = onp.zeros((J, M + 1, N + 1), dtype=onp.float64)

    x = mu_np  # (J,)
    one_minus_x2 = onp.maximum(0.0, 1.0 - x * x)

    for m in range(M + 1):
        # P_m^m
        if m == 0:
            Pmm = onp.ones_like(x)
        else:
            Pmm = ((-1) ** m) * _double_factorial_odd(2 * m - 1) * (one_minus_x2 ** (m / 2.0))
        Pmn[:, m, m] = Pmm

        if m < N:
            Pm1m = x * (2 * m + 1) * Pmm
            Pmn[:, m, m + 1] = Pm1m

        # upward recurrence for n >= m+2
        for n in range(m + 2, N + 1):
            Pn1 = Pmn[:, m, n - 1]
            Pn2 = Pmn[:, m, n - 2]
            Pn = ((2 * n - 1) * x * Pn1 - (n + m - 1) * Pn2) / (n - m)
            Pmn[:, m, n] = Pn

        # Hmn from derivative identity:
        # H = (1-x^2) dP = (n+m) P_{n-1}^m - n x P_n^m
        for n in range(m, N + 1):
            if n == m:
                Pnm1 = 0.0
            else:
                Pnm1 = Pmn[:, m, n - 1]
            Pn = Pmn[:, m, n]
            H = (n + m) * Pnm1 - n * x * Pn
            Hmn[:, m, n] = H

    # legacy sign flip for odd m
    for m in range(1, M + 1, 2):
        Pmn[:, m, :] *= -1.0
        Hmn[:, m, :] *= -1.0

    # apply scaling
    for m in range(M + 1):
        for n in range(m, N + 1):
            s = _scaling_term(n, m)
            Pmn[:, m, n] *= s
            Hmn[:, m, n] *= s

    return jnp.asarray(Pmn, dtype=dtype), jnp.asarray(Hmn, dtype=dtype)


# Alias for backward compatibility with original SWAMPE
def PmnHmn(J: int, M: int, N: int, mus: jnp.ndarray):
    """Compatibility wrapper for the original SWAMPE `PmnHmn`.

    The original code uses `scipy.special.lpmn` to build the associated Legendre basis.
    For numerical parity, we use SciPy when available; if not, we fall back to the
    recurrence implementation in `compute_legendre_basis`.

    Parameters
    ----------
    J : int
        Number of latitudes.
    M : int
        Highest wavenumber.
    N : int
        Highest degree of Legendre functions for m=0.
    mus : array (J,)
        Gaussian latitudes (roots of Legendre polynomial), in [-1, 1].

    Returns
    -------
    Pmn : array (J, M+1, N+1)
        Scaled associated Legendre polynomials.
    Hmn : array (J, M+1, N+1)
        Scaled derivatives multiplied by (1 - mus^2) (as in the original).
    """
    use_scipy = False
    try:
        import scipy  # noqa: F401
        use_scipy = True
    except Exception:
        use_scipy = False

    Pmn, Hmn = compute_legendre_basis(mus, M, N, use_scipy=use_scipy)
    return Pmn, Hmn


def fwd_leg(data: jnp.ndarray, J: int, M: int, N: int, Pmn: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Forward Legendre transform: FFT-truncated Fourier coeffs -> spectral (m,n).
    
    Parameters
    ----------
    data : array (J, M+1)
        Fourier coefficients at each latitude
    J : int
        Number of latitudes
    M : int
        Highest wavenumber
    N : int
        Highest degree of Legendre polynomials
    Pmn : array (J, M+1, N+1)
        Associated Legendre polynomials at Gaussian latitudes
    w : array (J,)
        Gauss-Legendre weights
        
    Returns
    -------
    legcoeff : array (M+1, N+1)
        Spectral coefficients
    """
    # Original loop-based implementation:
    # for m in range(0, M+1):
    #     for j in range(0, J):
    #         legterm[j,m,:] = w[j] * data[j,m] * Pmn[j,m,:]
    # legcoeff = np.sum(legterm, 0)
    #
    # Vectorized as einsum:
    return jnp.einsum("j,jm,jmn->mn", w, data, Pmn)


def fwd_fft_trunc(data: jnp.ndarray, I: int, M: int) -> jnp.ndarray:
    """Forward FFT in longitude with truncation to m=0..M.
    
    Parameters
    ----------
    data : array (J, I)
        Physical space data
    I : int
        Number of longitudes
    M : int
        Highest wavenumber for truncation
        
    Returns
    -------
    datam : array (J, M+1)
        Truncated Fourier coefficients
    """
    coeff = jnp.fft.fft(data, axis=1) / float(I)  # (J, I)
    return coeff[:, : (M + 1)]


def invrs_leg(legcoeff: jnp.ndarray, I: int, J: int, M: int, N: int, Pmn: jnp.ndarray) -> jnp.ndarray:
    """Inverse Legendre transform: spectral (m,n) -> full Fourier coeffs (J, I).
    
    Parameters
    ----------
    legcoeff : array (M+1, N+1)
        Spectral coefficients
    I : int
        Number of longitudes
    J : int
        Number of latitudes
    M : int
        Highest wavenumber
    N : int
        Highest degree of Legendre polynomials
    Pmn : array (J, M+1, N+1)
        Associated Legendre polynomials at Gaussian latitudes
        
    Returns
    -------
    approxXim : array (J, I) complex
        Fourier coefficients at each latitude
    """
    # Initialize output arrays
    approxXim = jnp.zeros((J, I), dtype=jnp.complex128)
    approxXimPos = jnp.zeros((J, M + 1), dtype=jnp.complex128)
    approxXimNeg = jnp.zeros((J, M), dtype=jnp.complex128)
    
    # For each m, sum only over n = m to N (triangular constraint)
    # Original code:
    # for m in range(0, M+1):
    #     approxXimPos[:,m] = np.matmul(Pmn[:,m,m:N+1], legcoeff[m,m:N+1])
    #     if m != 0:
    #         negPmn = ((-1)**m) * Pmn[:,m,m:N+1]
    #         negXileg = ((-1)**m) * np.conj(legcoeff[m,m:N+1])
    #         approxXimNeg[:,-m] = np.matmul(negPmn, negXileg)
    
    # Vectorized implementation using a loop for the triangular structure
    # (JAX will trace through this and create efficient code)
    def compute_m_contribution(m, carry):
        approxXimPos, approxXimNeg = carry
        
        # Positive m: sum Pmn[:,m,m:N+1] @ legcoeff[m,m:N+1]
        # Create a mask for n >= m
        n_indices = jnp.arange(N + 1)
        mask = (n_indices >= m).astype(jnp.float64)  # 1 where n >= m, 0 elsewhere
        
        # Apply mask to get triangular slice effect
        Pmn_masked = Pmn[:, m, :] * mask[None, :]  # (J, N+1)
        legcoeff_masked = legcoeff[m, :] * mask    # (N+1,)
        
        pos_contrib = jnp.dot(Pmn_masked, legcoeff_masked)  # (J,)
        approxXimPos = approxXimPos.at[:, m].set(pos_contrib)
        
        # Negative m (for m > 0): use symmetry
        # negPmn = (-1)^m * Pmn, negXileg = (-1)^m * conj(legcoeff)
        # Product: (-1)^(2m) * Pmn * conj(legcoeff) = Pmn * conj(legcoeff)
        def compute_neg(args):
            approxXimNeg, Pmn_masked, legcoeff_masked = args
            neg_contrib = jnp.dot(Pmn_masked, jnp.conj(legcoeff_masked))
            # Index -m in the negative array (which has M elements for m=1..M)
            # -m maps to index M-m in a reversed sense, but we want index m-1 for m=1..M
            return approxXimNeg.at[:, m - 1].set(neg_contrib)
        
        def skip_neg(args):
            approxXimNeg, _, _ = args
            return approxXimNeg
        
        approxXimNeg = jnp.where(
            m > 0,
            compute_neg((approxXimNeg, Pmn_masked, legcoeff_masked)),
            approxXimNeg
        )
        
        return (approxXimPos, approxXimNeg)
    
    # Use a simple Python loop - JAX will trace through it
    for m in range(M + 1):
        # Positive m: sum Pmn[:,m,m:N+1] @ legcoeff[m,m:N+1]
        pos_contrib = jnp.dot(Pmn[:, m, m:N+1], legcoeff[m, m:N+1])
        approxXimPos = approxXimPos.at[:, m].set(pos_contrib)
        
        # Negative m (for m > 0)
        if m > 0:
            # (-1)^m factors cancel: (-1)^m * Pmn * (-1)^m * conj(legcoeff) = Pmn * conj(legcoeff)
            neg_contrib = jnp.dot(Pmn[:, m, m:N+1], jnp.conj(legcoeff[m, m:N+1]))
            # Store at index m-1 (since approxXimNeg has shape (J, M) for m=1..M)
            approxXimNeg = approxXimNeg.at[:, m - 1].set(neg_contrib)
    
    # Assemble full Fourier coefficient array
    # Layout: [m=0, m=1, ..., m=M, zeros..., m=-M, ..., m=-1]
    # Positive part: columns 0 to M
    approxXim = approxXim.at[:, 0:M+1].set(approxXimPos)
    
    # Negative part: columns I-M to I-1 (for m = -M to -1)
    # approxXimNeg[:, 0] corresponds to m=1's negative, should go to position I-1
    # approxXimNeg[:, M-1] corresponds to m=M's negative, should go to position I-M
    # So we need to reverse the order
    approxXim = approxXim.at[:, I-M:I].set(approxXimNeg[:, ::-1])
    
    return approxXim


def invrs_fft(approxXim: jnp.ndarray, I: int) -> jnp.ndarray:
    """Inverse FFT in longitude.
    
    Parameters
    ----------
    approxXim : array (J, I) complex
        Fourier coefficients
    I : int
        Number of longitudes
        
    Returns
    -------
    array (J, I)
        Physical space data
    """
    return jnp.fft.ifft(float(I) * approxXim, axis=1)


# ---------------------------- DIAGNOSTICS ----------------------------

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
    """Compute U, V from spectral divergence/vorticity.
    
    Implements equations (5.24)-(5.25) from Hack and Jakob (1992).
    
    Parameters
    ----------
    deltamn : array (M+1, N+1)
        Spectral coefficients of divergence
    etamn : array (M+1, N+1)
        Spectral coefficients of absolute vorticity
    fmn : array (M+1, N+1)
        Spectral coefficients of Coriolis parameter
    I : int
        Number of longitudes
    J : int
        Number of latitudes
    M : int
        Highest wavenumber
    N : int
        Highest degree of Legendre polynomials
    Pmn : array (J, M+1, N+1)
        Associated Legendre polynomials
    Hmn : array (J, M+1, N+1)
        H-derivatives of Legendre polynomials
    tstepcoeffmn : array (M+1, N+1)
        Time stepping coefficients a/(n(n+1))
    marray : array (M+1, N+1)
        Array of m values
        
    Returns
    -------
    Unew : array (J, I)
        Zonal velocity
    Vnew : array (J, I)
        Meridional velocity
    """
    # Do not sum over n=0 (see Hack and Jakob 1992 equations 5.24-5.25)
    deltamn0 = deltamn.at[:, 0].set(0.0)
    etamn0 = etamn.at[:, 0].set(0.0)

    Um1 = invrs_leg((1j) * (marray * deltamn0) * tstepcoeffmn, I, J, M, N, Pmn)
    Um2 = invrs_leg((etamn0 - fmn) * tstepcoeffmn, I, J, M, N, Hmn)

    Vm1 = invrs_leg((1j) * (marray * (etamn0 - fmn)) * tstepcoeffmn, I, J, M, N, Pmn)
    Vm2 = invrs_leg(deltamn0 * tstepcoeffmn, I, J, M, N, Hmn)

    Unew = -invrs_fft(Um1 - Um2, I)
    Vnew = -invrs_fft(Vm1 + Vm2, I)
    
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
    """Compute eta/delta from winds.
    
    Implements equations (5.26)-(5.27) from Hack and Jakob (1992).
    
    Parameters
    ----------
    Um : array (J, M+1)
        Fourier coefficients of zonal wind
    Vm : array (J, M+1)
        Fourier coefficients of meridional wind
    fmn : array (M+1, N+1)
        Spectral Coriolis parameter
    I : int
        Number of longitudes
    J : int
        Number of latitudes
    M : int
        Highest wavenumber
    N : int
        Highest degree of Legendre polynomials
    Pmn : array (J, M+1, N+1)
        Associated Legendre polynomials
    Hmn : array (J, M+1, N+1)
        H-derivatives
    w : array (J,)
        Gauss-Legendre weights
    tstepcoeff : array (J, M+1)
        Time stepping coefficient 2dt/(a(1-mu^2))
    mJarray : array (J, M+1)
        Array of m values
    dt : float
        Time step
        
    Returns
    -------
    neweta : array (J, I)
        Absolute vorticity in physical space
    newdelta : array (J, I)
        Divergence in physical space
    etamn : array (M+1, N+1)
        Spectral absolute vorticity
    deltamn : array (M+1, N+1)
        Spectral divergence
    """
    coeff = tstepcoeff / (2.0 * dt)

    zetamn = fwd_leg(coeff * (1j) * mJarray * Vm, J, M, N, Pmn, w) + fwd_leg(
        coeff * Um, J, M, N, Hmn, w
    )
    etamn = zetamn + fmn

    deltamn = fwd_leg(coeff * (1j) * mJarray * Um, J, M, N, Pmn, w) - fwd_leg(
        coeff * Vm, J, M, N, Hmn, w
    )

    newdeltam = invrs_leg(deltamn, I, J, M, N, Pmn)
    newdelta = invrs_fft(newdeltam, I)

    newetam = invrs_leg(etamn, I, J, M, N, Pmn)
    neweta = invrs_fft(newetam, I)

    return neweta, newdelta, etamn, deltamn
