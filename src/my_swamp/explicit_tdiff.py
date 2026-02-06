# -*- coding: utf-8 -*-
"""my_swamp.explicit_tdiff

Explicit (leapfrog-style) time differencing following Hack and Jakob (1992).

NOTE (2026-02-06)
-----------------
BUGFIX (mathematics):
    The original JAX port (and the historical numpy SWAMPE it mirrored) computed the
    divergence update using only the carry-over term::

        deltamntstep = fwd_leg(deltam0)

    while silently dropping the advective + pressure-gradient terms
    (deltacomp2/3/4) that were already computed in the function. This is almost
    certainly unintended: it makes the divergence equation effectively decoupled
    from the rest of the dynamics.

    This file now restores the full update that the comments already describe::

        deltamntstep = deltacomp1 + deltacomp2 + deltacomp3 + deltacomp4

    This will change trajectories compared to previous behavior.

JAX-compatibility (no change in values):
    Python `if` statements on `forcflag`/`diffflag` have been replaced with
    `jax.lax.cond` so the functions remain traceable/jittable even if the flags
    are provided as JAX booleans. For ordinary Python bool flags, outputs are
    unchanged.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from . import filters
from . import spectral_transform as st

_LOGGER = logging.getLogger(__name__)

_WARNED_DELTA_FIX = False


def _warn_once(msg: str) -> None:
    """Emit a one-time warning through both logging and `warnings`."""
    global _WARNED_DELTA_FIX
    if _WARNED_DELTA_FIX:
        return
    _WARNED_DELTA_FIX = True
    _LOGGER.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)


def _cond(pred: Any, true_fun: Callable[[Any], Any], false_fun: Callable[[Any], Any], operand: Any) -> Any:
    """Small helper to keep JAX control-flow consistent."""
    return jax.lax.cond(jnp.asarray(pred), true_fun, false_fun, operand)


def phi_timestep(
    etam0,
    etam1,
    deltam0,
    deltam1,
    Phim0,
    Phim1,
    I,
    J,
    M,
    N,
    Am,
    Bm,
    Cm,
    Dm,
    Em,
    Fm,
    Gm,
    Um,
    Vm,
    Pmn,
    Hmn,
    w,
    tstepcoeff1,
    tstepcoeff2,
    mJarray,
    narray,
    PhiFm,
    dt,
    a,
    Phibar,
    taurad,
    taudrag,
    forcflag,
    diffflag,
    sigma,
    sigmaPhi,
    test,
    t,
):
    """Explicit time-stepping for geopotential Phi."""

    # Component 1: Forward Legendre of Phim0
    Phicomp1 = st.fwd_leg(Phim0, J, M, N, Pmn, w)

    # Component 2: tstepcoeff1 * (1j)*mJarray * Cm, then forward Legendre
    Phicomp2prep = tstepcoeff1 * (1j) * mJarray * Cm
    Phicomp2 = st.fwd_leg(Phicomp2prep, J, M, N, Pmn, w)

    # Component 3: tstepcoeff1 * Dm, then forward Legendre with Hmn
    Phicomp3prep = tstepcoeff1 * Dm
    Phicomp3 = st.fwd_leg(Phicomp3prep, J, M, N, Hmn, w)

    # Component 4: 2*dt*Phibar times forward Legendre of deltam1
    Phicomp4 = 2.0 * dt * Phibar * st.fwd_leg(deltam1, J, M, N, Pmn, w)

    Phimntstep = Phicomp1 - Phicomp2 + Phicomp3 - Phicomp4

    def add_forcing(x):
        Phiforcing = st.fwd_leg(2.0 * dt * PhiFm, J, M, N, Pmn, w)
        return x + Phiforcing

    Phimntstep = _cond(forcflag, add_forcing, lambda x: x, Phimntstep)
    Phimntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigmaPhi), lambda x: x, Phimntstep)

    # Transform back to physical space
    newPhimtstep = st.invrs_leg(Phimntstep, I, J, M, N, Pmn)
    newPhitstep = st.invrs_fft(newPhimtstep, I)

    return Phimntstep, newPhitstep


def delta_timestep(
    etam0,
    etam1,
    deltam0,
    deltam1,
    Phim0,
    Phim1,
    I,
    J,
    M,
    N,
    Am,
    Bm,
    Cm,
    Dm,
    Em,
    Fm,
    Gm,
    Um,
    Vm,
    Pmn,
    Hmn,
    w,
    tstepcoeff1,
    tstepcoeff2,
    mJarray,
    narray,
    PhiFm,
    dt,
    a,
    Phibar,
    taurad,
    taudrag,
    forcflag,
    diffflag,
    sigma,
    sigmaPhi,
    test,
    t,
):
    """Explicit time-stepping for divergence delta (BUGFIXED)."""

    _warn_once(
        "SWAMPE-JAX BUGFIX: explicit_tdiff.delta_timestep now uses the full divergence update "
        "deltacomp1+deltacomp2+deltacomp3+deltacomp4 (previously only deltacomp1 was used). "
        "Expect different trajectories vs older outputs."
    )

    # Component 1: Forward Legendre of deltam0 (leapfrog carry-over)
    deltacomp1 = st.fwd_leg(deltam0, J, M, N, Pmn, w)

    # Component 2: tstepcoeff1 * (1j)*mJarray * Bm
    deltacomp2prep = tstepcoeff1 * (1j) * mJarray * Bm
    deltacomp2 = st.fwd_leg(deltacomp2prep, J, M, N, Pmn, w)

    # Component 3: tstepcoeff1 * Am with Hmn
    deltacomp3prep = tstepcoeff1 * Am
    deltacomp3 = st.fwd_leg(deltacomp3prep, J, M, N, Hmn, w)

    # Component 4: tstepcoeff2 * (Phim1 + Em)
    deltacomp4prep = tstepcoeff2 * (Phim1 + Em)
    deltacomp4 = st.fwd_leg(deltacomp4prep, J, M, N, Pmn, w)
    deltacomp4 = narray * deltacomp4

    # BUGFIX: restore the full formula that was already present in comments.
    deltamntstep = deltacomp1 + deltacomp2 + deltacomp3 + deltacomp4

    def add_forcing(x):
        # deltaf1
        deltaf1prep = tstepcoeff1 * (1j) * mJarray * Um / taudrag
        deltaf1 = st.fwd_leg(deltaf1prep, J, M, N, Pmn, w)

        # deltaf2
        deltaf2prep = tstepcoeff1 * Vm / taudrag
        deltaf2 = st.fwd_leg(deltaf2prep, J, M, N, Hmn, w)

        # deltaf3
        deltaf3prep = tstepcoeff1 * (1j) * mJarray * Fm
        deltaf3 = st.fwd_leg(deltaf3prep, J, M, N, Pmn, w)

        # deltaf4
        deltaf4prep = tstepcoeff1 * Gm
        deltaf4 = st.fwd_leg(deltaf4prep, J, M, N, Hmn, w)

        deltaforcing = -deltaf1 + deltaf2 + deltaf3 - deltaf4
        return x + deltaforcing

    deltamntstep = _cond(forcflag, add_forcing, lambda x: x, deltamntstep)
    deltamntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigma), lambda x: x, deltamntstep)

    # Transform back
    newdeltamtstep = st.invrs_leg(deltamntstep, I, J, M, N, Pmn)
    newdeltatstep = st.invrs_fft(newdeltamtstep, I)

    return deltamntstep, newdeltatstep


def eta_timestep(
    etam0,
    etam1,
    deltam0,
    deltam1,
    Phim0,
    Phim1,
    I,
    J,
    M,
    N,
    Am,
    Bm,
    Cm,
    Dm,
    Em,
    Fm,
    Gm,
    Um,
    Vm,
    Pmn,
    Hmn,
    w,
    tstepcoeff1,
    tstepcoeff2,
    mJarray,
    narray,
    PhiFm,
    dt,
    a,
    Phibar,
    taurad,
    taudrag,
    forcflag,
    diffflag,
    sigma,
    sigmaPhi,
    test,
    t,
):
    """Explicit time-stepping for absolute vorticity eta."""

    # Component 1: Forward Legendre of etam0
    etacomp1 = st.fwd_leg(etam0, J, M, N, Pmn, w)

    # Component 2: tstepcoeff1 * (1j)*mJarray * Am
    etacomp2prep = tstepcoeff1 * (1j) * mJarray * Am
    etacomp2 = st.fwd_leg(etacomp2prep, J, M, N, Pmn, w)

    # Component 3: tstepcoeff1 * Bm with Hmn
    etacomp3prep = tstepcoeff1 * Bm
    etacomp3 = st.fwd_leg(etacomp3prep, J, M, N, Hmn, w)

    etamntstep = etacomp1 - etacomp2 + etacomp3

    def add_forcing(x):
        # etaf1
        etaf1prep = tstepcoeff1 * (1j) * mJarray * Vm / taudrag
        etaf1 = st.fwd_leg(etaf1prep, J, M, N, Pmn, w)

        # etaf2
        etaf2prep = tstepcoeff1 * Um / taudrag
        etaf2 = st.fwd_leg(etaf2prep, J, M, N, Hmn, w)

        # etaf3
        etaf3prep = tstepcoeff1 * (1j) * mJarray * Gm
        etaf3 = st.fwd_leg(etaf3prep, J, M, N, Pmn, w)

        # etaf4
        etaf4prep = tstepcoeff1 * Fm
        etaf4 = st.fwd_leg(etaf4prep, J, M, N, Hmn, w)

        etaforcing = -etaf1 + etaf2 + etaf3 + etaf4
        return x + etaforcing

    etamntstep = _cond(forcflag, add_forcing, lambda x: x, etamntstep)
    etamntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigma), lambda x: x, etamntstep)

    # Transform back
    newetamtstep = st.invrs_leg(etamntstep, I, J, M, N, Pmn)
    newetatstep = st.invrs_fft(newetamtstep, I)

    return etamntstep, newetatstep
