# -*- coding: utf-8 -*-
"""my_swamp.explicit_tdiff

Explicit (leapfrog-style) time differencing following Hack and Jakob (1992).

This module is written to reproduce the reference NumPy SWAMPE implementation
as closely as possible, including historical quirks in the explicit branch.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from . import filters
from . import spectral_transform as st


def _cond(pred: Any, true_fun: Callable[[Any], Any], false_fun: Callable[[Any], Any], operand: Any) -> Any:
    """JAX-friendly conditional that also works with Python bools."""
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
    """Explicit update for geopotential Phi (reference SWAMPE parity)."""

    # Component 1 (carry-over)
    Phicomp1 = st.fwd_leg(Phim0, J, M, N, Pmn, w)

    # Component 2
    Phicomp2prep = tstepcoeff1 * (1j) * mJarray * Cm
    Phicomp2 = st.fwd_leg(Phicomp2prep, J, M, N, Pmn, w)

    # Component 3
    Phicomp3prep = tstepcoeff1 * Dm
    Phicomp3 = st.fwd_leg(Phicomp3prep, J, M, N, Hmn, w)

    # Component 4
    Phicomp4 = 2.0 * dt * Phibar * st.fwd_leg(deltam1, J, M, N, Pmn, w)

    Phimntstep = Phicomp1 - Phicomp2 + Phicomp3 - Phicomp4

    def _add_forcing(x):
        Phiforcing = st.fwd_leg(2.0 * dt * PhiFm, J, M, N, Pmn, w)
        return x + Phiforcing

    Phimntstep = _cond(forcflag, _add_forcing, lambda x: x, Phimntstep)
    Phimntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigmaPhi), lambda x: x, Phimntstep)

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
    """Explicit update for divergence delta (reference SWAMPE parity).

    Note: The reference SWAMPE explicit branch computes the additional
    divergence tendency components (deltacomp2/3/4) but then drops them in the
    final assignment, using only the carry-over term.
    """

    # Component 1 (carry-over)
    deltacomp1 = st.fwd_leg(deltam0, J, M, N, Pmn, w)

    # Components 2/3/4 are computed for parity with the reference code.
    deltacomp2prep = tstepcoeff1 * (1j) * mJarray * Bm
    _ = st.fwd_leg(deltacomp2prep, J, M, N, Pmn, w)

    deltacomp3prep = tstepcoeff1 * Am
    _ = st.fwd_leg(deltacomp3prep, J, M, N, Hmn, w)

    deltacomp4prep = tstepcoeff2 * (Phim1 + Em)
    _deltacomp4 = st.fwd_leg(deltacomp4prep, J, M, N, Pmn, w)
    _ = narray * _deltacomp4

    # Reference behavior (historical quirk): use ONLY deltacomp1.
    deltamntstep = deltacomp1

    def _add_forcing(x):
        # The reference explicit scheme includes additional terms proportional
        # to U/taudrag and V/taudrag *in addition* to Fm/Gm (which already
        # include Rayleigh drag via forcing.Rfun). This is preserved for parity.
        deltaf1prep = (tstepcoeff1 * (1j) * mJarray * Um) / taudrag
        deltaf1 = st.fwd_leg(deltaf1prep, J, M, N, Pmn, w)

        deltaf2prep = (tstepcoeff1 * Vm) / taudrag
        deltaf2 = st.fwd_leg(deltaf2prep, J, M, N, Hmn, w)

        deltaf3prep = tstepcoeff1 * (1j) * mJarray * Fm
        deltaf3 = st.fwd_leg(deltaf3prep, J, M, N, Pmn, w)

        deltaf4prep = tstepcoeff1 * Gm
        deltaf4 = st.fwd_leg(deltaf4prep, J, M, N, Hmn, w)

        deltaforcing = -deltaf1 + deltaf2 + deltaf3 - deltaf4
        return x + deltaforcing

    deltamntstep = _cond(forcflag, _add_forcing, lambda x: x, deltamntstep)
    deltamntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigma), lambda x: x, deltamntstep)

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
    """Explicit update for absolute vorticity eta (reference SWAMPE parity)."""

    etacomp1 = st.fwd_leg(etam0, J, M, N, Pmn, w)
    etacomp2prep = tstepcoeff1 * (1j) * mJarray * Am
    etacomp2 = st.fwd_leg(etacomp2prep, J, M, N, Pmn, w)
    etacomp3prep = tstepcoeff1 * Bm
    etacomp3 = st.fwd_leg(etacomp3prep, J, M, N, Hmn, w)

    etamntstep = etacomp1 - etacomp2 + etacomp3

    def _add_forcing(x):
        etaf1prep = (tstepcoeff1 * (1j) * mJarray * Vm) / taudrag
        etaf1 = st.fwd_leg(etaf1prep, J, M, N, Pmn, w)

        etaf2prep = (tstepcoeff1 * Um) / taudrag
        etaf2 = st.fwd_leg(etaf2prep, J, M, N, Hmn, w)

        etaf3prep = tstepcoeff1 * (1j) * mJarray * Gm
        etaf3 = st.fwd_leg(etaf3prep, J, M, N, Pmn, w)

        etaf4prep = tstepcoeff1 * Fm
        etaf4 = st.fwd_leg(etaf4prep, J, M, N, Hmn, w)

        etaforcing = -etaf1 + etaf2 + etaf3 + etaf4
        return x + etaforcing

    etamntstep = _cond(forcflag, _add_forcing, lambda x: x, etamntstep)
    etamntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigma), lambda x: x, etamntstep)

    newetamtstep = st.invrs_leg(etamntstep, I, J, M, N, Pmn)
    newetatstep = st.invrs_fft(newetamtstep, I)

    return etamntstep, newetatstep
