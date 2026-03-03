# -*- coding: utf-8 -*-
# ruff: noqa: E741
"""my_swamp.explicit_tdiff

Explicit (leapfrog-style) time differencing following Hack and Jakob (1992).

This module is written to reproduce the reference NumPy SWAMPE implementation
as closely as possible, including historical quirks in the explicit branch.
"""

from __future__ import annotations

import jax.numpy as jnp

from .branching import maybe_apply
from . import filters
from . import spectral_transform as st

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
    Pmnw,
    Hmnw,
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

    # Components 1/2/4 share Pmn basis; evaluate them in one batched contraction.
    Phicomp2prep = tstepcoeff1 * (1j) * mJarray * Cm
    Phicomp3prep = tstepcoeff1 * Dm
    Phicomp1, Phicomp2, delta_leg = st.fwd_leg_w_batch(jnp.stack((Phim0, Phicomp2prep, deltam1), axis=0), Pmnw)
    Phicomp3 = st.fwd_leg_w(Phicomp3prep, Hmnw)

    # Component 4
    Phicomp4 = 2.0 * dt * Phibar * delta_leg

    Phimntstep = Phicomp1 - Phicomp2 + Phicomp3 - Phicomp4

    def _add_forcing(x):
        Phiforcing = st.fwd_leg_w(2.0 * dt * PhiFm, Pmnw)
        return x + Phiforcing

    Phimntstep = maybe_apply(forcflag, _add_forcing, Phimntstep)
    Phimntstep = maybe_apply(diffflag, lambda x: filters.diffusion(x, sigmaPhi), Phimntstep)

    newPhimtstep = st.invrs_leg(Phimntstep, I, J, M, N, Pmn)
    newPhim_trunc = newPhimtstep[:, : (M + 1)]
    newPhitstep = st.invrs_fft(newPhimtstep, I)

    return Phimntstep, newPhitstep, newPhim_trunc


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
    Pmnw,
    Hmnw,
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

    Note: Reference SWAMPE explicit behavior updates delta with carry-over only.
    The dropped tendency components are intentionally omitted here.
    """

    # Component 1 (carry-over)
    deltacomp1 = st.fwd_leg_w(deltam0, Pmnw)

    # Reference behavior (historical quirk): use ONLY deltacomp1.
    deltamntstep = deltacomp1

    def _add_forcing(x):
        # The reference explicit scheme includes additional terms proportional
        # to U/taudrag and V/taudrag *in addition* to Fm/Gm (which already
        # include Rayleigh drag via forcing.Rfun). This is preserved for parity.
        deltaf1prep = (tstepcoeff1 * (1j) * mJarray * Um) / taudrag
        deltaf2prep = (tstepcoeff1 * Vm) / taudrag
        deltaf3prep = tstepcoeff1 * (1j) * mJarray * Fm
        deltaf4prep = tstepcoeff1 * Gm
        deltaf1, deltaf3 = st.fwd_leg_w_batch(jnp.stack((deltaf1prep, deltaf3prep), axis=0), Pmnw)
        deltaf2, deltaf4 = st.fwd_leg_w_batch(jnp.stack((deltaf2prep, deltaf4prep), axis=0), Hmnw)

        deltaforcing = -deltaf1 + deltaf2 + deltaf3 - deltaf4
        return x + deltaforcing

    deltamntstep = maybe_apply(forcflag, _add_forcing, deltamntstep)
    deltamntstep = maybe_apply(diffflag, lambda x: filters.diffusion(x, sigma), deltamntstep)

    newdeltamtstep = st.invrs_leg(deltamntstep, I, J, M, N, Pmn)
    newdeltam_trunc = newdeltamtstep[:, : (M + 1)]
    newdeltatstep = st.invrs_fft(newdeltamtstep, I)

    return deltamntstep, newdeltatstep, newdeltam_trunc


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
    Pmnw,
    Hmnw,
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

    etacomp2prep = tstepcoeff1 * (1j) * mJarray * Am
    etacomp3prep = tstepcoeff1 * Bm
    etacomp1, etacomp2 = st.fwd_leg_w_batch(jnp.stack((etam0, etacomp2prep), axis=0), Pmnw)
    etacomp3 = st.fwd_leg_w(etacomp3prep, Hmnw)

    etamntstep = etacomp1 - etacomp2 + etacomp3

    def _add_forcing(x):
        etaf1prep = (tstepcoeff1 * (1j) * mJarray * Vm) / taudrag
        etaf2prep = (tstepcoeff1 * Um) / taudrag
        etaf3prep = tstepcoeff1 * (1j) * mJarray * Gm
        etaf4prep = tstepcoeff1 * Fm
        etaf1, etaf3 = st.fwd_leg_w_batch(jnp.stack((etaf1prep, etaf3prep), axis=0), Pmnw)
        etaf2, etaf4 = st.fwd_leg_w_batch(jnp.stack((etaf2prep, etaf4prep), axis=0), Hmnw)

        etaforcing = -etaf1 + etaf2 + etaf3 + etaf4
        return x + etaforcing

    etamntstep = maybe_apply(forcflag, _add_forcing, etamntstep)
    etamntstep = maybe_apply(diffflag, lambda x: filters.diffusion(x, sigma), etamntstep)

    newetamtstep = st.invrs_leg(etamntstep, I, J, M, N, Pmn)
    newetam_trunc = newetamtstep[:, : (M + 1)]
    newetatstep = st.invrs_fft(newetamtstep, I)

    return etamntstep, newetatstep, newetam_trunc
