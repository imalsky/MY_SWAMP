# -*- coding: utf-8 -*-
# ruff: noqa: E741
"""my_swamp.modEuler_tdiff

Modified-Euler time differencing following Hack and Jakob (1992).

This module is written to reproduce the reference NumPy SWAMPE implementation
as closely as possible, including historical coefficient quirks in that code.

In the reference SWAMPE implementation:
  * Phi and delta updates effectively use tstepcoeff/4 and tstepcoeff2/4 due to
    a double-halving conversion from the leapfrog (2*dt) coefficient.
  * eta uses the unscaled tstepcoeff when forcflag=True, and tstepcoeff/2 when
    forcflag=False.
  * delta uses (Bm+Fm) and (Am-Gm) even when forcflag=False (a historical quirk).

The JAX port below implements these behaviors directly but remains fully
vectorized and differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp

from .branching import maybe_apply, select
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
    """Modified-Euler update for geopotential Phi (reference SWAMPE parity)."""

    # Reference SWAMPE quirk: effective conversion is /4.
    tstep1 = tstepcoeff1 / 4.0
    tstep2 = tstepcoeff2 / 4.0

    # Use forced/non-forced A,B coupling exactly as in reference.
    B_eff = select(forcflag, Bm + Fm, Bm)
    A_eff = select(forcflag, Am - Gm, Am)

    p_terms = jnp.stack(
        (
            Phim1,
            tstep1 * (1j) * mJarray * Cm,
            deltam1,
            tstep1 * (1j) * mJarray * B_eff,
            tstep2 * Em,
        ),
        axis=0,
    )
    Phicomp1, Phicomp2, delta_leg, deltacomp2, deltacomp5 = st.fwd_leg_w_batch(p_terms, Pmnw)
    Phicomp3, deltacomp3 = st.fwd_leg_w_batch(jnp.stack((tstep1 * Dm, tstep1 * A_eff), axis=0), Hmnw)
    Phicomp4 = dt * Phibar * delta_leg
    deltacomp5 = narray * deltacomp5

    Phimntstep = (
        Phicomp1
        - Phicomp2
        + Phicomp3
        - Phicomp4
        - Phibar
        * 0.5
        * (deltacomp2 + deltacomp3 + deltacomp5 + (1.0 / (a**2)) * (narray * Phicomp1))
    )

    def _add_forcing(x):
        Phiforcing = st.fwd_leg_w(dt * PhiFm, Pmnw)
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
    """Modified-Euler update for divergence delta (reference SWAMPE parity)."""

    # Reference SWAMPE quirk: effective conversion is /4.
    tstep1 = tstepcoeff1 / 4.0
    tstep2 = tstepcoeff2 / 4.0

    # Reference SWAMPE quirk: uses forced A,B terms even when forcflag=False.
    B_force = Bm + Fm
    A_force = Am - Gm

    p_terms = jnp.stack(
        (
            deltam1,
            tstep1 * (1j) * mJarray * B_force,
            tstep2 * Phim1,
            tstep2 * Em,
            tstep1 * (1j) * mJarray * Cm,
        ),
        axis=0,
    )
    deltacomp1, deltacomp2, deltacomp4, deltacomp5, Phicomp2 = st.fwd_leg_w_batch(p_terms, Pmnw)
    deltacomp3, Phicomp3 = st.fwd_leg_w_batch(jnp.stack((tstep1 * A_force, tstep1 * Dm), axis=0), Hmnw)
    deltacomp4 = narray * deltacomp4
    deltacomp5 = narray * deltacomp5

    deltamntstep = (
        deltacomp1
        + deltacomp2
        + deltacomp3
        + deltacomp4
        + deltacomp5
        + (narray * (Phicomp2 + Phicomp3)) / (2.0 * (a**2))
        - Phibar * (narray * deltacomp1) / (a**2)
    )

    def _add_forcing(x):
        # Reference SWAMPE includes dt/2 here.
        Phiforcing = (narray * st.fwd_leg_w((dt / 2.0) * PhiFm, Pmnw)) / (a**2)
        return x + Phiforcing

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
    """Modified-Euler update for absolute vorticity eta (reference SWAMPE parity)."""

    # Reference SWAMPE quirk: forced branch uses unscaled tstepcoeff1; unforced uses /2.
    tstep1 = select(forcflag, tstepcoeff1, tstepcoeff1 / 2.0)
    A_eff = select(forcflag, Am - Gm, Am)
    B_eff = select(forcflag, Bm + Fm, Bm)

    etacomp1, etacomp2 = st.fwd_leg_w_batch(jnp.stack((etam1, tstep1 * (1j) * mJarray * A_eff), axis=0), Pmnw)
    etacomp3 = st.fwd_leg_w(tstep1 * B_eff, Hmnw)

    etamntstep = etacomp1 - etacomp2 + etacomp3
    etamntstep = maybe_apply(diffflag, lambda x: filters.diffusion(x, sigma), etamntstep)

    newetamtstep = st.invrs_leg(etamntstep, I, J, M, N, Pmn)
    newetam_trunc = newetamtstep[:, : (M + 1)]
    newetatstep = st.invrs_fft(newetamtstep, I)

    return etamntstep, newetatstep, newetam_trunc
