# -*- coding: utf-8 -*-
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
    """Modified-Euler update for geopotential Phi (reference SWAMPE parity)."""

    # Reference SWAMPE quirk: effective conversion is /4.
    tstep1 = tstepcoeff1 / 4.0
    tstep2 = tstepcoeff2 / 4.0

    # Use forced/non-forced A,B coupling exactly as in reference.
    B_eff = jax.lax.select(jnp.asarray(forcflag), Bm + Fm, Bm)
    A_eff = jax.lax.select(jnp.asarray(forcflag), Am - Gm, Am)

    Phicomp1 = st.fwd_leg(Phim1, J, M, N, Pmn, w)
    Phicomp2 = st.fwd_leg(tstep1 * (1j) * mJarray * Cm, J, M, N, Pmn, w)
    Phicomp3 = st.fwd_leg(tstep1 * Dm, J, M, N, Hmn, w)
    Phicomp4 = dt * Phibar * st.fwd_leg(deltam1, J, M, N, Pmn, w)

    deltacomp2 = st.fwd_leg(tstep1 * (1j) * mJarray * B_eff, J, M, N, Pmn, w)
    deltacomp3 = st.fwd_leg(tstep1 * A_eff, J, M, N, Hmn, w)

    deltacomp5 = st.fwd_leg(tstep2 * Em, J, M, N, Pmn, w)
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
        Phiforcing = st.fwd_leg(dt * PhiFm, J, M, N, Pmn, w)
        return x + Phiforcing

    Phimntstep = _cond(forcflag, _add_forcing, lambda x: x, Phimntstep)
    Phimntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigmaPhi), lambda x: x, Phimntstep)

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
    """Modified-Euler update for divergence delta (reference SWAMPE parity)."""

    # Reference SWAMPE quirk: effective conversion is /4.
    tstep1 = tstepcoeff1 / 4.0
    tstep2 = tstepcoeff2 / 4.0

    # Reference SWAMPE quirk: uses forced A,B terms even when forcflag=False.
    B_force = Bm + Fm
    A_force = Am - Gm

    deltacomp1 = st.fwd_leg(deltam1, J, M, N, Pmn, w)
    deltacomp2 = st.fwd_leg(tstep1 * (1j) * mJarray * B_force, J, M, N, Pmn, w)
    deltacomp3 = st.fwd_leg(tstep1 * A_force, J, M, N, Hmn, w)

    deltacomp4 = st.fwd_leg(tstep2 * Phim1, J, M, N, Pmn, w)
    deltacomp4 = narray * deltacomp4

    deltacomp5 = st.fwd_leg(tstep2 * Em, J, M, N, Pmn, w)
    deltacomp5 = narray * deltacomp5

    Phicomp2 = st.fwd_leg(tstep1 * (1j) * mJarray * Cm, J, M, N, Pmn, w)
    Phicomp3 = st.fwd_leg(tstep1 * Dm, J, M, N, Hmn, w)

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
        Phiforcing = (narray * st.fwd_leg((dt / 2.0) * PhiFm, J, M, N, Pmn, w)) / (a**2)
        return x + Phiforcing

    deltamntstep = _cond(forcflag, _add_forcing, lambda x: x, deltamntstep)
    deltamntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigma), lambda x: x, deltamntstep)

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
    """Modified-Euler update for absolute vorticity eta (reference SWAMPE parity)."""

    forc_pred = jnp.asarray(forcflag)
    # Reference SWAMPE quirk: forced branch uses unscaled tstepcoeff1; unforced uses /2.
    tstep1 = jax.lax.select(forc_pred, tstepcoeff1, tstepcoeff1 / 2.0)
    A_eff = jax.lax.select(forc_pred, Am - Gm, Am)
    B_eff = jax.lax.select(forc_pred, Bm + Fm, Bm)

    etacomp1 = st.fwd_leg(etam1, J, M, N, Pmn, w)
    etacomp2 = st.fwd_leg(tstep1 * (1j) * mJarray * A_eff, J, M, N, Pmn, w)
    etacomp3 = st.fwd_leg(tstep1 * B_eff, J, M, N, Hmn, w)

    etamntstep = etacomp1 - etacomp2 + etacomp3
    etamntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigma), lambda x: x, etamntstep)

    newetamtstep = st.invrs_leg(etamntstep, I, J, M, N, Pmn)
    newetam_trunc = newetamtstep[:, : (M + 1)]
    newetatstep = st.invrs_fft(newetamtstep, I)

    return etamntstep, newetatstep, newetam_trunc
