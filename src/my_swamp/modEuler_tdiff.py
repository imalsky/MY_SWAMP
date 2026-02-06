# -*- coding: utf-8 -*-
"""my_swamp.modEuler_tdiff

Functions associated with the modified-Euler time-stepping scheme.

The public API (function names + signatures) is intentionally kept compatible with
the historical SWAMPE layout because `time_stepping.tstepping(...)` calls these
functions with many positional arguments.

NOTE (2026-02-06)
-----------------
BUGFIX (mathematics / control flow):
    The previous JAX port contained several copy/paste and control-flow issues:

    1) `phi_timestep` and `delta_timestep` computed a full update using
       `tstepcoeff1/=2` and `tstepcoeff2/=2`, but then *always overwrote* the
       result inside the `forcflag`/`else` blocks after dividing the coefficients
       by 2 *again*. In practice, the first computation was always overwritten, so
       the effective coefficient was consistently "double-halved" (phi/delta used
       an effective tstepcoeff1/4).

       Option B behavior: this JAX version intentionally applies the (2*dt -> dt)
       conversion exactly once (effective tstepcoeff1/2). This is a behavior change
       relative to historical SWAMPE trajectories.

    2) `delta_timestep` used forced expressions `(Bm + Fm)` and `(Am - Gm)` even
       in the `forcflag == False` branch.

       Fix: when `forcflag` is false, use unforced `Am`/`Bm` consistently and do
       not add the Phi forcing contribution.

    3) `eta_timestep` applied the (2*dt -> dt) coefficient conversion only in the
       `forcflag == False` path, making forced vs unforced runs use different
       effective timesteps.

       Fix: apply the conversion unconditionally so both forced/unforced modes
       use the same scheme.

JAX-compatibility (no change in values for static flags):
    Python `if` statements on `forcflag`/`diffflag` are replaced with
    `jax.lax.cond` / `jax.lax.select` so the functions remain traceable/jittable
    when flags are provided as JAX booleans.

USER WARNING:
    These fixes can change model trajectories compared to previous outputs when
    `expflag=False` (modified Euler branch).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import jax
import jax.numpy as jnp

from . import filters
from . import spectral_transform as st

_LOGGER = logging.getLogger(__name__)
_WARNED_MODEULER_FIX = False


def _warn_once(msg: str) -> None:
    global _WARNED_MODEULER_FIX
    if _WARNED_MODEULER_FIX:
        return
    _WARNED_MODEULER_FIX = True
    _LOGGER.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)


def _cond(pred: Any, true_fun: Callable[[Any], Any], false_fun: Callable[[Any], Any], operand: Any) -> Any:
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
    """Modified-Euler update for geopotential Phi."""

    _warn_once(
        "SWAMPE-JAX OPTION B (corrected physics): modEuler_tdiff uses mathematically consistent dt scaling that differs from historical SWAMPE. Historically, phi/delta effectively used tstepcoeff1/4 ('double-halving' due to overwrite structure), and eta used inconsistent scaling (forced eta path used un-halved tstepcoeff1). This version uses tstepcoeff1/2 (and tstepcoeff2/2) uniformly. Expect different trajectories vs SWAMPE for expflag=False; output parity is not expected."
    )

    # Convert the shared coefficients from the leapfrog convention (2*dt) to a single-step (dt).
    tstepcoeff1 = tstepcoeff1 / 2.0
    tstepcoeff2 = tstepcoeff2 / 2.0

    forc_pred = jnp.asarray(forcflag)

    # Forcing enters through the wind forcing terms (Fm, Gm) in the divergence-related pieces.
    A_eff = jax.lax.select(forc_pred, Am - Gm, Am)
    B_eff = jax.lax.select(forc_pred, Bm + Fm, Bm)

    # Main Phi terms
    Phicomp1 = st.fwd_leg(Phim1, J, M, N, Pmn, w)
    Phicomp2 = st.fwd_leg(tstepcoeff1 * (1j) * mJarray * Cm, J, M, N, Pmn, w)
    Phicomp3 = st.fwd_leg(tstepcoeff1 * Dm, J, M, N, Hmn, w)
    Phicomp4 = dt * Phibar * st.fwd_leg(deltam1, J, M, N, Pmn, w)

    # Divergence-related coupling pieces
    deltacomp2 = st.fwd_leg(tstepcoeff1 * (1j) * mJarray * B_eff, J, M, N, Pmn, w)
    deltacomp3 = st.fwd_leg(tstepcoeff1 * A_eff, J, M, N, Hmn, w)
    deltacomp5 = st.fwd_leg(tstepcoeff2 * Em, J, M, N, Pmn, w)
    deltacomp5 = narray * deltacomp5

    Phimntstep = (
        Phicomp1
        - Phicomp2
        + Phicomp3
        - Phicomp4
        - Phibar
        * 0.5
        * (deltacomp2 + deltacomp3 + deltacomp5 + (1.0 / (a**2)) * jnp.multiply(narray, Phicomp1))
    )

    def add_phi_forcing(x):
        Phiforcing = st.fwd_leg(dt * PhiFm, J, M, N, Pmn, w)
        return x + Phiforcing

    Phimntstep = _cond(forc_pred, add_phi_forcing, lambda x: x, Phimntstep)
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
    """Modified-Euler update for divergence delta."""

    _warn_once(
        "SWAMPE-JAX OPTION B (corrected physics): modEuler_tdiff uses mathematically consistent dt scaling that differs from historical SWAMPE. Historically, phi/delta effectively used tstepcoeff1/4 ('double-halving' due to overwrite structure), and eta used inconsistent scaling (forced eta path used un-halved tstepcoeff1). This version uses tstepcoeff1/2 (and tstepcoeff2/2) uniformly. Expect different trajectories vs SWAMPE for expflag=False; output parity is not expected."
    )

    tstepcoeff1 = tstepcoeff1 / 2.0
    tstepcoeff2 = tstepcoeff2 / 2.0

    forc_pred = jnp.asarray(forcflag)

    A_eff = jax.lax.select(forc_pred, Am - Gm, Am)
    B_eff = jax.lax.select(forc_pred, Bm + Fm, Bm)

    deltacomp1 = st.fwd_leg(deltam1, J, M, N, Pmn, w)

    deltacomp2 = st.fwd_leg(tstepcoeff1 * (1j) * mJarray * B_eff, J, M, N, Pmn, w)
    deltacomp3 = st.fwd_leg(tstepcoeff1 * A_eff, J, M, N, Hmn, w)

    deltacomp4 = st.fwd_leg(tstepcoeff2 * Phim1, J, M, N, Pmn, w)
    deltacomp4 = narray * deltacomp4

    deltacomp5 = st.fwd_leg(tstepcoeff2 * Em, J, M, N, Pmn, w)
    deltacomp5 = narray * deltacomp5

    Phicomp2 = st.fwd_leg(tstepcoeff1 * (1j) * mJarray * Cm, J, M, N, Pmn, w)
    Phicomp3 = st.fwd_leg(tstepcoeff1 * Dm, J, M, N, Hmn, w)

    deltamntstep = (
        deltacomp1
        + deltacomp2
        + deltacomp3
        + deltacomp4
        + deltacomp5
        + jnp.multiply(narray, (Phicomp2 + Phicomp3) / 2.0) / (a**2)
        - Phibar * jnp.multiply(narray, deltacomp1) / (a**2)
    )

    def add_phi_forcing(x):
        # Retain the historical dt/2 factor here: this term represents the Phi-forcing contribution
        # appearing through the (Phi^{n+1}+Phi^{n})/2 average used in the divergence update.
        Phiforcing = jnp.multiply(narray, st.fwd_leg((dt / 2.0) * PhiFm, J, M, N, Pmn, w)) / (a**2)
        return x + Phiforcing

    deltamntstep = _cond(forc_pred, add_phi_forcing, lambda x: x, deltamntstep)
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
    """Modified-Euler update for absolute vorticity eta."""

    _warn_once(
        "SWAMPE-JAX OPTION B (corrected physics): modEuler_tdiff uses mathematically consistent dt scaling that differs from historical SWAMPE. Historically, phi/delta effectively used tstepcoeff1/4 ('double-halving' due to overwrite structure), and eta used inconsistent scaling (forced eta path used un-halved tstepcoeff1). This version uses tstepcoeff1/2 (and tstepcoeff2/2) uniformly. Expect different trajectories vs SWAMPE for expflag=False; output parity is not expected."
    )

    # Consistent (2*dt -> dt) conversion regardless of forcflag.
    tstepcoeff1 = tstepcoeff1 / 2.0

    forc_pred = jnp.asarray(forcflag)

    A_eff = jax.lax.select(forc_pred, Am - Gm, Am)
    B_eff = jax.lax.select(forc_pred, Bm + Fm, Bm)

    etacomp1 = st.fwd_leg(etam1, J, M, N, Pmn, w)
    etacomp2 = st.fwd_leg(tstepcoeff1 * (1j) * mJarray * A_eff, J, M, N, Pmn, w)
    etacomp3 = st.fwd_leg(tstepcoeff1 * B_eff, J, M, N, Hmn, w)

    etamntstep = etacomp1 - etacomp2 + etacomp3

    etamntstep = _cond(diffflag, lambda x: filters.diffusion(x, sigma), lambda x: x, etamntstep)

    newetamtstep = st.invrs_leg(etamntstep, I, J, M, N, Pmn)
    newetatstep = st.invrs_fft(newetamtstep, I)

    return etamntstep, newetatstep