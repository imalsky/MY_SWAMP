from __future__ import annotations

import jax
import jax.numpy as jnp

from . import explicit_tdiff as exp_tdiff
from . import modEuler_tdiff as mod_tdiff
from . import spectral_transform as st


def tstepping(
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
    fmn,
    Pmn,
    Hmn,
    w,
    tstepcoeff,
    tstepcoeff2,
    tstepcoeffmn,
    marray,
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
    expflag,
    sigma,
    sigmaPhi,
    test,
    t,
):
    """Top-level time stepping wrapper.

    Returns updated spectral coefficients and physical-space fields for eta, delta,
    Phi and winds.

    NOTE (2026-02-06)
    -----------------
    JAX-compatibility: scheme selection (explicit vs modified Euler) is now done via
    `jax.lax.cond` instead of a Python `if`, so `expflag` can be a traced JAX boolean.
    Output values for ordinary Python bool flags are unchanged.
    """

    def run_scheme(tdiff):
        newPhimn, newPhitstep = tdiff.phi_timestep(
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
            tstepcoeff,
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
        )

        newdeltamn, newdeltatstep = tdiff.delta_timestep(
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
            tstepcoeff,
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
        )

        newetamn, newetatstep = tdiff.eta_timestep(
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
            tstepcoeff,
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
        )

        Unew, Vnew = st.invrsUV(newdeltamn, newetamn, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray)

        return newetamn, newetatstep, newdeltamn, newdeltatstep, newPhimn, newPhitstep, Unew, Vnew

    def do_explicit(_: object):
        return run_scheme(exp_tdiff)

    def do_modeuler(_: object):
        return run_scheme(mod_tdiff)

    return jax.lax.cond(jnp.asarray(expflag), do_explicit, do_modeuler, operand=None)


def tstepcoeffmn(M: int, N: int, a: float) -> jnp.ndarray:
    n = jnp.arange(N + 1, dtype=jnp.float64)
    coeff = n * (n + 1)
    coeff = coeff.at[0].set(1.0)
    tstep = a / coeff
    tstep = tstep.at[0].set(0.0)
    return jnp.tile(tstep[None, :], (M + 1, 1))


def tstepcoeff2(J: int, M: int, dt: float, a: float) -> jnp.ndarray:
    return jnp.ones((J, M + 1), dtype=jnp.float64) * (2.0 * dt / (a**2))


def narray(M: int, N: int) -> jnp.ndarray:
    n = jnp.arange(N + 1, dtype=jnp.float64)
    nnp1 = n * (n + 1)
    return jnp.tile(nnp1[None, :], (M + 1, 1))


def tstepcoeff(J: int, M: int, dt: float, mus: jnp.ndarray, a: float) -> jnp.ndarray:
    mu = mus[:, None]
    denom = jnp.maximum(1e-30, 1.0 - mu**2)
    base = (2.0 * dt) / (a * denom)  # (J,1)
    return jnp.tile(base, (1, M + 1))


def mJarray(J: int, M: int) -> jnp.ndarray:
    m = jnp.arange(M + 1, dtype=jnp.float64)[None, :]
    return jnp.tile(m, (J, 1))


def marray(M: int, N: int) -> jnp.ndarray:
    m = jnp.arange(M + 1, dtype=jnp.float64)[:, None]
    return jnp.tile(m, (1, N + 1))


def RMS_winds(a: float, I: int, J: int, lambdas: jnp.ndarray, mus: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Compute RMS winds (legacy discretization, vectorized).

    The original implementation cancels the latitude factors analytically, yielding:
        rms = sqrt( (a^2*dphi*dlambda/area_planet) * sum(U^2 + V^2) )
    where dphi and dlambda are inferred from the uniform grids.
    """
    phis = jnp.arcsin(mus)  # (J,)
    deltalambda = lambdas[2] - lambdas[1]
    deltaphi = phis[2] - phis[1]
    area_planet = 4.0 * jnp.pi * a**2
    const = (a**2) * deltaphi * deltalambda / area_planet
    return jnp.sqrt(const * jnp.sum(U**2 + V**2))
