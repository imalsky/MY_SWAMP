from __future__ import annotations

import jax
import jax.numpy as jnp
from .dtypes import float_dtype

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

    The explicit and modified-Euler implementations are written to reproduce the
    reference NumPy SWAMPE behavior as closely as possible.

    Scheme selection uses `jax.lax.cond` so `expflag` can be a traced JAX boolean.
    """

    def do_explicit(_: object):
        newPhimn, newPhitstep, newPhim = exp_tdiff.phi_timestep(
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

        newdeltamn, newdeltatstep, newdeltam = exp_tdiff.delta_timestep(
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

        newetamn, newetatstep, newetam = exp_tdiff.eta_timestep(
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

        Unew, Vnew, newUm, newVm = st.invrsUV_with_coeffs(newdeltamn, newetamn, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray)

        return newetamn, newetatstep, newetam, newdeltamn, newdeltatstep, newdeltam, newPhimn, newPhitstep, newPhim, Unew, Vnew, newUm, newVm

    def do_modeuler(_: object):
        newPhimn, newPhitstep, newPhim = mod_tdiff.phi_timestep(
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

        newdeltamn, newdeltatstep, newdeltam = mod_tdiff.delta_timestep(
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

        newetamn, newetatstep, newetam = mod_tdiff.eta_timestep(
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

        Unew, Vnew, newUm, newVm = st.invrsUV_with_coeffs(newdeltamn, newetamn, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray)

        return newetamn, newetatstep, newetam, newdeltamn, newdeltatstep, newdeltam, newPhimn, newPhitstep, newPhim, Unew, Vnew, newUm, newVm

    return jax.lax.cond(jnp.asarray(expflag), do_explicit, do_modeuler, operand=None)


def tstepcoeffmn(M: int, N: int, a: float) -> jnp.ndarray:
    n = jnp.arange(N + 1, dtype=float_dtype())
    coeff = n * (n + 1)
    coeff = coeff.at[0].set(1.0)
    tstep = a / coeff
    tstep = tstep.at[0].set(0.0)
    return jnp.tile(tstep[None, :], (M + 1, 1))


def tstepcoeff2(J: int, M: int, dt: float, a: float) -> jnp.ndarray:
    return jnp.ones((J, M + 1), dtype=float_dtype()) * (2.0 * dt / (a**2))


def narray(M: int, N: int) -> jnp.ndarray:
    n = jnp.arange(N + 1, dtype=float_dtype())
    nnp1 = n * (n + 1)
    return jnp.tile(nnp1[None, :], (M + 1, 1))


def tstepcoeff(J: int, M: int, dt: float, mus: jnp.ndarray, a: float) -> jnp.ndarray:
    mu = mus[:, None]
    # Match NumPy SWAMPE: Gauss–Legendre `mus` are strictly in (-1, 1), so
    # no division-by-zero guard is applied.
    base = (2.0 * dt) / (a * (1.0 - mu**2))  # (J,1)
    return jnp.tile(base, (1, M + 1))


def mJarray(J: int, M: int) -> jnp.ndarray:
    m = jnp.arange(M + 1, dtype=float_dtype())[None, :]
    return jnp.tile(m, (J, 1))


def marray(M: int, N: int) -> jnp.ndarray:
    m = jnp.arange(M + 1, dtype=float_dtype())[:, None]
    return jnp.tile(m, (1, N + 1))


def RMS_winds(a: float, I: int, J: int, lambdas: jnp.ndarray, mus: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Compute RMS winds.

    This matches the reference SWAMPE discretization (vectorized):
        area_comp = a^2 * (sin(phi + pi/2))^2 * dphi * dlambda
        integrand = (U/cos(phi))^2 + (V/cos(phi))^2
        rms = sqrt( sum(area_comp * integrand / area_planet) )
    """
    phis = jnp.arcsin(mus)[:, None]  # (J,1)
    deltalambda = lambdas[2] - lambdas[1]
    deltaphi = phis[2, 0] - phis[1, 0]
    area_planet = 4.0 * jnp.pi * a**2

    area_comp = (a**2) * (jnp.sin(phis + jnp.pi / 2.0) ** 2) * deltaphi * deltalambda  # (J,1)
    integrand = (U / jnp.cos(phis)) ** 2 + (V / jnp.cos(phis)) ** 2  # (J,I)

    return jnp.sqrt(jnp.sum(area_comp * integrand / area_planet))
