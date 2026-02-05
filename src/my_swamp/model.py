"""Differentiable SWAMPE model in JAX.

Primary entry point: run_model_scan(...)
  - Pure JAX computations (no side effects), suitable for jax.grad and jax.jit.
  - Returns time series arrays on the physical (mu, lambda) grid.

A legacy-style wrapper run_model(...) is provided for signature compatibility,
but raises NotImplementedError — see its docstring for details.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from . import continuation
from . import filters
from . import forcing
from . import initial_conditions
from . import spectral_transform as st
from . import time_stepping


# =============================================================================
# Types
# =============================================================================

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Static:
    """Static (time-invariant) model data."""

    I: int
    J: int
    M: int
    N: int

    dt: float
    a: float
    omega: float
    g: float
    Phibar: float

    lambdas: jnp.ndarray
    mus: jnp.ndarray
    w: jnp.ndarray

    Pmn: jnp.ndarray
    Hmn: jnp.ndarray

    fmn: jnp.ndarray
    tstepcoeffmn: jnp.ndarray
    tstepcoeff: jnp.ndarray
    tstepcoeff2: jnp.ndarray
    mJarray: jnp.ndarray
    marray: jnp.ndarray
    narray: jnp.ndarray

    sigma: jnp.ndarray
    sigmaPhi: jnp.ndarray

    Phieq: jnp.ndarray

    def tree_flatten(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        children = (
            self.lambdas, self.mus, self.w,
            self.Pmn, self.Hmn,
            self.fmn,
            self.tstepcoeffmn, self.tstepcoeff, self.tstepcoeff2,
            self.mJarray, self.marray, self.narray,
            self.sigma, self.sigmaPhi,
            self.Phieq,
        )
        aux = dict(
            I=self.I, J=self.J, M=self.M, N=self.N,
            dt=self.dt, a=self.a, omega=self.omega, g=self.g, Phibar=self.Phibar,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: Dict[str, Any], children: Tuple[Any, ...]) -> "Static":
        (
            lambdas, mus, w,
            Pmn, Hmn,
            fmn,
            tstepcoeffmn, tstepcoeff, tstepcoeff2,
            mJarray, marray, narray,
            sigma, sigmaPhi,
            Phieq,
        ) = children
        return cls(
            lambdas=lambdas, mus=mus, w=w,
            Pmn=Pmn, Hmn=Hmn,
            fmn=fmn,
            tstepcoeffmn=tstepcoeffmn, tstepcoeff=tstepcoeff, tstepcoeff2=tstepcoeff2,
            mJarray=mJarray, marray=marray, narray=narray,
            sigma=sigma, sigmaPhi=sigmaPhi,
            Phieq=Phieq,
            **aux,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State:
    """Two-level (prev/curr) spectral+physical state for leapfrog-style stepping."""

    etam_prev: jnp.ndarray
    etam_curr: jnp.ndarray
    deltam_prev: jnp.ndarray
    deltam_curr: jnp.ndarray
    Phim_prev: jnp.ndarray
    Phim_curr: jnp.ndarray

    eta_prev: jnp.ndarray
    delta_prev: jnp.ndarray
    Phi_prev: jnp.ndarray

    eta_curr: jnp.ndarray
    delta_curr: jnp.ndarray
    Phi_curr: jnp.ndarray
    U_curr: jnp.ndarray
    V_curr: jnp.ndarray

    Am_curr: jnp.ndarray
    Bm_curr: jnp.ndarray
    Cm_curr: jnp.ndarray
    Dm_curr: jnp.ndarray
    Em_curr: jnp.ndarray

    PhiFm_curr: jnp.ndarray
    Fm_curr: jnp.ndarray
    Gm_curr: jnp.ndarray

    dead: jnp.ndarray  # scalar bool

    def tree_flatten(self):
        children = (
            self.etam_prev, self.etam_curr,
            self.deltam_prev, self.deltam_curr,
            self.Phim_prev, self.Phim_curr,
            self.eta_prev, self.delta_prev, self.Phi_prev,
            self.eta_curr, self.delta_curr, self.Phi_curr,
            self.U_curr, self.V_curr,
            self.Am_curr, self.Bm_curr, self.Cm_curr, self.Dm_curr, self.Em_curr,
            self.PhiFm_curr, self.Fm_curr, self.Gm_curr,
            self.dead,
        )
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            etam_prev, etam_curr,
            deltam_prev, deltam_curr,
            Phim_prev, Phim_curr,
            eta_prev, delta_prev, Phi_prev,
            eta_curr, delta_curr, Phi_curr,
            U_curr, V_curr,
            Am_curr, Bm_curr, Cm_curr, Dm_curr, Em_curr,
            PhiFm_curr, Fm_curr, Gm_curr,
            dead,
        ) = children
        return cls(
            etam_prev=etam_prev, etam_curr=etam_curr,
            deltam_prev=deltam_prev, deltam_curr=deltam_curr,
            Phim_prev=Phim_prev, Phim_curr=Phim_curr,
            eta_prev=eta_prev, delta_prev=delta_prev, Phi_prev=Phi_prev,
            eta_curr=eta_curr, delta_curr=delta_curr, Phi_curr=Phi_curr,
            U_curr=U_curr, V_curr=V_curr,
            Am_curr=Am_curr, Bm_curr=Bm_curr, Cm_curr=Cm_curr, Dm_curr=Dm_curr, Em_curr=Em_curr,
            PhiFm_curr=PhiFm_curr, Fm_curr=Fm_curr, Gm_curr=Gm_curr,
            dead=dead,
        )


# =============================================================================
# Run flags
# =============================================================================

@dataclass(frozen=True)
class RunFlags:
    forcflag: bool = True
    diffflag: bool = True
    expflag: bool = True
    modalflag: bool = False
    alpha: float = 0.05
    blowup_rms: float = 8000.0


# =============================================================================
# Model building and initialization
# =============================================================================

def build_static(
    *,
    M: int,
    dt: float,
    a: float,
    omega: float,
    g: float,
    Phibar: float,
    K6: float,
    K6Phi: float,
    DPhieq: float,
    test: Optional[int],
    use_scipy_basis: bool = True,
) -> Static:
    """Build all static arrays (grid, basis, coefficients, filters)."""
    N, I, J, _otherdt, lambdas, mus, w = initial_conditions.spectral_params(M)

    if use_scipy_basis:
        try:
            import scipy  # noqa: F401
        except Exception:
            use_scipy_basis = False
    Pmn, Hmn = st.compute_legendre_basis(mus, M, N, use_scipy=use_scipy_basis)

    fmn = initial_conditions.coriolismn(M, omega)

    _tstepcoeffmn = time_stepping.tstepcoeffmn(M, N, a)
    _tstepcoeff = time_stepping.tstepcoeff(J, M, dt, mus, a)
    _tstepcoeff2 = time_stepping.tstepcoeff2(J, M, dt, a)
    _mJarray = time_stepping.mJarray(J, M)
    _marray = time_stepping.marray(M, N)
    _narray = time_stepping.narray(M, N)

    sigma = filters.sigma6(M, N, K6, a, dt)
    sigmaPhi = filters.sigma6Phi(M, N, K6Phi, a, dt)

    if test is None:
        Phieq = forcing.Phieqfun(Phibar, DPhieq, lambdas, mus, I, J, g)
    else:
        Phieq = jnp.zeros((J, I), dtype=jnp.float64)

    return Static(
        I=I, J=J, M=M, N=N,
        dt=float(dt), a=float(a), omega=float(omega), g=float(g), Phibar=float(Phibar),
        lambdas=lambdas, mus=mus, w=w,
        Pmn=Pmn, Hmn=Hmn,
        fmn=fmn,
        tstepcoeffmn=_tstepcoeffmn,
        tstepcoeff=_tstepcoeff,
        tstepcoeff2=_tstepcoeff2,
        mJarray=_mJarray,
        marray=_marray,
        narray=_narray,
        sigma=sigma,
        sigmaPhi=sigmaPhi,
        Phieq=Phieq,
    )


def _init_state(static: Static, *, test: Optional[int], a1: float, taurad: float, taudrag: float) -> State:
    """Initialize (prev,curr) physical fields, winds, nonlinear terms, and forcing."""
    I, J, M, N = static.I, static.J, static.M, static.N

    SU0, sina, cosa, etaamp, Phiamp = initial_conditions.test1_init(static.a, static.omega, a1)

    if test in (1, 2):
        eta0, eta1, delta0, delta1, Phi0, Phi1 = initial_conditions.state_var_init(
            I, J, static.mus, static.lambdas, test, etaamp,
            static.a, sina, cosa, static.Phibar, Phiamp
        )
    else:
        eta0, eta1, delta0, delta1, Phi0, Phi1 = initial_conditions.state_var_init(
            I, J, static.mus, static.lambdas, test, etaamp
        )

    U0, V0 = initial_conditions.velocity_init(I, J, SU0, cosa, sina, static.mus, static.lambdas, test)

    A1, B1, C1, D1, E1 = initial_conditions.ABCDE_init(U0, V0, eta1, Phi1, static.mus, I, J)

    etam0 = st.fwd_fft_trunc(eta0, I, M)
    etam1 = st.fwd_fft_trunc(eta1, I, M)
    deltam0 = st.fwd_fft_trunc(delta0, I, M)
    deltam1 = st.fwd_fft_trunc(delta1, I, M)
    Phim0 = st.fwd_fft_trunc(Phi0, I, M)
    Phim1 = st.fwd_fft_trunc(Phi1, I, M)

    Am1 = st.fwd_fft_trunc(A1, I, M)
    Bm1 = st.fwd_fft_trunc(B1, I, M)
    Cm1 = st.fwd_fft_trunc(C1, I, M)
    Dm1 = st.fwd_fft_trunc(D1, I, M)
    Em1 = st.fwd_fft_trunc(E1, I, M)

    if test is None:
        Q1 = forcing.Qfun(static.Phieq, Phi1, static.Phibar, taurad)
        PhiF1 = Q1
        F1, G1 = forcing.Rfun(U0, V0, Q1, Phi1, static.Phibar, taudrag)
        PhiFm1 = st.fwd_fft_trunc(PhiF1, I, M)
        Fm1 = st.fwd_fft_trunc(F1, I, M)
        Gm1 = st.fwd_fft_trunc(G1, I, M)
    else:
        PhiFm1 = jnp.zeros((J, M + 1), dtype=jnp.complex128)
        Fm1 = jnp.zeros((J, M + 1), dtype=jnp.complex128)
        Gm1 = jnp.zeros((J, M + 1), dtype=jnp.complex128)

    return State(
        etam_prev=etam0, etam_curr=etam1,
        deltam_prev=deltam0, deltam_curr=deltam1,
        Phim_prev=Phim0, Phim_curr=Phim1,
        eta_prev=eta0, delta_prev=delta0, Phi_prev=Phi0,
        eta_curr=eta1, delta_curr=delta1, Phi_curr=Phi1,
        U_curr=U0, V_curr=V0,
        Am_curr=Am1, Bm_curr=Bm1, Cm_curr=Cm1, Dm_curr=Dm1, Em_curr=Em1,
        PhiFm_curr=PhiFm1, Fm_curr=Fm1, Gm_curr=Gm1,
        dead=jnp.array(False),
    )


def _init_state_from_fields(
    static: Static,
    *,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    Phi: jnp.ndarray,
    test: Optional[int],
    taurad: float,
    taudrag: float,
) -> State:
    """Initialize the model state from provided physical fields (restart/continuation)."""
    I, J, M, N = static.I, static.J, static.M, static.N

    eta0 = jnp.asarray(eta, dtype=jnp.float64)
    delta0 = jnp.asarray(delta, dtype=jnp.float64)
    Phi0 = jnp.asarray(Phi, dtype=jnp.float64)

    etam0 = st.fwd_fft_trunc(eta0, I, M)
    deltam0 = st.fwd_fft_trunc(delta0, I, M)
    Phim0 = st.fwd_fft_trunc(Phi0, I, M)

    etamn0 = st.fwd_leg(etam0, J, M, N, static.Pmn, static.w)
    deltamn0 = st.fwd_leg(deltam0, J, M, N, static.Pmn, static.w)

    U0_c, V0_c = st.invrsUV(
        deltamn0, etamn0, static.fmn,
        I, J, M, N,
        static.Pmn, static.Hmn, static.tstepcoeffmn, static.marray,
    )
    U0 = jnp.real(U0_c)
    V0 = jnp.real(V0_c)

    A1, B1, C1, D1, E1 = initial_conditions.ABCDE_init(U0, V0, eta0, Phi0, static.mus, I, J)
    Am1 = st.fwd_fft_trunc(A1, I, M)
    Bm1 = st.fwd_fft_trunc(B1, I, M)
    Cm1 = st.fwd_fft_trunc(C1, I, M)
    Dm1 = st.fwd_fft_trunc(D1, I, M)
    Em1 = st.fwd_fft_trunc(E1, I, M)

    if test is None:
        Q1 = forcing.Qfun(static.Phieq, Phi0, static.Phibar, taurad)
        PhiF1 = Q1
        F1, G1 = forcing.Rfun(U0, V0, Q1, Phi0, static.Phibar, taudrag)
        PhiFm1 = st.fwd_fft_trunc(PhiF1, I, M)
        Fm1 = st.fwd_fft_trunc(F1, I, M)
        Gm1 = st.fwd_fft_trunc(G1, I, M)
    else:
        PhiFm1 = jnp.zeros((J, M + 1), dtype=jnp.complex128)
        Fm1 = jnp.zeros((J, M + 1), dtype=jnp.complex128)
        Gm1 = jnp.zeros((J, M + 1), dtype=jnp.complex128)

    return State(
        etam_prev=etam0, etam_curr=etam0,
        deltam_prev=deltam0, deltam_curr=deltam0,
        Phim_prev=Phim0, Phim_curr=Phim0,
        eta_prev=eta0, delta_prev=delta0, Phi_prev=Phi0,
        eta_curr=eta0, delta_curr=delta0, Phi_curr=Phi0,
        U_curr=U0, V_curr=V0,
        Am_curr=Am1, Bm_curr=Bm1, Cm_curr=Cm1, Dm_curr=Dm1, Em_curr=Em1,
        PhiFm_curr=PhiFm1, Fm_curr=Fm1, Gm_curr=Gm1,
        dead=jnp.array(False),
    )


def _ra_filter(prev: jnp.ndarray, curr: jnp.ndarray, nxt: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Robert-Asselin filter for leapfrog time stepping."""
    return curr + alpha * (prev - 2.0 * curr + nxt)


def _step_once(
    state: State,
    t: jnp.ndarray,
    static: Static,
    flags: RunFlags,
    taurad: float,
    taudrag: float,
    test: Optional[int],
    a1: float,
) -> Tuple[State, Dict[str, jnp.ndarray]]:
    """One step matching the legacy loop index t (starting at 2)."""
    I, J, M, N = static.I, static.J, static.M, static.N

    rms = time_stepping.RMS_winds(static.a, I, J, static.lambdas, static.mus, state.U_curr, state.V_curr)
    spin_min = jnp.min(jnp.sqrt(state.U_curr**2 + state.V_curr**2))
    phi_min = jnp.min(state.Phi_curr)
    phi_max = jnp.max(state.Phi_curr)

    dead_next = jnp.logical_or(state.dead, rms > flags.blowup_rms)

    def skip_update(_):
        out = dict(
            rms=rms, spin_min=spin_min,
            phi_min=phi_min, phi_max=phi_max,
            eta=state.eta_curr, delta=state.delta_curr, Phi=state.Phi_curr,
            U=state.U_curr, V=state.V_curr,
        )
        new_state = State(
            etam_prev=state.etam_prev, etam_curr=state.etam_curr,
            deltam_prev=state.deltam_prev, deltam_curr=state.deltam_curr,
            Phim_prev=state.Phim_prev, Phim_curr=state.Phim_curr,
            eta_prev=state.eta_prev, delta_prev=state.delta_prev, Phi_prev=state.Phi_prev,
            eta_curr=state.eta_curr, delta_curr=state.delta_curr, Phi_curr=state.Phi_curr,
            U_curr=state.U_curr, V_curr=state.V_curr,
            Am_curr=state.Am_curr, Bm_curr=state.Bm_curr, Cm_curr=state.Cm_curr,
            Dm_curr=state.Dm_curr, Em_curr=state.Em_curr,
            PhiFm_curr=state.PhiFm_curr, Fm_curr=state.Fm_curr, Gm_curr=state.Gm_curr,
            dead=dead_next,
        )
        return new_state, out

    def do_update(_):
        Um = st.fwd_fft_trunc(state.U_curr, I, M)
        Vm = st.fwd_fft_trunc(state.V_curr, I, M)

        newetamn, neweta_c, newdeltamn, newdelta_c, newPhimn, newPhi_c, newU_c, newV_c = time_stepping.tstepping(
            state.etam_prev, state.etam_curr,
            state.deltam_prev, state.deltam_curr,
            state.Phim_prev, state.Phim_curr,
            I, J, M, N,
            state.Am_curr, state.Bm_curr, state.Cm_curr, state.Dm_curr, state.Em_curr,
            state.Fm_curr, state.Gm_curr,
            Um, Vm,
            static.fmn, static.Pmn, static.Hmn, static.w,
            static.tstepcoeff, static.tstepcoeff2, static.tstepcoeffmn,
            static.marray, static.mJarray, static.narray,
            state.PhiFm_curr,
            static.dt, static.a, static.Phibar,
            taurad, taudrag,
            flags.forcflag, flags.diffflag, flags.expflag,
            static.sigma, static.sigmaPhi,
            test, t,
        )

        neweta = jnp.real(neweta_c)
        newdelta = jnp.real(newdelta_c)
        newPhi = jnp.real(newPhi_c)
        newU = jnp.real(newU_c)
        newV = jnp.real(newV_c)

        if test == 1:
            SU0, sina, cosa, _etaamp, _Phiamp = initial_conditions.test1_init(static.a, static.omega, a1)
            Uic, Vic = initial_conditions.velocity_init(I, J, SU0, cosa, sina, static.mus, static.lambdas, test)
            newU = Uic
            newV = Vic

        # Update forcing
        if test is None:
            Q2 = forcing.Qfun(static.Phieq, newPhi, static.Phibar, taurad)
            F2, G2 = forcing.Rfun(newU, newV, Q2, newPhi, static.Phibar, taudrag)
            PhiFm2 = st.fwd_fft_trunc(Q2, I, M)
            Fm2 = st.fwd_fft_trunc(F2, I, M)
            Gm2 = st.fwd_fft_trunc(G2, I, M)
        else:
            PhiFm2 = state.PhiFm_curr
            Fm2 = state.Fm_curr
            Gm2 = state.Gm_curr

        # Nonlinear terms
        A2, B2, C2, D2, E2 = initial_conditions.ABCDE_init(newU, newV, neweta, newPhi, static.mus, I, J)
        Am2 = st.fwd_fft_trunc(A2, I, M)
        Bm2 = st.fwd_fft_trunc(B2, I, M)
        Cm2 = st.fwd_fft_trunc(C2, I, M)
        Dm2 = st.fwd_fft_trunc(D2, I, M)
        Em2 = st.fwd_fft_trunc(E2, I, M)

        # Robert-Asselin filter
        do_ra = jnp.logical_and(jnp.array(flags.modalflag), t > 2)

        def apply_ra(_):
            return (
                _ra_filter(state.eta_prev, state.eta_curr, neweta, flags.alpha),
                _ra_filter(state.delta_prev, state.delta_curr, newdelta, flags.alpha),
                _ra_filter(state.Phi_prev, state.Phi_curr, newPhi, flags.alpha),
            )

        def no_ra(_):
            return state.eta_curr, state.delta_curr, state.Phi_curr

        eta_mid, delta_mid, Phi_mid = jax.lax.cond(do_ra, apply_ra, no_ra, operand=None)

        phi_min_out = jnp.min(Phi_mid)
        phi_max_out = jnp.max(Phi_mid)

        new_state = State(
            etam_prev=state.etam_curr,
            etam_curr=st.fwd_fft_trunc(neweta, I, M),
            deltam_prev=state.deltam_curr,
            deltam_curr=st.fwd_fft_trunc(newdelta, I, M),
            Phim_prev=state.Phim_curr,
            Phim_curr=st.fwd_fft_trunc(newPhi, I, M),
            eta_prev=eta_mid, delta_prev=delta_mid, Phi_prev=Phi_mid,
            eta_curr=neweta, delta_curr=newdelta, Phi_curr=newPhi,
            U_curr=newU, V_curr=newV,
            Am_curr=Am2, Bm_curr=Bm2, Cm_curr=Cm2, Dm_curr=Dm2, Em_curr=Em2,
            PhiFm_curr=PhiFm2, Fm_curr=Fm2, Gm_curr=Gm2,
            dead=dead_next,
        )

        out = dict(
            rms=rms, spin_min=spin_min,
            phi_min=phi_min_out, phi_max=phi_max_out,
            eta=neweta, delta=newdelta, Phi=newPhi,
            U=newU, V=newV,
        )
        return new_state, out

    return jax.lax.cond(dead_next, skip_update, do_update, operand=None)


# =============================================================================
# Main entry points
# =============================================================================

def run_model_scan(
    *,
    tmax: int,
    M: int = 42,
    dt: float = 600.0,
    a: float = 6.37122e6,
    omega: float = 7.292e-5,
    g: float = 9.80616,
    Phibar: float = 3.0e5,
    taurad: float = 10.0 * 24.0 * 3600.0,
    taudrag: float = 10.0 * 24.0 * 3600.0,
    K6: float = 1.0e16,
    K6Phi: float = 1.0e16,
    DPhieq: float = 0.0,
    test: Optional[int] = None,
    a1: float = 0.0,
    flags: RunFlags = RunFlags(),
    use_scipy_basis: bool = True,
    jit: bool = True,
    starttime: int = 2,
    state0: Optional[State] = None,
) -> Dict[str, jnp.ndarray]:
    """Run the model via jax.lax.scan — fully differentiable, no side effects.

    Returns a dict with keys: eta, delta, Phi, U, V, spinup, geopot, lambdas, mus.
    Each field array has shape (tmax - starttime, J, I).
    """
    static = build_static(
        M=M, dt=dt, a=a, omega=omega, g=g, Phibar=Phibar,
        K6=K6, K6Phi=K6Phi, DPhieq=DPhieq, test=test,
        use_scipy_basis=use_scipy_basis,
    )

    if state0 is None:
        state0 = _init_state(static, test=test, a1=a1, taurad=taurad, taudrag=taudrag)

    steps = int(max(0, tmax - int(starttime)))
    ts = jnp.arange(int(starttime), int(starttime) + steps, dtype=jnp.int32)

    def body(carry: State, t: jnp.ndarray):
        return _step_once(carry, t, static, flags, taurad, taudrag, test, a1)

    body_fn = jax.jit(body) if jit else body
    _final_state, outs = jax.lax.scan(body_fn, state0, ts)

    return dict(
        eta=outs["eta"],
        delta=outs["delta"],
        Phi=outs["Phi"],
        U=outs["U"],
        V=outs["V"],
        spinup=jnp.stack([outs["spin_min"], outs["rms"]], axis=1),
        geopot=jnp.stack([outs["phi_min"], outs["phi_max"]], axis=1),
        lambdas=static.lambdas,
        mus=static.mus,
    )


def run_model(
    M: int,
    dt: float,
    tmax: int,
    Phibar: float,
    omega: float,
    a: float,
    test: Optional[int] = None,
    **kwargs,
):
    """SWAMPE-compatible wrapper — NOT IMPLEMENTED.

    This function preserves the original SWAMPE ``model.run_model()`` call
    signature but does not work. It exists solely so that import-time
    references (e.g. ``from .model import run_model``) don't break.

    Why it doesn't work
    -------------------
    The original SWAMPE ``run_model()`` interleaves JAX computation with
    Python-side I/O at every time step:

    * **Periodic checkpointing** — pickling (eta, delta, Phi, U, V) to disk
      every ``savefreq`` steps so a crashed run can be resumed.
    * **Periodic plotting** — generating matplotlib figures every ``plotfreq``
      steps during the run.
    * **Continuation / restart** — loading a previous pickle checkpoint via
      ``contflag=True`` and resuming the time loop from that point.

    These I/O side effects require a Python-level ``for`` loop (not
    ``jax.lax.scan``), and the variables that controlled the branching between
    the scan path and the Python-loop path (``wants_periodic``, ``out_dir``)
    were never defined — so the function crashes with ``NameError`` before
    doing any computation.

    What to use instead
    -------------------
    * For **differentiable / JIT-compiled runs**, call ``run_model_scan(...)``
      directly.  It accepts an optional ``state0`` argument for continuation.

    * For **periodic saving**, run ``run_model_scan`` in chunks::

          state = None
          for chunk_start in range(2, tmax, chunk_size):
              out = run_model_scan(
                  tmax=min(chunk_start + chunk_size, tmax),
                  starttime=chunk_start,
                  state0=state,
                  ...
              )
              save_checkpoint(out)  # your own I/O
              state = ...           # reconstruct State from out

    * For **plotting**, pull fields out of the returned dict and plot after
      the run completes, or use the chunked approach above.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "run_model() is not functional in this JAX port.\n"
        "\n"
        "The legacy SWAMPE wrapper requires a Python-level time loop with periodic\n"
        "disk I/O (checkpointing, plotting, continuation), but the variables that\n"
        "controlled this path ('wants_periodic', 'out_dir') were never defined.\n"
        "\n"
        "Use run_model_scan(...) instead for the differentiable / JIT-compiled path.\n"
        "See the docstring of this function for a chunked-save pattern if you need\n"
        "periodic checkpointing."
    )