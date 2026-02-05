from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from . import filters
from . import forcing
from . import initial_conditions
from . import spectral_transform as st
from . import time_stepping
from . import continuation


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class RunFlags:
    """Runtime flags for the JAX core.

    Defaults are chosen to match the original numpy SWAMPE driver.
    """

    forcflag: bool = True
    diffflag: bool = True
    expflag: bool = False
    modalflag: bool = True
    alpha: float = 0.01
    blowup_rms: float = 8000.0


@dataclass(frozen=True)
class Static:
    """Static, precomputed objects used by the model."""

    M: int
    N: int
    I: int
    J: int
    dt: float
    a: float
    omega: float
    g: float
    Phibar: float
    lambdas: jnp.ndarray
    mus: jnp.ndarray
    Pmn: jnp.ndarray
    Hmn: jnp.ndarray
    w: jnp.ndarray
    fmn: jnp.ndarray
    marray: jnp.ndarray
    tstepcoeffmn: jnp.ndarray
    sigma: jnp.ndarray
    sigmaPhi: jnp.ndarray
    Phieq: jnp.ndarray


@dataclass(frozen=True)
class State:
    """Model state for leapfrog stepping (two time levels + spectral helpers)."""

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

    dead: jnp.ndarray  # bool scalar


# =============================================================================
# Setup helpers
# =============================================================================

def build_static(
    *,
    M: int,
    dt: float,
    a: float,
    omega: float,
    g: float,
    Phibar: float,
    taurad: float,
    taudrag: float,
    DPhieq: float,
    K6: float,
    K6Phi: float,
    test: Optional[int],
    use_scipy_basis: bool = True,
) -> Static:
    """Build all static (time-invariant) data for the run."""
    N = M
    I = 2 * M + 1
    J = M + 1

    lambdas = st.build_lambdas(I)
    mus = st.build_mus(J)

    # Basis and weights
    if use_scipy_basis:
        Pmn, Hmn, w = st.build_legendre_basis_scipy(J, M, N, mus)
    else:
        Pmn, Hmn, w = st.build_legendre_basis(J, M, N, mus)

    fmn = st.build_fmn(M, N, omega)
    marray = st.build_marray(M, N)
    tstepcoeffmn = st.build_tstepcoeffmn(M, N, a)

    sigma = filters.sigma6(M, N, K6, a, dt)
    sigmaPhi = filters.sigma6Phi(M, N, K6Phi, a, dt)

    if test is None:
        Phieq = forcing.Phieqfun(Phibar, DPhieq, lambdas, mus, I, J, g)
    else:
        Phieq = jnp.zeros((J, I), dtype=jnp.float64)

    return Static(
        M=M,
        N=N,
        I=I,
        J=J,
        dt=float(dt),
        a=float(a),
        omega=float(omega),
        g=float(g),
        Phibar=float(Phibar),
        lambdas=lambdas,
        mus=mus,
        Pmn=Pmn,
        Hmn=Hmn,
        w=w,
        fmn=fmn,
        marray=marray,
        tstepcoeffmn=tstepcoeffmn,
        sigma=sigma,
        sigmaPhi=sigmaPhi,
        Phieq=Phieq,
    )


def _init_state(
    static: Static,
    *,
    test: Optional[int],
    a1: float,
    taurad: float,
    taudrag: float,
) -> State:
    """Initialize the model state from canonical initial conditions."""
    I, J, M, N = static.I, static.J, static.M, static.N

    eta0, eta1, delta0, delta1, Phi0, Phi1, U0, V0 = initial_conditions.state_var_init(
        test=test,
        a1=a1,
        Phibar=static.Phibar,
        lambdas=static.lambdas,
        mus=static.mus,
        I=I,
        J=J,
    )

    eta0 = jnp.asarray(eta0, dtype=jnp.float64)
    eta1 = jnp.asarray(eta1, dtype=jnp.float64)
    delta0 = jnp.asarray(delta0, dtype=jnp.float64)
    delta1 = jnp.asarray(delta1, dtype=jnp.float64)
    Phi0 = jnp.asarray(Phi0, dtype=jnp.float64)
    Phi1 = jnp.asarray(Phi1, dtype=jnp.float64)
    U0 = jnp.asarray(U0, dtype=jnp.float64)
    V0 = jnp.asarray(V0, dtype=jnp.float64)

    # Spectral transforms for current fields.
    etam0 = st.fwd_fft_trunc(eta0, I, M)
    etam1 = st.fwd_fft_trunc(eta1, I, M)
    deltam0 = st.fwd_fft_trunc(delta0, I, M)
    deltam1 = st.fwd_fft_trunc(delta1, I, M)
    Phim0 = st.fwd_fft_trunc(Phi0, I, M)
    Phim1 = st.fwd_fft_trunc(Phi1, I, M)

    A1, B1, C1, D1, E1 = initial_conditions.ABCDE_init(U0, V0, eta1, Phi1, static.mus, I, J)
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
        PhiFm1 = jnp.zeros((J, M), dtype=jnp.complex128)
        Fm1 = jnp.zeros((J, M), dtype=jnp.complex128)
        Gm1 = jnp.zeros((J, M), dtype=jnp.complex128)

    return State(
        etam_prev=etam0,
        etam_curr=etam1,
        deltam_prev=deltam0,
        deltam_curr=deltam1,
        Phim_prev=Phim0,
        Phim_curr=Phim1,
        eta_prev=eta0,
        delta_prev=delta0,
        Phi_prev=Phi0,
        eta_curr=eta1,
        delta_curr=delta1,
        Phi_curr=Phi1,
        U_curr=U0,
        V_curr=V0,
        Am_curr=Am1,
        Bm_curr=Bm1,
        Cm_curr=Cm1,
        Dm_curr=Dm1,
        Em_curr=Em1,
        PhiFm_curr=PhiFm1,
        Fm_curr=Fm1,
        Gm_curr=Gm1,
        dead=jnp.asarray(False),
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
        PhiFm1 = jnp.zeros((J, M), dtype=jnp.complex128)
        Fm1 = jnp.zeros((J, M), dtype=jnp.complex128)
        Gm1 = jnp.zeros((J, M), dtype=jnp.complex128)

    # Match SWAMPE continuation: previous == current initially.
    return State(
        etam_prev=etam0,
        etam_curr=etam0,
        deltam_prev=deltam0,
        deltam_curr=deltam0,
        Phim_prev=Phim0,
        Phim_curr=Phim0,
        eta_prev=eta0,
        delta_prev=delta0,
        Phi_prev=Phi0,
        eta_curr=eta0,
        delta_curr=delta0,
        Phi_curr=Phi0,
        U_curr=U0,
        V_curr=V0,
        Am_curr=Am1,
        Bm_curr=Bm1,
        Cm_curr=Cm1,
        Dm_curr=Dm1,
        Em_curr=Em1,
        PhiFm_curr=PhiFm1,
        Fm_curr=Fm1,
        Gm_curr=Gm1,
        dead=jnp.asarray(False),
    )


# =============================================================================
# One step (core)
# =============================================================================

def _step_once(
    state: State,
    t: jnp.ndarray,  # int scalar
    static: Static,
    flags: RunFlags,
    taurad: float,
    taudrag: float,
    test: Optional[int],
    a1: float,
) -> tuple[State, Dict[str, Any]]:
    """Single leapfrog step; returns (new_state, diagnostics+new fields)."""
    I, J, M, N = static.I, static.J, static.M, static.N

    dead_prev = state.dead
    rms = time_stepping.RMS_winds(static.a, I, J, static.lambdas, static.mus, state.U_curr, state.V_curr)
    spin_min = jnp.min(jnp.sqrt(state.U_curr * state.U_curr + state.V_curr * state.V_curr))
    dead_next = jnp.logical_or(dead_prev, rms > flags.blowup_rms)

    def skip_update(_: Any) -> tuple[State, Dict[str, Any]]:
        out = dict(
            rms=rms,
            spin_min=spin_min,
            phi_min=jnp.min(state.Phi_curr),
            phi_max=jnp.max(state.Phi_curr),
            eta=state.eta_curr,
            delta=state.delta_curr,
            Phi=state.Phi_curr,
            U=state.U_curr,
            V=state.V_curr,
        )
        return state, out

    def do_update(_: Any) -> tuple[State, Dict[str, Any]]:
        # Forcing in physical space
        def _forcing_fields() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if test is None and flags.forcflag:
                Q1 = forcing.Qfun(static.Phieq, state.Phi_curr, static.Phibar, taurad)
                PhiF1 = Q1
                F1, G1 = forcing.Rfun(state.U_curr, state.V_curr, Q1, state.Phi_curr, static.Phibar, taudrag)
                return PhiF1, F1, G1
            return (
                jnp.zeros((J, I), dtype=jnp.float64),
                jnp.zeros((J, I), dtype=jnp.float64),
                jnp.zeros((J, I), dtype=jnp.float64),
            )

        PhiF1, F1, G1 = _forcing_fields()
        PhiFm2 = st.fwd_fft_trunc(PhiF1, I, M)
        Fm2 = st.fwd_fft_trunc(F1, I, M)
        Gm2 = st.fwd_fft_trunc(G1, I, M)

        # Leapfrog step in spectral space (calls into JAX port of time stepping)
        neweta, newdelta, newPhi, newU, newV, Am2, Bm2, Cm2, Dm2, Em2 = time_stepping.tstepping(
            static.a,
            static.omega,
            static.g,
            static.Phibar,
            state.etam_prev,
            state.etam_curr,
            state.deltam_prev,
            state.deltam_curr,
            state.Phim_prev,
            state.Phim_curr,
            state.Am_curr,
            state.Bm_curr,
            state.Cm_curr,
            state.Dm_curr,
            state.Em_curr,
            PhiFm2,
            Fm2,
            Gm2,
            static.sigma,
            static.sigmaPhi,
            static.lambdas,
            static.mus,
            static.Pmn,
            static.Hmn,
            static.w,
            static.fmn,
            static.tstepcoeffmn,
            static.marray,
            I,
            J,
            M,
            N,
            float(static.dt),
            int(t),
            test,
            float(a1),
            bool(flags.expflag),
            bool(flags.diffflag),
        )

        # For test 1, keep U,V fixed at their initial values.
        def _fix_uv(u: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            if test == 1:
                _, _, _, _, _, _, Uic, Vic = initial_conditions.state_var_init(
                    test=1,
                    a1=a1,
                    Phibar=static.Phibar,
                    lambdas=static.lambdas,
                    mus=static.mus,
                    I=I,
                    J=J,
                )
                return jnp.asarray(Uic, dtype=jnp.float64), jnp.asarray(Vic, dtype=jnp.float64)
            return u, v

        newU, newV = _fix_uv(newU, newV)

        # Modal splitting / Robert-Asselin filter (matches numpy SWAMPE modal_splitting)
        do_ra = jnp.logical_and(flags.modalflag, t > 2)

        def apply_ra(_: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            eta_mid = state.eta_curr + flags.alpha * (state.eta_prev - 2.0 * state.eta_curr + neweta)
            delta_mid = state.delta_curr + flags.alpha * (state.delta_prev - 2.0 * state.delta_curr + newdelta)
            Phi_mid = state.Phi_curr + flags.alpha * (state.Phi_prev - 2.0 * state.Phi_curr + newPhi)
            return eta_mid, delta_mid, Phi_mid

        def no_ra(_: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
            eta_prev=eta_mid,
            delta_prev=delta_mid,
            Phi_prev=Phi_mid,
            eta_curr=neweta,
            delta_curr=newdelta,
            Phi_curr=newPhi,
            U_curr=newU,
            V_curr=newV,
            Am_curr=Am2,
            Bm_curr=Bm2,
            Cm_curr=Cm2,
            Dm_curr=Dm2,
            Em_curr=Em2,
            PhiFm_curr=PhiFm2,
            Fm_curr=Fm2,
            Gm_curr=Gm2,
            dead=dead_next,
        )

        out = dict(
            rms=rms,
            spin_min=spin_min,
            phi_min=phi_min_out,
            phi_max=phi_max_out,
            eta=neweta,
            delta=newdelta,
            Phi=newPhi,
            U=newU,
            V=newV,
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
    dt: float = 300.0,
    a: float = 6.0e6,
    omega: float = 7.0e-5,
    g: float = 9.80616,
    Phibar: float = 4.0e6,
    DPhieq: float = 0.0,
    taurad: float = 86400.0,
    taudrag: float = 86400.0,
    a1: float = 0.05,
    K6: float = 1.0e16,
    K6Phi: Optional[float] = None,
    test: Optional[int] = None,
    flags: RunFlags = RunFlags(),
    starttime: int = 2,
    state0: Optional[State] = None,
    use_scipy_basis: bool = True,
) -> Dict[str, Any]:
    """Differentiable full run returning time histories (JAX scan).

    Note: this returns steps for t = starttime..tmax-1 (length tmax-starttime).
    SWAMPE’s plotting triggers at absolute t (e.g. t%plotfreq==0), so you must
    map scan index i -> t = starttime + i.
    """
    if tmax < starttime:
        raise ValueError(f"tmax={tmax} must be >= starttime={starttime}")
    if dt <= 0:
        raise ValueError("dt must be positive")

    K6Phi_eff = float(K6) if K6Phi is None else float(K6Phi)

    static = build_static(
        M=int(M),
        dt=float(dt),
        a=float(a),
        omega=float(omega),
        g=float(g),
        Phibar=float(Phibar),
        taurad=float(taurad),
        taudrag=float(taudrag),
        DPhieq=float(DPhieq),
        K6=float(K6),
        K6Phi=float(K6Phi_eff),
        test=test,
        use_scipy_basis=bool(use_scipy_basis),
    )

    if state0 is None:
        state0 = _init_state(static, test=test, a1=float(a1), taurad=float(taurad), taudrag=float(taudrag))

    def scan_step(s: State, t_int: jnp.ndarray) -> tuple[State, Dict[str, Any]]:
        return _step_once(s, t_int, static, flags, float(taurad), float(taudrag), test, float(a1))

    ts = jnp.arange(int(starttime), int(tmax), dtype=jnp.int32)
    stateT, outs = jax.lax.scan(scan_step, state0, ts)

    return dict(
        eta=outs["eta"],
        delta=outs["delta"],
        Phi=outs["Phi"],
        U=outs["U"],
        V=outs["V"],
        spinup=jnp.stack([outs["spin_min"], outs["rms"]], axis=-1),
        geopot=jnp.stack([outs["phi_min"], outs["phi_max"]], axis=-1),
        lambdas=static.lambdas,
        mus=static.mus,
        stateT=stateT,
    )


def run_model(
    M: int,
    dt: float,
    tmax: int,
    Phibar: float,
    omega: float,
    a: float,
    test: Optional[int] = None,
    g: float = 9.8,
    forcflag: bool = True,
    taurad: float = 86400.0,
    taudrag: float = 86400.0,
    DPhieq: float = 4 * (10**6),
    a1: float = 0.05,
    plotflag: bool = True,
    plotfreq: int = 5,
    minlevel: Optional[float] = None,
    maxlevel: Optional[float] = None,
    diffflag: bool = True,
    modalflag: bool = True,
    alpha: float = 0.01,
    contflag: bool = False,
    saveflag: bool = True,
    expflag: bool = False,
    savefreq: int = 150,
    K6: float = 1.24 * (10**33),
    custompath: Optional[str] = None,
    contTime: Optional[str] = None,
    timeunits: str = "hours",
    verbose: bool = True,
    *,
    # JAX-only extension: allow different hyperviscosity for Phi if desired.
    K6Phi: Optional[float] = None,
    # JAX-only extension: choose which spectral basis builder to use.
    use_scipy_basis: bool = True,
    # JAX-only extension: whether to jit the per-step function (recommended).
    jit: bool = True,
) -> Dict[str, Any]:
    """SWAMPE-compatible driver (plots/saves like the original numpy code).

    This is intentionally *not* differentiable (it does Python control flow, I/O,
    and optional matplotlib plotting). For differentiable workflows, use
    :func:`run_model_scan` instead.

    Returns
    -------
    dict
        Convenience outputs (final snapshot + spinup/geopot arrays). Legacy SWAMPE
        callers typically ignore the return value.
    """
    import numpy as np

    if test not in (None, 1, 2):
        raise NotImplementedError(
            f"test={test!r} is not supported by this port. Supported: None (forced), 1, 2."
        )
    if tmax < 2:
        raise ValueError(f"tmax must be >= 2 for leapfrog stepping; got {tmax}.")
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}.")

    # Static precomputations (basis, coefficients, equilibrium forcing, etc.)
    K6Phi_eff = float(K6) if K6Phi is None else float(K6Phi)
    static = build_static(
        M=int(M),
        dt=float(dt),
        a=float(a),
        omega=float(omega),
        g=float(g),
        Phibar=float(Phibar),
        taurad=float(taurad),
        taudrag=float(taudrag),
        DPhieq=float(DPhieq),
        K6=float(K6),
        K6Phi=float(K6Phi_eff),
        test=test,
        use_scipy_basis=bool(use_scipy_basis),
    )

    flags = RunFlags(
        forcflag=bool(forcflag),
        diffflag=bool(diffflag),
        expflag=bool(expflag),
        modalflag=bool(modalflag),
        alpha=float(alpha),
    )

    # Initialize or restart state.
    if contflag:
        if contTime is None:
            raise ValueError("contflag=True requires contTime (a timestamp string/int).")
        # SWAMPE continuation uses the timestamp (in `timeunits`) to recover the timestep index.
        starttime = continuation.compute_t_from_timestamp(timeunits, int(contTime), float(dt))

        eta0 = continuation.read_pickle(f"eta-{contTime}", custompath=custompath)
        delta0 = continuation.read_pickle(f"delta-{contTime}", custompath=custompath)
        Phi0 = continuation.read_pickle(f"Phi-{contTime}", custompath=custompath)

        state = _init_state_from_fields(
            static,
            eta=jnp.asarray(eta0),
            delta=jnp.asarray(delta0),
            Phi=jnp.asarray(Phi0),
            test=test,
            taurad=float(taurad),
            taudrag=float(taudrag),
        )
    else:
        starttime = 2
        state = _init_state(static, test=test, a1=float(a1), taurad=float(taurad), taudrag=float(taudrag))

    # Preallocate spin-up diagnostics (matches numpy SWAMPE shape/layout).
    spinupdata = np.zeros((int(tmax), 2), dtype=np.float64)
    geopotdata = np.zeros((int(tmax), 2), dtype=np.float64)

    # Initial diagnostics (numpy SWAMPE only filled these when not continuing).
    if not contflag:
        U0 = state.U_curr
        V0 = state.V_curr
        spinupdata[0, 0] = float(jnp.min(jnp.sqrt(U0 * U0 + V0 * V0)))
        spinupdata[0, 1] = float(
            time_stepping.RMS_winds(static.a, static.I, static.J, static.lambdas, static.mus, U0, V0)
        )

        geopotdata[0, 0] = float(jnp.min(state.Phi_prev))
        geopotdata[0, 1] = float(jnp.max(state.Phi_prev))

    # Jitted step (keeps Python loop from re-tracing on every t).
    def _step(s: State, t: jnp.ndarray) -> tuple[State, Dict[str, Any]]:
        return _step_once(s, t, static, flags, float(taurad), float(taudrag), test, float(a1))

    step_fn = jax.jit(_step) if jit else _step

    # Convert basis arrays once for plotting (if enabled).
    if plotflag:
        from . import plotting  # local import to avoid importing matplotlib unless needed

        mus_np = np.asarray(static.mus)
        lambdas_np = np.asarray(static.lambdas)

    last_out: Optional[Dict[str, Any]] = None

    for t in range(int(starttime), int(tmax)):
        t_j = jnp.asarray(t, dtype=jnp.int32)
        state, out_t = step_fn(state, t_j)
        last_out = out_t

        # Diagnostics are recorded at index (t-1) in the original SWAMPE driver.
        spinupdata[t - 1, 0] = float(out_t["spin_min"])
        spinupdata[t - 1, 1] = float(out_t["rms"])
        geopotdata[t - 1, 0] = float(out_t["phi_min"])
        geopotdata[t - 1, 1] = float(out_t["phi_max"])

        # Optional periodic saving.
        if saveflag and savefreq and (t % int(savefreq) == 0):
            ts = continuation.compute_timestamp(timeunits, int(t), float(dt))
            continuation.save_data(
                ts,
                np.asarray(out_t["eta"]),
                np.asarray(out_t["delta"]),
                np.asarray(out_t["Phi"]),
                np.asarray(out_t["U"]),
                np.asarray(out_t["V"]),
                spinupdata,
                geopotdata,
                custompath=custompath,
            )

        # Stop early if the run blows up (matches numpy SWAMPE behavior).
        if spinupdata[t - 1, 1] > float(flags.blowup_rms):
            print(
                "Time stepping stopped due to wind blow up. "
                f"Max RMS winds = {spinupdata[t - 1, 1]}"
            )
            break

        # Verbose progress.
        if verbose:
            stride = max(1, int(int(tmax) / 10))
            if t % stride == 0:
                print(f"time step {t} completed")

        # Optional plotting.
        if plotflag and plotfreq and (t % int(plotfreq) == 0):
            ts = continuation.compute_timestamp(timeunits, int(t), float(dt))

            Uplot = np.asarray(out_t["U"])
            Vplot = np.asarray(out_t["V"])
            Phiplot = np.asarray(out_t["Phi"]) + float(Phibar)

            plotting.mean_zonal_wind_plot(Uplot, mus_np, ts, units=timeunits)
            plotting.quiver_geopot_plot(
                Uplot,
                Vplot,
                Phiplot,
                lambdas_np,
                mus_np,
                ts,
                units=timeunits,
                minlevel=minlevel,
                maxlevel=maxlevel,
            )
            plotting.spinup_plot(spinupdata, float(dt), units=timeunits)

    if verbose:
        print("GCM run completed!")

    # Convenience return (legacy SWAMPE scripts typically ignore this).
    if last_out is None:
        last_out = dict(
            eta=np.asarray(state.eta_curr),
            delta=np.asarray(state.delta_curr),
            Phi=np.asarray(state.Phi_curr),
            U=np.asarray(state.U_curr),
            V=np.asarray(state.V_curr),
        )

    return dict(
        eta=np.asarray(last_out["eta"]),
        delta=np.asarray(last_out["delta"]),
        Phi=np.asarray(last_out["Phi"]),
        U=np.asarray(last_out["U"]),
        V=np.asarray(last_out["V"]),
        spinup=np.asarray(spinupdata),
        geopot=np.asarray(geopotdata),
        lambdas=np.asarray(static.lambdas),
        mus=np.asarray(static.mus),
    )
