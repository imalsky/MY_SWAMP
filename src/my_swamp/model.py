# ruff: noqa: E741
"""my_swamp.model

JAX rewrite of SWAMPE's main driver (spectral shallow-water model).

Design
------
- `run_model_scan(...)` is the differentiable, side-effect-free core:
    * builds static spectral machinery
    * initializes the state
    * advances with `jax.lax.scan`
    * returns time histories as JAX arrays

- `run_model(...)` preserves the original SWAMPE call signature and performs
  optional side effects (plotting / saving / continuation) outside the
  differentiable core.

Differentiability notes
-----------------------
This file avoids coercing JAX tracers to Python scalars (e.g. via `float(...)`).
Such coercions break `jax.grad` and `jax.jit` when you differentiate with
respect to scalar parameters (e.g. DPhieq, taurad, taudrag, K6, etc.).

The continuation / plotting / pickle I/O paths necessarily use Python-side
operations and are not meant to be differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from .dtypes import Scalar, float_dtype
import numpy as np

from .branching import cond

from . import continuation
from . import filters
from . import forcing
from . import initial_conditions
from . import spectral_transform as st
from . import time_stepping


def _is_python_scalar(x: Any) -> bool:
    """Return True when ``x`` is a concrete Python or NumPy float-like scalar."""
    return isinstance(x, (int, float, np.floating))


def _tree_has_tracer(pytree: Any) -> bool:
    """Return True if any leaf in ``pytree`` is a JAX tracer."""
    for leaf in jax.tree_util.tree_leaves(pytree):
        if isinstance(leaf, jax.core.Tracer):
            return True
    return False


@lru_cache(maxsize=None)
def _cached_geometry(M: int):
    """Cache quadrature + basis arrays that depend only on spectral truncation `M`.

    This avoids repeated SciPy / NumPy work inside `st.PmnHmn`, which is costly
    in optimization loops where only a handful of scalar parameters change.

    Parameters
    ----------
    M : int
        Spectral truncation order.

    Returns
    -------
    tuple[int, int, int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Cached grid sizes, longitude/latitude quadrature arrays, Legendre
        tables, and timestep helper arrays derived from ``M``.
    """

    N, I, J, _, lambdas, mus, w = initial_conditions.spectral_params(int(M))
    Pmn, Hmn = st.PmnHmn(J, int(M), N, mus)
    marray = time_stepping.marray(int(M), N)
    mJarray = time_stepping.mJarray(J, int(M))
    narray = time_stepping.narray(int(M), N)
    return N, I, J, lambdas, mus, w, Pmn, Hmn, marray, mJarray, narray

@lru_cache(maxsize=None)
def _get_simulate_scan_jit(*, test: Optional[int], donate_state: bool):
    """Get a cached jitted wrapper around `simulate_scan` for the given mode.
    
    Parameters
    ----------
    test : Optional[int]
        Static test-case selector forwarded into :func:`simulate_scan`.
    donate_state : bool
        Whether to donate the initial state buffer to the compiled JAX function.
    
    Returns
    -------
    Any
        Cached jitted callable wrapping :func:`simulate_scan` for the given
        static configuration.
    """

    def _fn(state0: State, t_seq: jnp.ndarray, static: Static, flags: RunFlags, Uic: jnp.ndarray, Vic: jnp.ndarray):
        """Run the cached scan kernel with captured static configuration."""
        return simulate_scan(static=static, flags=flags, state0=state0, t_seq=t_seq, test=test, Uic=Uic, Vic=Vic)

    return jax.jit(_fn, donate_argnums=(0,) if donate_state else ())


@lru_cache(maxsize=None)
def _get_simulate_scan_last_jit(*, test: Optional[int], donate_state: bool, remat_step: bool):
    """Get a cached jitted wrapper around `simulate_scan_last` for the given mode.
    
    Parameters
    ----------
    test : Optional[int]
        Static test-case selector forwarded into :func:`simulate_scan_last`.
    donate_state : bool
        Whether to donate the initial state buffer to the compiled JAX function.
    remat_step : bool
        Whether to checkpoint each per-step update inside the scan.
    
    Returns
    -------
    Any
        Cached jitted callable wrapping :func:`simulate_scan_last` for the
        requested static configuration.
    """

    def _fn(
        state0: State,
        t_seq: jnp.ndarray,
        static: Static,
        flags: RunFlags,
        Uic: jnp.ndarray,
        Vic: jnp.ndarray,
    ) -> State:
        """Run the cached state-only scan kernel with captured static configuration."""
        return simulate_scan_last(
            static=static,
            flags=flags,
            state0=state0,
            t_seq=t_seq,
            test=test,
            Uic=Uic,
            Vic=Vic,
            remat_step=remat_step,
        )

    return jax.jit(_fn, donate_argnums=(0,) if donate_state else ())


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RunFlags:
    """Immutable runtime switches for forcing, diffusion, and diagnostics.

    This dataclass stores scalar configuration values that control optional
    model branches during a run. Boolean fields toggle major numerical paths,
    while `alpha` and `blowup_rms` are scalar floating-point thresholds used by
    the stepper and diagnostics logic.
    """

    forcflag: bool = True
    diffflag: bool = True
    expflag: bool = False
    modalflag: bool = True
    diagnostics: bool = True
    alpha: Scalar = 0.01
    blowup_rms: Scalar = 8000.0


    def tree_flatten(self):
        """Flatten into JAX-array children (alpha, blowup_rms) and Python aux data (the bool flags)."""
        children = (
            jnp.asarray(self.alpha, dtype=float_dtype()),
            jnp.asarray(self.blowup_rms, dtype=float_dtype()),
        )
        aux_data = (self.forcflag, self.diffflag, self.expflag, self.modalflag, self.diagnostics)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct :class:`RunFlags` from JAX pytree children and metadata."""
        forcflag, diffflag, expflag, modalflag, diagnostics = aux_data
        alpha, blowup_rms = children
        return cls(
            forcflag=bool(forcflag),
            diffflag=bool(diffflag),
            expflag=bool(expflag),
            modalflag=bool(modalflag),
            diagnostics=bool(diagnostics),
            alpha=alpha,
            blowup_rms=blowup_rms,
        )

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Static:
    """Frozen container for geometry, operators, and static physical constants.

    The fields hold integer spectral/grid sizes plus JAX arrays containing
    timestep scalars, planetary constants, Gaussian-grid geometry, Legendre
    tables, forcing coefficients, and diffusion operators. Instances are
    treated as JAX PyTrees so the static numerical context can flow through
    compiled simulation functions without rebuilding these tensors each step.
    """

    M: int
    N: int
    I: int
    J: int

    dt: jnp.ndarray
    a: jnp.ndarray
    omega: jnp.ndarray
    g: jnp.ndarray
    Phibar: jnp.ndarray
    taurad: jnp.ndarray
    taudrag: jnp.ndarray

    lambdas: jnp.ndarray
    mus: jnp.ndarray
    w: jnp.ndarray

    Pmn: jnp.ndarray
    Hmn: jnp.ndarray
    Pmnw: jnp.ndarray
    Hmnw: jnp.ndarray

    fmn: jnp.ndarray

    tstepcoeff: jnp.ndarray
    tstepcoeff2: jnp.ndarray
    tstepcoeffmn: jnp.ndarray
    marray: jnp.ndarray
    mJarray: jnp.ndarray
    narray: jnp.ndarray

    sigma: jnp.ndarray
    sigmaPhi: jnp.ndarray

    Phieq: jnp.ndarray


    def tree_flatten(self):
        """Flatten into JAX-array children (geometry, scalars, basis) and Python aux data (M, N, I, J)."""
        children = (
            self.dt,
            self.a,
            self.omega,
            self.g,
            self.Phibar,
            self.taurad,
            self.taudrag,
            self.lambdas,
            self.mus,
            self.w,
            self.Pmn,
            self.Hmn,
            self.Pmnw,
            self.Hmnw,
            self.fmn,
            self.tstepcoeff,
            self.tstepcoeff2,
            self.tstepcoeffmn,
            self.marray,
            self.mJarray,
            self.narray,
            self.sigma,
            self.sigmaPhi,
            self.Phieq,
        )
        aux_data = (self.M, self.N, self.I, self.J)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct :class:`Static` from JAX pytree children and metadata."""
        M, N, I, J = aux_data
        (
            dt,
            a,
            omega,
            g,
            Phibar,
            taurad,
            taudrag,
            lambdas,
            mus,
            w,
            Pmn,
            Hmn,
            Pmnw,
            Hmnw,
            fmn,
            tstepcoeff,
            tstepcoeff2,
            tstepcoeffmn,
            marray,
            mJarray,
            narray,
            sigma,
            sigmaPhi,
            Phieq,
        ) = children
        return cls(
            M=int(M),
            N=int(N),
            I=int(I),
            J=int(J),
            dt=dt,
            a=a,
            omega=omega,
            g=g,
            Phibar=Phibar,
            taurad=taurad,
            taudrag=taudrag,
            lambdas=lambdas,
            mus=mus,
            w=w,
            Pmn=Pmn,
            Hmn=Hmn,
            Pmnw=Pmnw,
            Hmnw=Hmnw,
            fmn=fmn,
            tstepcoeff=tstepcoeff,
            tstepcoeff2=tstepcoeff2,
            tstepcoeffmn=tstepcoeffmn,
            marray=marray,
            mJarray=mJarray,
            narray=narray,
            sigma=sigma,
            sigmaPhi=sigmaPhi,
            Phieq=Phieq,
        )

class State(NamedTuple):
    """Scan carry: all JAX arrays."""

    etam_prev: jnp.ndarray
    etam_curr: jnp.ndarray
    deltam_prev: jnp.ndarray
    deltam_curr: jnp.ndarray
    Phim_prev: jnp.ndarray
    Phim_curr: jnp.ndarray

    # Physical fields (for diagnostics + Robert–Asselin filter)
    eta_prev: jnp.ndarray
    eta_curr: jnp.ndarray
    delta_prev: jnp.ndarray
    delta_curr: jnp.ndarray
    Phi_prev: jnp.ndarray
    Phi_curr: jnp.ndarray

    U_curr: jnp.ndarray
    V_curr: jnp.ndarray

    # Fourier of winds (used in time stepping)
    Um_curr: jnp.ndarray
    Vm_curr: jnp.ndarray

    # Nonlinear terms in spectral space for current step
    Am_curr: jnp.ndarray
    Bm_curr: jnp.ndarray
    Cm_curr: jnp.ndarray
    Dm_curr: jnp.ndarray
    Em_curr: jnp.ndarray

    # Forcing in spectral space for current step
    PhiFm_curr: jnp.ndarray
    Fm_curr: jnp.ndarray
    Gm_curr: jnp.ndarray

    dead: jnp.ndarray  # bool scalar


def _dedupe_state_for_donation(state: State) -> State:
    """Clone aliased array leaves so JAX buffer donation can succeed.
    
    JAX donation rejects pytrees that contain the same underlying array object
    in multiple leaf positions. Our two-level initialization intentionally
    starts with prev==curr values, which can alias at the object level.
    
    Parameters
    ----------
    state : State
    
    Returns
    -------
    State
    """
    seen_ids: set[int] = set()

    def _dedupe_leaf(x: Any) -> Any:
        """Return ``x`` if first time seen, else a fresh copy so JAX donation accepts it."""
        if isinstance(x, jax.Array):
            obj_id = id(x)
            if obj_id in seen_ids:
                return jnp.copy(x)
            seen_ids.add(obj_id)
        return x

    return jax.tree_util.tree_map(_dedupe_leaf, state)


def build_static(
    *,
    M: int,
    dt: Scalar,
    a: Scalar,
    omega: Scalar,
    g: Scalar,
    Phibar: Scalar,
    taurad: Scalar,
    taudrag: Scalar,
    DPhieq: Scalar,
    K6: Scalar,
    K6Phi: Optional[Scalar],
    test: Optional[int],
) -> Static:
    """Build time-invariant arrays (quadrature, basis, diffusion, coefficients).

    Parameters
    ----------
    M : int
    dt : Scalar
    a : Scalar
    omega : Scalar
    g : Scalar
    Phibar : Scalar
    taurad : Scalar
    taudrag : Scalar
    DPhieq : Scalar
    K6 : Scalar
    K6Phi : Optional[Scalar]
    test : Optional[int]
    
    Returns
    -------
    Static
    """

    N, I, J, lambdas, mus, w, Pmn, Hmn, marray, mJarray, narray = _cached_geometry(int(M))
    Pmnw = st.weighted_legendre_basis(Pmn, w)
    Hmnw = st.weighted_legendre_basis(Hmn, w)

    # Keep scalars as JAX values to preserve differentiability.
    dt_j = jnp.asarray(dt, dtype=float_dtype())
    a_j = jnp.asarray(a, dtype=float_dtype())
    omega_j = jnp.asarray(omega, dtype=float_dtype())
    g_j = jnp.asarray(g, dtype=float_dtype())
    Phibar_j = jnp.asarray(Phibar, dtype=float_dtype())
    taurad_j = jnp.asarray(taurad, dtype=float_dtype())
    taudrag_j = jnp.asarray(taudrag, dtype=float_dtype())

    fmn = initial_conditions.coriolismn(int(M), omega_j)

    tstepcoeffmn = time_stepping.tstepcoeffmn(int(M), N, a_j)
    tstepcoeff = time_stepping.tstepcoeff(J, int(M), dt_j, mus, a_j)
    tstepcoeff2 = time_stepping.tstepcoeff2(J, int(M), dt_j, a_j)
    K6_j = jnp.asarray(K6, dtype=float_dtype())
    K6Phi_eff = K6_j if K6Phi is None else jnp.asarray(K6Phi, dtype=float_dtype())

    sigma = filters.sigma6(int(M), N, K6_j, a_j, dt_j)
    sigmaPhi = filters.sigma6Phi(int(M), N, K6Phi_eff, a_j, dt_j)

    if test is None:
        DPhieq_j = jnp.asarray(DPhieq, dtype=float_dtype())
        Phieq = forcing.Phieqfun(Phibar_j, DPhieq_j, lambdas, mus, I, J, g_j)
    else:
        Phieq = jnp.zeros((J, I), dtype=float_dtype())

    return Static(
        M=int(M),
        N=int(N),
        I=int(I),
        J=int(J),
        dt=dt_j,
        a=a_j,
        omega=omega_j,
        g=g_j,
        Phibar=Phibar_j,
        taurad=taurad_j,
        taudrag=taudrag_j,
        lambdas=lambdas,
        mus=mus,
        w=w,
        Pmn=Pmn,
        Hmn=Hmn,
        Pmnw=Pmnw,
        Hmnw=Hmnw,
        fmn=fmn,
        tstepcoeff=tstepcoeff,
        tstepcoeff2=tstepcoeff2,
        tstepcoeffmn=tstepcoeffmn,
        marray=marray,
        mJarray=mJarray,
        narray=narray,
        sigma=sigma,
        sigmaPhi=sigmaPhi,
        Phieq=Phieq,
    )


def _forcing_phys(
    *,
    static: Static,
    flags: RunFlags,
    test: Optional[int],
    Phi: jnp.ndarray,
    U: jnp.ndarray,
    V: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (PhiF, F, G) in physical space for the current state.

    Matches reference SWAMPE: when ``test is None`` the forcing fields F/G/PhiF
    are computed *unconditionally* of ``flags.forcflag``. The reference SWAMPE
    timestepper has a historical quirk where the modified-Euler delta update
    uses ``Bm + Fm`` and ``Am - Gm`` even in its unforced branch, so F/G must
    leak into the divergence tendency for parity.

    The ``forcflag`` switch only controls whether the forcing terms are *added*
    inside the timestepper (Phiforcing addition, etc.); it does not gate the
    computation of F/G/PhiF here.

    Parameters
    ----------
    static : Static
    flags : RunFlags
    test : Optional[int]
    Phi : jnp.ndarray
    U : jnp.ndarray
    V : jnp.ndarray

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    """

    if test is None:
        Q = forcing.Qfun(static.Phieq, Phi, static.Phibar, static.taurad)
        PhiF = Q
        F, G = forcing.Rfun(U, V, Q, Phi, static.Phibar, static.taudrag)
        return PhiF, F, G

    J, I = static.J, static.I
    z = jnp.zeros((J, I), dtype=float_dtype())
    return z, z, z


def _nonlinear_spectral(
    *,
    static: Static,
    eta: jnp.ndarray,
    Phi: jnp.ndarray,
    U: jnp.ndarray,
    V: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (Am,Bm,Cm,Dm,Em) Fourier coefficients for nonlinear terms.
    
    Parameters
    ----------
    static : Static
    eta : jnp.ndarray
    Phi : jnp.ndarray
    U : jnp.ndarray
    V : jnp.ndarray
    
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    """

    # Match the reference SWAMPE ordering and semantics:
    #   ABCDE_init(U, V, eta, Phi, mus, I, J)
    A, B, C, D, E = initial_conditions.ABCDE_init(
        U,
        V,
        eta,
        Phi,
        static.mus,
        static.I,
        static.J,
    )
    Am, Bm, Cm, Dm, Em = st.fwd_fft_trunc_batch(jnp.stack((A, B, C, D, E), axis=0), static.I, static.M)
    return Am, Bm, Cm, Dm, Em


def _diagnose_winds(eta: jnp.ndarray, delta: jnp.ndarray, static: Static) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Diagnose U, V winds from eta and delta via spectral transforms.

    Performs forward FFT + Legendre transform on eta/delta, then inverts the
    wind relation to obtain physical-space U and V.
    """
    etam, deltam = st.fwd_fft_trunc_batch(jnp.stack((eta, delta), axis=0), static.I, static.M)
    etamn = st.fwd_leg(etam, static.J, static.M, static.N, static.Pmn, static.w)
    deltamn = st.fwd_leg(deltam, static.J, static.M, static.N, static.Pmn, static.w)
    Uc, Vc = st.invrsUV(
        deltamn,
        etamn,
        static.fmn,
        static.I,
        static.J,
        static.M,
        static.N,
        static.Pmn,
        static.Hmn,
        static.tstepcoeffmn,
        static.marray,
    )
    return jnp.real(Uc), jnp.real(Vc)


def _analytic_ic(
    static: Static, test: Optional[int], a1: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute analytic initial conditions (eta0, delta0, Phi0, U0, V0).

    Dispatches to test-case-specific initializers from :mod:`initial_conditions`.
    """
    SU0, sina, cosa, etaamp, Phiamp = initial_conditions.test1_init(static.a, static.omega, a1)

    if test in (1, 2):
        eta0, _, delta0, _, Phi0, _ = initial_conditions.state_var_init(
            static.I, static.J, static.mus, static.lambdas, test, etaamp,
            static.a, sina, cosa, static.Phibar, Phiamp,
        )
    else:
        eta0, _, delta0, _, Phi0, _ = initial_conditions.state_var_init(
            static.I, static.J, static.mus, static.lambdas, test, etaamp,
        )

    U0, V0 = initial_conditions.velocity_init(
        static.I, static.J, SU0, cosa, sina, static.mus, static.lambdas, test,
    )
    return eta0, delta0, Phi0, U0, V0


def _init_state_from_fields(
    *,
    static: Static,
    flags: RunFlags,
    test: Optional[int],
    eta0: jnp.ndarray,
    delta0: jnp.ndarray,
    Phi0: jnp.ndarray,
    U0: jnp.ndarray,
    V0: jnp.ndarray,
) -> State:
    """Initialize the scan state with 2-level start (prev==curr==initial).
    
    Parameters
    ----------
    static : Static
    flags : RunFlags
    test : Optional[int]
    eta0 : jnp.ndarray
    delta0 : jnp.ndarray
    Phi0 : jnp.ndarray
    U0 : jnp.ndarray
    V0 : jnp.ndarray
    
    Returns
    -------
    State
    """

    # Forcing at initial step.
    PhiF0, F0, G0 = _forcing_phys(static=static, flags=flags, test=test, Phi=Phi0, U=U0, V=V0)

    # Fourier truncations.
    etam0, deltam0, Phim0, Um0, Vm0 = st.fwd_fft_trunc_batch(
        jnp.stack((eta0, delta0, Phi0, U0, V0), axis=0), static.I, static.M
    )
    PhiFm0, Fm0, Gm0 = st.fwd_fft_trunc_batch(jnp.stack((PhiF0, F0, G0), axis=0), static.I, static.M)

    Am0, Bm0, Cm0, Dm0, Em0 = _nonlinear_spectral(static=static, eta=eta0, Phi=Phi0, U=U0, V=V0)

    dead0 = jnp.asarray(False)

    # Two-level initialization: time levels 0 and 1 are identical.
    return State(
        etam_prev=etam0,
        etam_curr=etam0,
        deltam_prev=deltam0,
        deltam_curr=deltam0,
        Phim_prev=Phim0,
        Phim_curr=Phim0,
        eta_prev=eta0,
        eta_curr=eta0,
        delta_prev=delta0,
        delta_curr=delta0,
        Phi_prev=Phi0,
        Phi_curr=Phi0,
        U_curr=U0,
        V_curr=V0,
        Um_curr=Um0,
        Vm_curr=Vm0,
        Am_curr=Am0,
        Bm_curr=Bm0,
        Cm_curr=Cm0,
        Dm_curr=Dm0,
        Em_curr=Em0,
        PhiFm_curr=PhiFm0,
        Fm_curr=Fm0,
        Gm_curr=Gm0,
        dead=dead0,
    )


def _step_once(
    state: State,
    t: jnp.ndarray,
    static: Static,
    flags: RunFlags,
    test: Optional[int],
    Uic: jnp.ndarray,
    Vic: jnp.ndarray,
) -> Tuple[State, Dict[str, Any]]:
    """Single leapfrog update. Returns (new_state, outputs).

    Parameters
    ----------
    state : State
        Current two-level model state.
    t : jnp.ndarray
        Current integer timestep index.
    static : Static
        Cached geometry and constant model coefficients.
    flags : RunFlags
        Runtime switches that control forcing, diffusion, and diagnostics.
    test : Optional[int]
        Idealized test selector or ``None`` for forced mode.
    Uic : jnp.ndarray
        Initial zonal wind field with shape ``(J, I)`` used by test-case
        parity branches.
    Vic : jnp.ndarray
        Initial meridional wind field with shape ``(J, I)`` used by test-case
        parity branches.

    Returns
    -------
    Tuple[State, Dict[str, Any]]
        Updated state plus the per-step diagnostics/output dictionary emitted by
        the scan driver.
    """

    I, J, M, N = static.I, static.J, static.M, static.N

    # Diagnostics on the *current* state (time level 1).
    #
    # For optimization / autodiff runs you often want to skip these global reductions
    # and the blow-up gating branch. Use flags.diagnostics=False for that mode.
    if flags.diagnostics:
        rms = time_stepping.RMS_winds(static.a, I, J, static.lambdas, static.mus, state.U_curr, state.V_curr)
        spin_min = jnp.min(jnp.sqrt(state.U_curr * state.U_curr + state.V_curr * state.V_curr))
        dead_next = jnp.logical_or(state.dead, rms > jnp.asarray(flags.blowup_rms))
    else:
        rms = jnp.asarray(0.0, dtype=float_dtype())
        spin_min = jnp.asarray(0.0, dtype=float_dtype())
        dead_next = state.dead

    def skip_update(_: Any) -> Tuple[State, Dict[str, Any]]:
        """Blowup branch: keep the current state and emit current-state diagnostics."""
        out = dict(
            t=t,
            dead=dead_next,
            # these correspond to the "new" state, but we keep them at current on skip
            eta=state.eta_curr,
            delta=state.delta_curr,
            Phi=state.Phi_curr,
            U=state.U_curr,
            V=state.V_curr,
            # diagnostics correspond to time level 1 (current)
            rms=rms,
            spin_min=spin_min,
            phi_min=jnp.min(state.Phi_curr),
            phi_max=jnp.max(state.Phi_curr),
        )
        return state._replace(dead=dead_next), out

    def do_update(_: Any) -> Tuple[State, Dict[str, Any]]:
        """Healthy branch: advance one leapfrog step and build the next scan carry."""
        # Core time stepping (returns physical-space fields + spectral eta/delta/Phi)
        newetamn, neweta, newetam, newdeltamn, newdelta, newdeltam, newPhimn, newPhi, newPhim, newU, newV, newUm, newVm = time_stepping.tstepping(
            state.etam_prev,
            state.etam_curr,
            state.deltam_prev,
            state.deltam_curr,
            state.Phim_prev,
            state.Phim_curr,
            I,
            J,
            M,
            N,
            state.Am_curr,
            state.Bm_curr,
            state.Cm_curr,
            state.Dm_curr,
            state.Em_curr,
            state.Fm_curr,
            state.Gm_curr,
            state.Um_curr,
            state.Vm_curr,
            static.fmn,
            static.Pmn,
            static.Hmn,
            static.Pmnw,
            static.Hmnw,
            static.tstepcoeff,
            static.tstepcoeff2,
            static.tstepcoeffmn,
            static.marray,
            static.mJarray,
            static.narray,
            state.PhiFm_curr,
            static.dt,
            static.a,
            static.Phibar,
            static.taurad,
            static.taudrag,
            flags.forcflag,
            flags.diffflag,
            flags.expflag,
            static.sigma,
            static.sigmaPhi,
            test,
            t,
        )

        # The spectral transforms return complex physical-space fields (IFFT).
        # SWAMPE treats these as real (discarding the negligible imaginary
        # roundoff), and several downstream operations (min/max, comparisons,
        # forcing) require real values.
        neweta = jnp.real(neweta)
        newdelta = jnp.real(newdelta)
        newPhi = jnp.real(newPhi)
        newU = jnp.real(newU)
        newV = jnp.real(newV)

        # Test 1: keep winds fixed to the initial field (matches numpy SWAMPE)
        if test == 1:
            newU, newV = Uic, Vic
            # Keep spectral winds fixed too (avoid per-step FFTs).
            newUm, newVm = state.Um_curr, state.Vm_curr

        # Robert–Asselin / modal splitting affects diagnostics of the *current* level.
        do_ra = jnp.logical_and(jnp.asarray(flags.modalflag), t > 2)
        alpha = jnp.asarray(flags.alpha, dtype=float_dtype())

        def apply_ra(_: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Apply the Robert-Asselin three-level smoothing filter to eta/delta/Phi."""
            eta_mid = state.eta_curr + alpha * (state.eta_prev - 2.0 * state.eta_curr + neweta)
            delta_mid = state.delta_curr + alpha * (state.delta_prev - 2.0 * state.delta_curr + newdelta)
            Phi_mid = state.Phi_curr + alpha * (state.Phi_prev - 2.0 * state.Phi_curr + newPhi)
            return eta_mid, delta_mid, Phi_mid

        def no_ra(_: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Identity branch when the Robert-Asselin filter is disabled (e.g., t<=2)."""
            return state.eta_curr, state.delta_curr, state.Phi_curr

        eta_mid, delta_mid, Phi_mid = jax.lax.cond(do_ra, apply_ra, no_ra, operand=None)

        if flags.diagnostics:
            phi_min = jnp.min(Phi_mid)
            phi_max = jnp.max(Phi_mid)
        else:
            phi_min = jnp.asarray(0.0, dtype=float_dtype())
            phi_max = jnp.asarray(0.0, dtype=float_dtype())

        # Build forcing and nonlinear terms for the NEXT step (based on the new state).
        PhiF2, F2, G2 = _forcing_phys(static=static, flags=flags, test=test, Phi=newPhi, U=newU, V=newV)
        PhiFm2, Fm2, Gm2 = st.fwd_fft_trunc_batch(jnp.stack((PhiF2, F2, G2), axis=0), I, M)

        Am2, Bm2, Cm2, Dm2, Em2 = _nonlinear_spectral(static=static, eta=neweta, Phi=newPhi, U=newU, V=newV)

        # Fourier of prognostic and wind fields for the NEXT step.
        #
        # Avoid redundant physical→spectral FFTs by reusing the truncated Fourier
        # coefficients already computed inside the timestepper/inversion.
        etam2 = newetam
        deltam2 = newdeltam
        Phim2 = newPhim
        Um2 = newUm
        Vm2 = newVm

        new_state = State(
            etam_prev=state.etam_curr,
            etam_curr=etam2,
            deltam_prev=state.deltam_curr,
            deltam_curr=deltam2,
            Phim_prev=state.Phim_curr,
            Phim_curr=Phim2,
            eta_prev=eta_mid,
            eta_curr=neweta,
            delta_prev=delta_mid,
            delta_curr=newdelta,
            Phi_prev=Phi_mid,
            Phi_curr=newPhi,
            U_curr=newU,
            V_curr=newV,
            Um_curr=Um2,
            Vm_curr=Vm2,
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
            t=t,
            dead=dead_next,
            # new state at time t
            eta=neweta,
            delta=newdelta,
            Phi=newPhi,
            U=newU,
            V=newV,
            # diagnostics for time t-1 (current, possibly RA-filtered)
            rms=rms,
            spin_min=spin_min,
            phi_min=phi_min,
            phi_max=phi_max,
        )
        return new_state, out

    if not flags.diagnostics:
        return do_update(None)
    return cond(dead_next, skip_update, do_update, operand=None)


def _step_once_state_only(
    state: State,
    t: jnp.ndarray,
    static: Static,
    flags: RunFlags,
    test: Optional[int],
    Uic: jnp.ndarray,
    Vic: jnp.ndarray,
) -> State:
    """Single leapfrog update returning only the new `State` (no per-step outputs).
    
    This wrapper exists to support highly efficient forward simulations in
    optimization/inference loops where you do not need the per-step `outs`
    dictionary. It enables `simulate_scan_last` (and user-written `fori_loop`
    forward passes) to call a step function whose only output is the new carry.
    
    Notes
    -----
    - When used under `jax.jit`, JAX/XLA will eliminate computations that only
      contribute to the discarded outputs of `_step_once`.
    - For maximum performance in training/inference, set `flags.diagnostics=False`
      to skip global reductions and the blow-up gating branch.
    
    Parameters
    ----------
    state : State
    t : jnp.ndarray
    static : Static
    flags : RunFlags
    test : Optional[int]
    Uic : jnp.ndarray
    Vic : jnp.ndarray
    
    Returns
    -------
    State
    """
    new_state, _ = _step_once(state, t, static, flags, test, Uic, Vic)
    return new_state

def simulate_scan(
    *,
    static: Static,
    flags: RunFlags,
    state0: State,
    t_seq: jnp.ndarray,
    test: Optional[int],
    Uic: jnp.ndarray,
    Vic: jnp.ndarray,
) -> Tuple[State, Dict[str, Any]]:
    """Pure differentiable core: advance along `t_seq` with `lax.scan`.
    
    Parameters
    ----------
    static : Static
        Static geometry, precomputed transforms, and forcing coefficients.
    flags : RunFlags
        Boolean runtime switches and scalar thresholds controlling the solver.
    state0 : State
        Initial model state at the start of the scan.
    t_seq : jnp.ndarray
        One-dimensional sequence of timestep indices.
    test : Optional[int]
        Test selector forwarded into the timestep kernel.
    Uic : jnp.ndarray
        Initial zonal wind field used by the diagnostic updates.
    Vic : jnp.ndarray
        Initial meridional wind field used by the diagnostic updates.
    
    Returns
    -------
    Tuple[State, Dict[str, Any]]
        Final state together with stacked per-step diagnostics produced by
        :func:`_step_once`.
    """

    def step(carry: State, t: jnp.ndarray):
        """Advance one scan step and emit the recorded diagnostics."""
        return _step_once(carry, t, static, flags, test, Uic, Vic)

    last_state, outs = jax.lax.scan(step, state0, t_seq)
    return last_state, outs


def simulate_scan_last(
    *,
    static: Static,
    flags: RunFlags,
    state0: State,
    t_seq: jnp.ndarray,
    test: Optional[int],
    Uic: jnp.ndarray,
    Vic: jnp.ndarray,
    remat_step: bool = False,
) -> State:
    """Advance along `t_seq` but do NOT materialize a time history.
    
    This is the preferred core for optimization/inference where you only need the
    final state (e.g., the terminal `Phi_curr`) and a scalar loss.
    
    Notes
    -----
    - Returning an empty scan output prevents JAX from stacking ~10k copies of
      large 2-D fields.
    - When `remat_step=True`, the per-step computation is rematerialized
      (checkpointed) to trade compute for memory (mostly useful for reverse-mode).
    
    Parameters
    ----------
    static : Static
        Static geometry, precomputed transforms, and forcing coefficients.
    flags : RunFlags
        Boolean runtime switches and scalar thresholds controlling the solver.
    state0 : State
        Initial model state at the start of the scan.
    t_seq : jnp.ndarray
        One-dimensional sequence of timestep indices.
    test : Optional[int]
        Test selector forwarded into the timestep kernel.
    Uic : jnp.ndarray
        Initial zonal wind field used by the diagnostic updates.
    Vic : jnp.ndarray
        Initial meridional wind field used by the diagnostic updates.
    remat_step : bool
        If ``True``, rematerialize the per-step update during reverse-mode
        differentiation to reduce peak memory use.
    
    Returns
    -------
    State
        Final solver state after consuming the full timestep sequence.
    """

    step_core = _step_once_state_only
    if remat_step:
        # `test` is a Python control selector inside `_step_once`; mark it static
        # for checkpointed traces so Python branching remains valid.
        step_core = jax.checkpoint(_step_once_state_only, static_argnums=(4,))

    def step(carry: State, t: jnp.ndarray):
        """Advance one scan step without materializing per-step outputs."""
        new_state = step_core(carry, t, static, flags, test, Uic, Vic)
        return new_state, ()

    last_state, _ = jax.lax.scan(step, state0, t_seq)
    return last_state


def run_model_scan(
    *,
    M: int,
    dt: Scalar,
    tmax: int,
    Phibar: Scalar,
    omega: Scalar,
    a: Scalar,
    test: Optional[int] = None,
    g: Scalar = 9.8,
    forcflag: bool = True,
    taurad: Scalar = 86400.0,
    taudrag: Scalar = 86400.0,
    DPhieq: Scalar = 4 * (10**6),
    a1: Scalar = 0.05,
    diffflag: bool = True,
    modalflag: bool = True,
    alpha: Scalar = 0.01,
    expflag: bool = False,
    K6: Scalar = 1.24 * (10**33),
    K6Phi: Optional[Scalar] = None,
    contflag: bool = False,
    custompath: Optional[str] = None,
    contTime: Optional[str] = None,
    timeunits: str = "hours",
    starttime: Optional[int] = None,
    # Optional: provide explicit initial state (enables differentiating wrt ICs).
    eta0_init: Optional[jnp.ndarray] = None,
    delta0_init: Optional[jnp.ndarray] = None,
    Phi0_init: Optional[jnp.ndarray] = None,
    U0_init: Optional[jnp.ndarray] = None,
    V0_init: Optional[jnp.ndarray] = None,
    # Performance knobs
    diagnostics: bool = True,
    return_history: bool = True,
    remat_step: bool = False,
    jit_scan: bool = True,
    donate_state: bool = False,
) -> Dict[str, Any]:
    """Differentiable full run returning time histories (JAX scan).

    Outputs correspond to times `t_seq = arange(starttime, tmax)`.

    Memory cost
    -----------
    With ``return_history=True`` (the default), ``outs`` materializes a
    ``(len(t_seq), J, I)`` array for each of the five physical fields
    (``eta``, ``delta``, ``Phi``, ``U``, ``V``) plus four ``(len(t_seq),)``
    scalar diagnostics. The dominant footprint is roughly::

        bytes ≈ 5 * len(t_seq) * J * I * itemsize

    where ``itemsize`` is 8 bytes in float64 mode. For the default ``M=42``
    grid (``J=64``, ``I=128``), this is ~328 KB per scan step, or ~33 GB at
    ``len(t_seq)=100_000``. For long integrations and for any optimization
    or inference loop, prefer :func:`run_model_scan_final` (or pass
    ``return_history=False``), which discards the per-step output and only
    materializes the terminal state.

    Parameters
    ----------
    M : int
        Spectral truncation with ``N=M``.
    dt : Scalar
        Float-like timestep in seconds. Concrete Python scalars are validated to
        be positive.
    tmax : int
        Final timestep index. The scan advances over ``arange(starttime, tmax)``.
    Phibar : Scalar
        Reference geopotential scalar.
    omega : Scalar
        Planetary rotation rate in radians per second.
    a : Scalar
        Planetary radius in meters.
    test : Optional[int]
        Idealized test selector or ``None`` for forced mode.
    g : Scalar
        Surface gravity scalar.
    forcflag : bool
        Enables thermal forcing.
    taurad : Scalar
        Radiative relaxation timescale in seconds.
    taudrag : Scalar
        Drag timescale in seconds.
    DPhieq : Scalar
        Day-night equilibrium geopotential contrast.
    a1 : Scalar
        Tilt angle used by the analytic test initial conditions.
    diffflag : bool
        Enables diffusion filtering.
    modalflag : bool
        Enables the modal/Robert-Asselin correction branch.
    alpha : Scalar
        Robert-Asselin filter coefficient.
    expflag : bool
        Selects the explicit timestepper when true and the modified-Euler
        scheme otherwise.
    K6 : Scalar
        Sixth-order diffusion coefficient for vorticity and divergence.
    K6Phi : Optional[Scalar]
        Optional sixth-order diffusion coefficient for geopotential.
    contflag : bool
        Enables continuation from saved disk state.
    custompath : Optional[str]
        Optional directory used for continuation loads.
    contTime : Optional[str]
        Continuation timestamp token or numeric string.
    timeunits : str
        Units used to interpret ``contTime`` when deriving ``starttime``.
    starttime : Optional[int]
        Explicit start index override. If omitted, the continuation timestamp or
        default startup index is used.
    eta0_init : Optional[jnp.ndarray]
        Optional physical-space absolute-vorticity field with shape ``(J, I)``.
    delta0_init : Optional[jnp.ndarray]
        Optional physical-space divergence field with shape ``(J, I)``.
    Phi0_init : Optional[jnp.ndarray]
        Optional physical-space geopotential field with shape ``(J, I)``.
    U0_init : Optional[jnp.ndarray]
        Optional physical-space zonal wind field with shape ``(J, I)``.
    V0_init : Optional[jnp.ndarray]
        Optional physical-space meridional wind field with shape ``(J, I)``.
    diagnostics : bool
        Enables RMS-wind diagnostics and blow-up gating during the scan.
    return_history : bool
        When true, return the full scan outputs; otherwise return only the
        terminal state payload.
    remat_step : bool
        Enables rematerialization of the per-step function to reduce memory.
    jit_scan : bool
        Wraps the scan in a cached `jax.jit` specialization when true.
    donate_state : bool
        Allows JAX to donate the initial state buffers to the compiled scan.

    Returns
    -------
    Dict[str, Any]
        Simulation payload containing the static setup, timestep sequence, and
        either the full output history or the terminal state depending on
        ``return_history``.
    """

    if tmax < 2:
        raise ValueError("tmax must be >= 2 (SWAMPE uses a 2-level initialization).")

    # Critical checks only when dt is a concrete Python scalar.
    if _is_python_scalar(dt) and float(dt) <= 0.0:
        raise ValueError("dt must be positive.")

    # Ensure `test` is a Python int/None (needed for caching/jit).
    if test is not None:
        test = int(test)

    flags = RunFlags(
        forcflag=bool(forcflag),
        diffflag=bool(diffflag),
        expflag=bool(expflag),
        modalflag=bool(modalflag),
        diagnostics=bool(diagnostics),
        alpha=alpha,
    )

    static = build_static(
        M=int(M),
        dt=dt,
        a=a,
        omega=omega,
        g=g,
        Phibar=Phibar,
        taurad=taurad,
        taudrag=taudrag,
        DPhieq=DPhieq,
        K6=K6,
        K6Phi=K6Phi,
        test=test,
    )

    # Determine absolute start time index.
    #
    # SWAMPE's continuation interface treats contTime as a numeric timestamp
    # (typically the integer token appended to saved file names). Be permissive
    # here and accept either an int/float or a numeric string (e.g. "50").
    #
    # Important: we keep the *original* contTime token for file I/O below.
    contTime_token: Optional[str] = None
    contTime_fallback: Optional[str] = None
    timestamp_val: Optional[float] = None

    # Normalize continuation timestamp/token if we are continuing from disk.
    if contflag:
        if contTime is None:
            raise ValueError("contflag=True requires contTime.")

        try:
            timestamp_val = float(contTime)
        except (TypeError, ValueError) as e:
            raise ValueError(f"contTime must be numeric (int/float or numeric string), got {contTime!r}.") from e

        contTime_fallback = str(contTime)

        # Prefer integer formatting when the numeric value is integer-like.
        ts_round = round(timestamp_val)
        if abs(timestamp_val - ts_round) < 1e-12:
            contTime_token = str(int(ts_round))
        else:
            contTime_token = contTime_fallback

    if starttime is None:
        if not contflag:
            starttime_eff = 2
        else:
            try:
                dt_float = float(dt)
            except TypeError as e:
                raise TypeError("dt must be a Python float when contflag=True (continuation uses Python I/O).") from e

            if timestamp_val is None:
                raise RuntimeError("Internal error: contflag=True but contTime was not parsed.")
            starttime_eff = continuation.compute_t_from_timestamp(timeunits, timestamp_val, dt_float)
    else:
        starttime_eff = int(starttime)

    if starttime_eff > tmax:
        raise ValueError(f"starttime={starttime_eff} must be <= tmax={tmax}.")

    # Initialize physical fields.
    have_explicit_ic = (eta0_init is not None) or (delta0_init is not None) or (Phi0_init is not None)

    if have_explicit_ic:
        if eta0_init is None or delta0_init is None or Phi0_init is None:
            raise ValueError("If providing explicit ICs, eta0_init, delta0_init, and Phi0_init must all be provided.")

        eta0 = jnp.asarray(eta0_init, dtype=float_dtype())
        delta0 = jnp.asarray(delta0_init, dtype=float_dtype())
        Phi0 = jnp.asarray(Phi0_init, dtype=float_dtype())

        expected_shape = (static.J, static.I)
        for name, arr in (("eta0_init", eta0), ("delta0_init", delta0), ("Phi0_init", Phi0)):
            if arr.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}.")

        if (U0_init is None) != (V0_init is None):
            raise ValueError("Provide both U0_init and V0_init, or neither.")

        if U0_init is not None:
            U0 = jnp.asarray(U0_init, dtype=float_dtype())
            V0 = jnp.asarray(V0_init, dtype=float_dtype())

            for name, arr in (("U0_init", U0), ("V0_init", V0)):
                if arr.shape != expected_shape:
                    raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}.")
        else:
            # Diagnose winds from eta/delta.
            U0, V0 = _diagnose_winds(eta0, delta0, static)

    elif not contflag:
        # Analytic initialization
        eta0, delta0, Phi0, U0, V0 = _analytic_ic(static, test, a1)

    else:
        # Continuation initialization (loads eta/delta/Phi and diagnoses winds).
        if contTime is None:
            raise ValueError("contflag=True requires contTime.")

        # Prefer the canonical integer token (e.g. "50") when contTime is
        # provided as a float-like string (e.g. "50.0"), but fall back to the
        # original string representation if the canonical name is not found.
        cont_key = contTime_token if contTime_token is not None else str(contTime)
        cont_fallback = str(contTime)

        def _read_with_fallback(prefix: str):
            """Read with fallback.
            
            Parameters
            ----------
            prefix : str
            
            Returns
            -------
            Any
            """
            try:
                return continuation.read_pickle(f"{prefix}-{cont_key}", custompath=custompath)
            except FileNotFoundError:
                if cont_fallback != cont_key:
                    return continuation.read_pickle(f"{prefix}-{cont_fallback}", custompath=custompath)
                raise

        eta0 = jnp.asarray(_read_with_fallback("eta"), dtype=float_dtype())
        delta0 = jnp.asarray(_read_with_fallback("delta"), dtype=float_dtype())
        Phi0 = jnp.asarray(_read_with_fallback("Phi"), dtype=float_dtype())

        U0, V0 = _diagnose_winds(eta0, delta0, static)

    # Constant winds for test==1 override.
    Uic = U0
    Vic = V0

    state0 = _init_state_from_fields(
        static=static,
        flags=flags,
        test=test,
        eta0=eta0,
        delta0=delta0,
        Phi0=Phi0,
        U0=U0,
        V0=V0,
    )

    t_seq = jnp.arange(starttime_eff, tmax, dtype=jnp.int32)

    # JIT only the time-advancement. Static/basis construction and the
    # initialization logic remain on the Python side so we don't repeatedly
    # compile or stage out large constant-building graphs.
    #
    # IMPORTANT for differentiability:
    #   Do NOT close over potentially-traced values (static/flags/Uic/Vic)
    #   inside the jitted function. Instead, pass them as explicit arguments.
    donate_eff = bool(donate_state) and (not _tree_has_tracer((state0, t_seq, static, flags, Uic, Vic)))
    if donate_eff:
        state0 = _dedupe_state_for_donation(state0)
        # Avoid aliasing a donated state buffer with non-donated explicit args.
        Uic = jnp.copy(Uic)
        Vic = jnp.copy(Vic)

    if return_history:
        if jit_scan:
            simulate_fn = _get_simulate_scan_jit(test=test, donate_state=donate_eff)
            last_state, outs = simulate_fn(state0, t_seq, static, flags, Uic, Vic)
        else:
            last_state, outs = simulate_scan(
                static=static,
                flags=flags,
                state0=state0,
                t_seq=t_seq,
                test=test,
                Uic=Uic,
                Vic=Vic,
            )

        # Surface the first dead-step index so callers know where the
        # trajectory froze if the blowup gate tripped. -1 means no blowup.
        # Computation is JAX-friendly so this stays cheap inside jit/grad.
        dead_first_idx = _first_dead_index(outs.get("dead"))

        return dict(
            static=static,
            t_seq=t_seq,
            outs=outs,
            last_state=last_state,
            starttime=starttime_eff,
            dead_first_idx=dead_first_idx,
        )

    # Final-only path: do not materialize the full trajectory.
    if jit_scan:
        simulate_fn = _get_simulate_scan_last_jit(test=test, donate_state=donate_eff, remat_step=bool(remat_step))
        last_state = simulate_fn(state0, t_seq, static, flags, Uic, Vic)
    else:
        last_state = simulate_scan_last(
            static=static,
            flags=flags,
            state0=state0,
            t_seq=t_seq,
            test=test,
            Uic=Uic,
            Vic=Vic,
            remat_step=bool(remat_step),
        )

    return dict(
        static=static,
        t_seq=t_seq,
        last_state=last_state,
        starttime=starttime_eff,
    )


def _first_dead_index(dead: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Return the first scan-step index at which `dead` is True, or -1.

    Useful when ``flags.diagnostics=True`` and the blowup gate may trip during
    a scan. The scan output array ``outs["dead"]`` has shape ``(len(t_seq),)``
    and is monotonic non-decreasing in boolean value.

    Parameters
    ----------
    dead : Optional[jnp.ndarray]
        Boolean per-step array of length ``len(t_seq)``. ``None`` is permitted
        for callers that did not record the diagnostic.

    Returns
    -------
    jnp.ndarray
        Scalar int32 index. ``-1`` if no step is dead (or ``dead`` is None).
    """
    if dead is None:
        return jnp.asarray(-1, dtype=jnp.int32)
    dead_arr = jnp.asarray(dead, dtype=jnp.bool_)
    n = dead_arr.shape[0]
    if n == 0:
        return jnp.asarray(-1, dtype=jnp.int32)
    # argmax returns the first True; if no True exists argmax returns 0,
    # so cross-check with .any() and substitute -1 in that case.
    first = jnp.argmax(dead_arr).astype(jnp.int32)
    return jnp.where(jnp.any(dead_arr), first, jnp.int32(-1))


def assert_finite_state(last_state: "State", *, raise_on_nan: bool = True) -> bool:
    """Assert that a terminal `State` has no NaN/Inf entries in its physical fields.

    This is intended as a final reliability check after `run_model_scan_final`
    (or `run_model_scan(..., return_history=False)`) when ``diagnostics=False``,
    where the in-scan blowup gate is bypassed. Callers should invoke this after
    the scan completes (host-side) to catch silent NaN propagation.

    Parameters
    ----------
    last_state : State
        Terminal scan carry returned by the scan driver.
    raise_on_nan : bool
        If True (default), raise ``RuntimeError`` on detection. If False,
        return ``False`` instead.

    Returns
    -------
    bool
        True when the state is finite. False (only when ``raise_on_nan=False``)
        when the state contains NaN/Inf.
    """
    fields = (
        ("eta_curr", last_state.eta_curr),
        ("delta_curr", last_state.delta_curr),
        ("Phi_curr", last_state.Phi_curr),
        ("U_curr", last_state.U_curr),
        ("V_curr", last_state.V_curr),
    )
    bad = []
    for name, arr in fields:
        if not bool(jnp.all(jnp.isfinite(arr))):
            bad.append(name)
    if bad:
        if raise_on_nan:
            raise RuntimeError(
                "Final state contains non-finite values in fields: "
                + ", ".join(bad)
                + ". This usually indicates the integration blew up. Consider "
                "enabling diagnostics=True to gate on RMS-wind blowup, or "
                "reducing dt / increasing K6."
            )
        return False
    return True


def run_model_scan_final(
    *,
    M: int,
    dt: Scalar,
    tmax: int,
    Phibar: Scalar,
    omega: Scalar,
    a: Scalar,
    test: Optional[int] = None,
    g: Scalar = 9.8,
    forcflag: bool = True,
    taurad: Scalar = 86400.0,
    taudrag: Scalar = 86400.0,
    DPhieq: Scalar = 4 * (10**6),
    a1: Scalar = 0.05,
    diffflag: bool = True,
    modalflag: bool = True,
    alpha: Scalar = 0.01,
    expflag: bool = False,
    K6: Scalar = 1.24 * (10**33),
    K6Phi: Optional[Scalar] = None,
    contflag: bool = False,
    custompath: Optional[str] = None,
    contTime: Optional[str] = None,
    timeunits: str = "hours",
    starttime: Optional[int] = None,
    # Optional: provide explicit initial state (enables differentiating wrt ICs).
    eta0_init: Optional[jnp.ndarray] = None,
    delta0_init: Optional[jnp.ndarray] = None,
    Phi0_init: Optional[jnp.ndarray] = None,
    U0_init: Optional[jnp.ndarray] = None,
    V0_init: Optional[jnp.ndarray] = None,
    # Performance knobs
    diagnostics: bool = False,
    remat_step: bool = False,
    jit_scan: bool = True,
    donate_state: bool = False,
) -> Dict[str, Any]:
    """Run the model but return only the terminal state (no time history).
    
    This is the recommended entrypoint for optimization/inference and forward-mode
    autodiff (JVP/Jacobian-vector products), where you typically need only the
    final `Phi_curr` (temperature map) and a scalar loss.
    
    See also
    --------
    run_model_scan : full-history scan (plotting / diagnostics)
    
    Parameters
    ----------
    M : int
    dt : Scalar
    tmax : int
    Phibar : Scalar
    omega : Scalar
    a : Scalar
    test : Optional[int]
    g : Scalar
    forcflag : bool
    taurad : Scalar
    taudrag : Scalar
    DPhieq : Scalar
    a1 : Scalar
    diffflag : bool
    modalflag : bool
    alpha : Scalar
    expflag : bool
    K6 : Scalar
    K6Phi : Optional[Scalar]
    contflag : bool
    custompath : Optional[str]
    contTime : Optional[str]
    timeunits : str
    starttime : Optional[int]
    eta0_init : Optional[jnp.ndarray]
    delta0_init : Optional[jnp.ndarray]
    Phi0_init : Optional[jnp.ndarray]
    U0_init : Optional[jnp.ndarray]
    V0_init : Optional[jnp.ndarray]
    diagnostics : bool
    remat_step : bool
    jit_scan : bool
    donate_state : bool
    
    Returns
    -------
    Dict[str, Any]
    """

    return run_model_scan(
        M=M,
        dt=dt,
        tmax=tmax,
        Phibar=Phibar,
        omega=omega,
        a=a,
        test=test,
        g=g,
        forcflag=forcflag,
        taurad=taurad,
        taudrag=taudrag,
        DPhieq=DPhieq,
        a1=a1,
        diffflag=diffflag,
        modalflag=modalflag,
        alpha=alpha,
        expflag=expflag,
        K6=K6,
        K6Phi=K6Phi,
        contflag=contflag,
        custompath=custompath,
        contTime=contTime,
        timeunits=timeunits,
        starttime=starttime,
        eta0_init=eta0_init,
        delta0_init=delta0_init,
        Phi0_init=Phi0_init,
        U0_init=U0_init,
        V0_init=V0_init,
        diagnostics=diagnostics,
        return_history=False,
        remat_step=remat_step,
        jit_scan=jit_scan,
        donate_state=donate_state,
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
    taurad: float = 86400,
    taudrag: float = 86400,
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
    K6: float = 1.24 * 10**33,
    custompath: Optional[str] = None,
    contTime: Optional[str] = None,
    timeunits: str = "hours",
    verbose: bool = True,
    *,
    K6Phi: Optional[float] = None,
    # Performance knobs
    jit_scan: bool = True,
    as_numpy: bool = True,
) -> Dict[str, Any]:
    """Compatibility wrapper matching the original SWAMPE `model.run_model` signature.
    
    Notes
    -----
    - The differentiable core is `run_model_scan(...)`.
    - Plotting/saving are done after the scan (no side effects in the core).
    
    Parameters
    ----------
    M : int
    dt : float
    tmax : int
    Phibar : float
    omega : float
    a : float
    test : Optional[int]
    g : float
    forcflag : bool
    taurad : float
    taudrag : float
    DPhieq : float
    a1 : float
    plotflag : bool
    plotfreq : int
    minlevel : Optional[float]
    maxlevel : Optional[float]
    diffflag : bool
    modalflag : bool
    alpha : float
    contflag : bool
    saveflag : bool
    expflag : bool
    savefreq : int
    K6 : float
    custompath : Optional[str]
    contTime : Optional[str]
    timeunits : str
    verbose : bool
    K6Phi : Optional[float]
    jit_scan : bool
    as_numpy : bool
    
    Returns
    -------
    Dict[str, Any]
    """

    result = run_model_scan(
        M=M,
        dt=dt,
        tmax=tmax,
        Phibar=Phibar,
        omega=omega,
        a=a,
        test=test,
        g=g,
        forcflag=forcflag,
        taurad=taurad,
        taudrag=taudrag,
        DPhieq=DPhieq,
        a1=a1,
        diffflag=diffflag,
        modalflag=modalflag,
        alpha=alpha,
        expflag=expflag,
        K6=K6,
        K6Phi=K6Phi,
        contflag=contflag,
        custompath=custompath,
        contTime=contTime,
        timeunits=timeunits,
        jit_scan=jit_scan,
    )

    static: Static = result["static"]
    t_seq_j = result["t_seq"]
    outs: Dict[str, Any] = result["outs"]

    # If the caller wants NumPy (legacy behavior) or any Python-side output
    # (saving/plotting), materialize histories on host.
    need_host = bool(as_numpy or saveflag or plotflag)
    if need_host:
        t_seq = np.asarray(t_seq_j)
        eta_hist = np.asarray(outs["eta"])
        delta_hist = np.asarray(outs["delta"])
        Phi_hist = np.asarray(outs["Phi"])
        U_hist = np.asarray(outs["U"])
        V_hist = np.asarray(outs["V"])

        rms_hist = np.asarray(outs["rms"])
        spin_min_hist = np.asarray(outs["spin_min"])
        phi_min_hist = np.asarray(outs["phi_min"])
        phi_max_hist = np.asarray(outs["phi_max"])
    else:
        t_seq = t_seq_j
        eta_hist = outs["eta"]
        delta_hist = outs["delta"]
        Phi_hist = outs["Phi"]
        U_hist = outs["U"]
        V_hist = outs["V"]

        rms_hist = outs["rms"]
        spin_min_hist = outs["spin_min"]
        phi_min_hist = outs["phi_min"]
        phi_max_hist = outs["phi_max"]

    # Reconstruct "long arrays" in the legacy shape (tmax,2), filled where defined.
    if need_host:
        spinupdata = np.zeros((tmax, 2), dtype=float)
        geopotdata = np.zeros((tmax, 2), dtype=float)

        # Fill diagnostics for indices (t-1) where t is in t_seq.
        for k, t in enumerate(t_seq):
            idx = int(t) - 1
            if 0 <= idx < tmax:
                spinupdata[idx, 0] = float(spin_min_hist[k])
                spinupdata[idx, 1] = float(rms_hist[k])
                geopotdata[idx, 0] = float(phi_min_hist[k])
                geopotdata[idx, 1] = float(phi_max_hist[k])
    else:
        # Pure JAX path (no host transfer): scatter the diagnostics into the
        # legacy (tmax,2) arrays.
        idx = t_seq_j - 1
        spinupdata = jnp.zeros((tmax, 2), dtype=float_dtype())
        geopotdata = jnp.zeros((tmax, 2), dtype=float_dtype())

        spinupdata = spinupdata.at[idx, 0].set(spin_min_hist)
        spinupdata = spinupdata.at[idx, 1].set(rms_hist)
        geopotdata = geopotdata.at[idx, 0].set(phi_min_hist)
        geopotdata = geopotdata.at[idx, 1].set(phi_max_hist)

    # Match SWAMPE: when *not* continuing from saved data, populate the
    # initial diagnostics at index 0 from the analytic initial conditions.
    if not contflag:
        _, _, Phi0_init_local, U0_init_local, V0_init_local = _analytic_ic(static, test, a1)

        wind0 = jnp.sqrt(U0_init_local * U0_init_local + V0_init_local * V0_init_local)
        spin0 = jnp.min(wind0)
        rms0 = time_stepping.RMS_winds(
            static.a,
            static.I,
            static.J,
            static.lambdas,
            static.mus,
            U0_init_local,
            V0_init_local,
        )
        phi_min0 = jnp.min(Phi0_init_local)
        phi_max0 = jnp.max(Phi0_init_local)

        if need_host:
            spinupdata[0, 0] = float(np.asarray(spin0))
            spinupdata[0, 1] = float(np.asarray(rms0))
            geopotdata[0, 0] = float(np.asarray(phi_min0))
            geopotdata[0, 1] = float(np.asarray(phi_max0))
        else:
            spinupdata = spinupdata.at[0, 0].set(spin0)
            spinupdata = spinupdata.at[0, 1].set(rms0)
            geopotdata = geopotdata.at[0, 0].set(phi_min0)
            geopotdata = geopotdata.at[0, 1].set(phi_max0)

    # Optional saving/plotting (legacy behavior).
    if saveflag:
        for k, t in enumerate(t_seq):
            if int(t) % int(savefreq) == 0:
                # FIXED: compute_timestamp signature is (units, t, dt)
                timestamp = continuation.compute_timestamp(timeunits, int(t), dt)
                continuation.save_data(
                    timestamp,
                    eta_hist[k],
                    delta_hist[k],
                    Phi_hist[k],
                    U_hist[k],
                    V_hist[k],
                    spinupdata,
                    geopotdata,
                    custompath=custompath,
                )

    if plotflag:
        # Lazy import: plotting pulls in matplotlib/imageio, which is expensive
        # and unnecessary for headless / HPC runs.
        from . import plotting
        for k, t in enumerate(t_seq):
            if int(t) % int(plotfreq) == 0:
                timestamp = continuation.compute_timestamp(timeunits, int(t), dt)
                plotting.mean_zonal_wind_plot(U_hist[k], np.asarray(static.mus), timestamp, units=timeunits)
                plotting.quiver_geopot_plot(
                    U_hist[k],
                    V_hist[k],
                    Phi_hist[k] + float(Phibar),
                    np.asarray(static.lambdas),
                    np.asarray(static.mus),
                    timestamp,
                    units=timeunits,
                    minlevel=minlevel,
                    maxlevel=maxlevel,
                )
                plotting.spinup_plot(spinupdata, float(dt), units=timeunits)

    if verbose:
        print("GCM run completed!")

    return dict(
        eta=eta_hist[-1] if eta_hist.shape[0] else None,
        delta=delta_hist[-1] if delta_hist.shape[0] else None,
        Phi=Phi_hist[-1] if Phi_hist.shape[0] else None,
        U=U_hist[-1] if U_hist.shape[0] else None,
        V=V_hist[-1] if V_hist.shape[0] else None,
        spinup=spinupdata,
        geopot=geopotdata,
        lambdas=np.asarray(static.lambdas) if need_host else static.lambdas,
        mus=np.asarray(static.mus) if need_host else static.mus,
        t_seq=t_seq,
    )


def run_model_gpu(*args, **kwargs) -> Dict[str, Any]:
    """GPU/AD-friendly wrapper around :func:`run_model`.
    
    This preserves the legacy default behavior of :func:`run_model` (plotting,
    saving, and host materialization) while providing a convenience entrypoint
    with performance-oriented defaults.
    
    Defaults applied when not explicitly provided by the caller:
      - plotflag=False
      - saveflag=False
      - as_numpy=False
      - jit_scan=True
    
    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to the wrapped callable.
    **kwargs : Any
        Keyword arguments forwarded to the wrapped callable.
    
    Returns
    -------
    Dict[str, Any]
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("plotflag", False)
    kwargs.setdefault("saveflag", False)
    kwargs.setdefault("as_numpy", False)
    kwargs.setdefault("jit_scan", True)
    return run_model(*args, **kwargs)
