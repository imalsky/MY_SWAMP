#!/usr/bin/env python3
"""pipeline.py

Importable, config-parameterized core of the differentiable SWAMP -> starry
phase-curve retrieval. This module factors the forward model, the starry
projector, the u-space parameterization, the likelihood/prior, and the BlackJAX
adaptive-tempered-SMC machinery out of the monolithic ``run_smc.py`` driver so
that every stage can be unit-tested and reused without running a full inference.

Numerics here are a faithful extraction of the proven ``run_smc.py`` forward
model; ``run_smc.py`` is now a thin driver around :func:`build_pipeline` plus the
inference/observation helpers defined here.

x64 / precision
---------------
JAX's x64 flag is process-global and must be set *before* JAX (and ``my_swamp``)
import. This module reads ``SWAMPE_JAX_ENABLE_X64`` from the environment at import
time and configures JAX accordingly. ``float_dtype()`` keys off the *actual* JAX
state, not the Config, so there is never a silent mismatch. To run in float64,
set ``SWAMPE_JAX_ENABLE_X64=1`` (and ``JAX_ENABLE_X64=1``) in the environment
before importing this module; :func:`build_pipeline` asserts the live JAX state
matches ``cfg.use_x64`` and raises a clear error otherwise.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- repo / retrieval layout (this file lives in retrieval/scripts/) ----------
_SCRIPTS_DIR = Path(__file__).resolve().parent           # retrieval/scripts
RETRIEVAL_ROOT = _SCRIPTS_DIR.parent                     # retrieval/
REPO_ROOT = RETRIEVAL_ROOT.parent                        # MY_SWAMP/
DATA_DIR = RETRIEVAL_ROOT / "data"                       # outputs (npz, config, logs)
PLOTS_DIR = RETRIEVAL_ROOT / "plots"                     # figures
STYLE_FILE = _SCRIPTS_DIR / "science.mplstyle"           # publication style guide

# --- use THIS working tree's my_swamp, not a stale pip-installed copy ---------
# The project conda env may have an older my_swamp installed in site-packages
# (different build_static signature, hardcoded float64). Prepend the repo src/
# so the current, differentiable, x64-aware package always wins. Mirrors run.sh's
# PYTHONPATH and gcmulator's conftest.
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# --- x64 / backend setup BEFORE importing jax + my_swamp ---------------------
_X64_ENV = os.environ.get("SWAMPE_JAX_ENABLE_X64", "0").strip().lower() not in ("0", "false", "no", "")
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1" if _X64_ENV else "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", _X64_ENV)

import my_swamp.model as swamp_model
from my_swamp.model import RunFlags, build_static

from jaxoplanet.orbits.keplerian import Body, Central
from jaxoplanet.starry.light_curves import light_curve as starry_light_curve
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

logger = logging.getLogger("swamp_retrieval")

RJUP_TO_RSUN = 0.10045


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Static configuration for one retrieval run.

    Identical field set to the original ``run_smc.py`` Config; defaults are the
    production (50-day, float64-capable) values. Fast local-CPU presets are built
    by :func:`fast_cpu_config`.
    """

    # I/O & reproducibility
    out_dir: Path = Path("swamp_jaxoplanet_retrieval_outputs")
    seed: int = 7
    log_level: str = "INFO"
    overwrite: bool = True

    # Numeric precision / XLA behavior
    use_x64: bool = False
    xla_preallocate: bool = False

    # SWAMP numerical params (SHAPES) - NOT inferred
    M: int = 42
    dt_seconds: float = 240.0
    model_days: float = 50.0
    starttime_index: int = 2

    # SWAMP physical params (defaults / truth)
    a_planet_m: float = 8.2e7
    omega_rad_s: float = 3.2e-5
    g_m_s2: float = 9.8
    Phibar: float = 3.0e5
    DPhieq: float = 1.0e6
    K6: float = 1.24e33
    K6Phi: Optional[float] = None

    force_rebuild_static: bool = False

    # RunFlags
    forcflag: bool = True
    diffflag: bool = True
    expflag: bool = False
    modalflag: bool = True
    alpha: float = 0.01
    diagnostics: bool = False
    blowup_rms: float = 1.0e30

    # Phi -> temperature -> intensity (emission layer)
    # emission_temp_mode:
    #   "geopotential" (default; matches the SWAMPE-JAX paper / Perez-Becker 2013):
    #        T = (Phibar + Phi) / R_d,  R_d = 3.78e3 J/kg/K. The model's physical
    #        temperature proxy; verified to recover both timescales (strong signal).
    #   "linear": T = T_ref + Phi / phi_to_T_scale (a tunable toy; T_ref=1000,
    #        phi_to_T_scale=600 give a ~1000-2500 K hot-Jupiter-like map).
    emission_temp_mode: str = "geopotential"  # {"geopotential", "linear"}
    R_d: float = 3.78e3              # specific gas constant (geopotential mode)
    T_ref: float = 1000.0           # linear mode only
    phi_to_T_scale: float = 600.0   # linear mode only
    Tmin_K: float = 1.0
    emission_model: str = "bolometric"
    planck_wavelength_m: float = 4.5e-6
    planck_x_clip: float = 80.0

    # starry / map projection
    ydeg: int = 10
    projector_ridge: float = 1.0e-6
    map_inc_rad: float = math.pi / 2
    map_obl_rad: float = 0.0
    phase_at_transit_rad: float = math.pi

    # Orbit/system geometry
    star_mass_msun: float = 1.0
    star_radius_rsun: float = 1.0
    planet_radius_rjup: float = 1.0
    impact_param: float = 0.0
    time_transit_days: float = 0.0
    orbital_period_override_days: Optional[float] = None
    planet_fpfs: float = 1500e-6

    # Synthetic observations
    generate_synthetic_data: bool = True
    n_times: int = 250
    n_orbits_observed: float = 1.0
    # Noise model for the synthetic phase-curve points:
    #   "white"  -> constant per-point sigma = obs_sigma (Gaussian)
    #   "photon" -> heteroscedastic photon noise: sigma_i = sigma_phot / sqrt(F_tot_i),
    #               with total relative flux F_tot_i = 1 (star) + planet flux_i, so brighter
    #               (dayside) points carry more photons and slightly smaller relative error.
    #               sigma_i is computed once from the truth and held fixed in the likelihood.
    noise_model: str = "white"  # {"white", "photon"}
    obs_sigma: float = 80e-6     # used when noise_model == "white"
    sigma_phot: float = 50e-6    # photon-floor per-point fractional noise at unit flux
    taurad_true_hours: float = 10.0
    taudrag_true_hours: float = 6.0

    # Inference toggles
    infer_tau_rad: bool = True
    infer_tau_drag: bool = True
    infer_planet_radius: bool = False
    infer_planet_fpfs: bool = False
    infer_Phibar: bool = False
    infer_DPhieq: bool = False
    infer_K6: bool = False
    infer_K6Phi: bool = False
    infer_omega: bool = False
    infer_a_planet: bool = False
    infer_g: bool = False

    # Priors
    # Timescales span >1 decade; log-uniform is the standard scale-parameter prior
    # (Vasist+2023; petitRADTRANS/POSEIDON convention). Set to "uniform" for a
    # linear-in-hours prior instead.
    prior_type_tau_rad: str = "log10_uniform"
    prior_type_tau_drag: str = "log10_uniform"
    prior_tau_rad_hours_min: float = 1.0
    prior_tau_rad_hours_max: float = 30.0
    prior_tau_drag_hours_min: float = 1.00
    prior_tau_drag_hours_max: float = 30.0
    prior_planet_radius_rjup_min: float = 0.3
    prior_planet_radius_rjup_max: float = 2.0
    prior_planet_fpfs_min: float = 100e-6
    prior_planet_fpfs_max: float = 5000e-6
    prior_Phibar_min: float = 1.0e5
    prior_Phibar_max: float = 1.0e6
    prior_DPhieq_min: float = 1.0e5
    prior_DPhieq_max: float = 5.0e6
    prior_K6_min: float = 1.0e31
    prior_K6_max: float = 1.0e35
    prior_K6Phi_min: float = 0.0
    prior_K6Phi_max: float = 1.0e34
    prior_omega_min: float = 1.0e-6
    prior_omega_max: float = 1.0e-4
    prior_a_planet_min: float = 3.0e7
    prior_a_planet_max: float = 2.0e8
    prior_g_min: float = 1.0
    prior_g_max: float = 40.0

    prior_type_planet_radius: str = "uniform"
    prior_type_planet_fpfs: str = "log10_uniform"
    prior_type_Phibar: str = "log10_uniform"
    prior_type_DPhieq: str = "log10_uniform"
    prior_type_K6: str = "log10_uniform"
    prior_type_K6Phi: str = "log10_uniform"
    prior_type_omega: str = "log10_uniform"
    prior_type_a_planet: str = "log10_uniform"
    prior_type_g: str = "log10_uniform"

    # Inference: BlackJAX Adaptive Tempered SMC
    run_inference: bool = True
    smc_num_particles: int = 64
    smc_target_ess_frac: float = 0.6
    smc_num_mcmc_steps: int = 32
    smc_mcmc_kernel: str = "mala"
    mala_step_size: float = 0.2
    hmc_step_size: float = 0.07
    hmc_num_integration_steps: int = 8

    # Optional auto-tuning of MCMC step size
    mcmc_auto_tune: bool = True
    mcmc_tune_beta: float = 0.5
    mcmc_tune_particles: int = 8
    mcmc_tune_steps: int = 8
    mcmc_tune_iters: int = 8
    mcmc_target_accept_mala: float = 0.75
    mcmc_target_accept_hmc: float = 0.80
    mcmc_step_size_min: float = 1.0e-3
    mcmc_step_size_max: float = 5.0
    mcmc_tune_gain: float = 0.7

    smc_resampling: str = "systematic"
    smc_max_steps: int = 32
    smc_use_custom_gradients: bool = True
    smc_custom_grad_max_dim: int = 8

    num_samples: int = 64
    num_chains: int = 2

    # Posterior predictive
    do_ppc: bool = True
    ppc_draws: int = 128
    ppc_chunk_size: int = 16

    # Plot config
    fig_dpi: int = 160
    render_res: int = 250
    render_phases: Tuple[float, ...] = (0.0, 0.25, 0.49, 0.51, 0.75)
    log_axis_orders_threshold: float = 3.0


_VALID_EMISSION = {"bolometric", "planck"}
_VALID_KERNELS = {"mala", "hmc"}
_VALID_RESAMPLING = {"systematic", "stratified", "multinomial"}


def validate_config(cfg: Config) -> None:
    """Fail-fast validation of a Config (ported from run_smc.py)."""
    if str(cfg.emission_model).strip().lower() not in _VALID_EMISSION:
        raise ValueError(f"cfg.emission_model must be one of {_VALID_EMISSION}, got {cfg.emission_model!r}")
    if str(cfg.smc_mcmc_kernel).strip().lower() not in _VALID_KERNELS:
        raise ValueError(f"cfg.smc_mcmc_kernel must be one of {_VALID_KERNELS}, got {cfg.smc_mcmc_kernel!r}")
    if str(cfg.smc_resampling).strip().lower() not in _VALID_RESAMPLING:
        raise ValueError(f"cfg.smc_resampling must be one of {_VALID_RESAMPLING}, got {cfg.smc_resampling!r}")
    if str(cfg.noise_model).strip().lower() not in {"white", "photon"}:
        raise ValueError(f"cfg.noise_model must be 'white' or 'photon', got {cfg.noise_model!r}")
    if str(cfg.emission_temp_mode).strip().lower() not in {"geopotential", "linear"}:
        raise ValueError(f"cfg.emission_temp_mode must be 'geopotential' or 'linear', got {cfg.emission_temp_mode!r}")
    if cfg.model_days <= 0:
        raise ValueError("cfg.model_days must be > 0")
    if cfg.dt_seconds <= 0:
        raise ValueError("cfg.dt_seconds must be > 0")
    if cfg.starttime_index < 2:
        raise ValueError("cfg.starttime_index must be >= 2 for leapfrog startup")
    if cfg.smc_num_particles <= 0:
        raise ValueError("cfg.smc_num_particles must be > 0")
    if cfg.num_chains <= 0 or cfg.num_samples <= 0:
        raise ValueError("cfg.num_chains and cfg.num_samples must be > 0")


# =============================================================================
# Small utilities
# =============================================================================


def float_dtype() -> Any:
    """JAX float dtype keyed off the *live* x64 state (not the Config)."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def np_float_dtype() -> Any:
    """NumPy dtype matching :func:`float_dtype`."""
    return np.float64 if jax.config.jax_enable_x64 else np.float32


def tau_hours_to_seconds(x_hours: Any) -> Any:
    """Convert a timescale from hours to seconds."""
    return 3600.0 * x_hours


def orbital_period_days_from_omega(omega_rad_s: Any) -> Any:
    """Synchronous period implied by omega (can be a JAX scalar)."""
    return (2.0 * jnp.pi / omega_rad_s) / 86400.0


def compute_n_steps(model_days: float, dt_seconds: float) -> int:
    """Convert a physical duration into an integer solver step count."""
    n = int(np.round(model_days * 86400.0 / dt_seconds))
    return max(n, 1)


def save_npz(path: Path, **arrays: Any) -> None:
    """Save named arrays into a compressed ``.npz`` archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def call_with_filtered_kwargs(func, kwargs: Dict[str, Any], *, name: Optional[str] = None):
    """Call ``func(**kwargs)`` dropping kwargs the signature doesn't accept.

    Supports multiple my_swamp / jaxoplanet versions whose signatures differ.
    """
    fn_name = name or getattr(func, "__name__", repr(func))
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(**kwargs)
    filtered: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered[k] = v
        else:
            dropped.append(k)
    if dropped:
        logger.warning(f"{fn_name}: dropped unexpected kwargs: {dropped}")
    return func(**filtered)


# =============================================================================
# Parameter registry
# =============================================================================


@dataclass(frozen=True)
class ParamSpec:
    """A single inferred parameter specification (see run_smc.py for the math)."""

    name: str
    label: str
    prior_type: str  # {"uniform", "log10_uniform"}
    lo: float
    hi: float
    truth: float


def specs_from_config(cfg: Config) -> List[ParamSpec]:
    """Build the active parameter list based on cfg.infer_* toggles."""
    specs: List[ParamSpec] = []

    def add(name: str, label: str, prior_type: str, lo: float, hi: float, truth: float) -> None:
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"Invalid prior bounds for {name}: lo={lo}, hi={hi}")
        if prior_type not in {"uniform", "log10_uniform"}:
            raise ValueError(f"Invalid prior_type for {name}: {prior_type!r}")
        if prior_type == "log10_uniform" and (lo <= 0.0 or hi <= 0.0):
            raise ValueError(
                f"log10_uniform prior requires strictly positive bounds for {name}: lo={lo}, hi={hi}."
            )
        specs.append(ParamSpec(name=name, label=label, prior_type=prior_type, lo=lo, hi=hi, truth=truth))

    if cfg.infer_tau_rad:
        add("tau_rad_hours", "tau_rad [h]", str(cfg.prior_type_tau_rad).strip().lower(),
            float(cfg.prior_tau_rad_hours_min), float(cfg.prior_tau_rad_hours_max), float(cfg.taurad_true_hours))
    if cfg.infer_tau_drag:
        add("tau_drag_hours", "tau_drag [h]", str(cfg.prior_type_tau_drag).strip().lower(),
            float(cfg.prior_tau_drag_hours_min), float(cfg.prior_tau_drag_hours_max), float(cfg.taudrag_true_hours))
    if cfg.infer_planet_radius:
        add("planet_radius_rjup", "R_p [Rjup]", str(cfg.prior_type_planet_radius).strip().lower(),
            float(cfg.prior_planet_radius_rjup_min), float(cfg.prior_planet_radius_rjup_max), float(cfg.planet_radius_rjup))
    if cfg.infer_planet_fpfs:
        add("planet_fpfs", "F_p/F_s", str(cfg.prior_type_planet_fpfs).strip().lower(),
            float(cfg.prior_planet_fpfs_min), float(cfg.prior_planet_fpfs_max), float(cfg.planet_fpfs))
    if cfg.infer_Phibar:
        add("Phibar", "Phibar", str(cfg.prior_type_Phibar).strip().lower(),
            float(cfg.prior_Phibar_min), float(cfg.prior_Phibar_max), float(cfg.Phibar))
    if cfg.infer_DPhieq:
        add("DPhieq", "DPhieq", str(cfg.prior_type_DPhieq).strip().lower(),
            float(cfg.prior_DPhieq_min), float(cfg.prior_DPhieq_max), float(cfg.DPhieq))
    if cfg.infer_K6:
        add("K6", "K6", str(cfg.prior_type_K6).strip().lower(),
            float(cfg.prior_K6_min), float(cfg.prior_K6_max), float(cfg.K6))
    if cfg.infer_K6Phi:
        truth_k6phi = 0.0 if cfg.K6Phi is None else float(cfg.K6Phi)
        add("K6Phi", "K6Phi", str(cfg.prior_type_K6Phi).strip().lower(),
            float(cfg.prior_K6Phi_min), float(cfg.prior_K6Phi_max), truth_k6phi)
    if cfg.infer_omega:
        add("omega_rad_s", "omega [rad/s]", str(cfg.prior_type_omega).strip().lower(),
            float(cfg.prior_omega_min), float(cfg.prior_omega_max), float(cfg.omega_rad_s))
    if cfg.infer_a_planet:
        add("a_planet_m", "a [m]", str(cfg.prior_type_a_planet).strip().lower(),
            float(cfg.prior_a_planet_min), float(cfg.prior_a_planet_max), float(cfg.a_planet_m))
    if cfg.infer_g:
        add("g_m_s2", "g [m/s^2]", str(cfg.prior_type_g).strip().lower(),
            float(cfg.prior_g_min), float(cfg.prior_g_max), float(cfg.g_m_s2))

    if len(specs) == 0:
        raise ValueError("No parameters enabled for inference. Set at least one cfg.infer_* = True.")
    return specs


# =============================================================================
# Pipeline container
# =============================================================================


class Pipeline:
    """Holds all built artifacts and JAX-traceable functions for one Config.

    Attributes are assigned by :func:`build_pipeline`. The important callables:
      - ``phase_curve_model(theta)`` / ``phase_curve_model_jit``
      - ``swamp_terminal_phi(...)``  (terminal Phi map)
      - ``theta_from_u(u)`` / ``log_prior_u(u)`` / ``sample_prior_u(key, n)``
      - ``log_likelihood_u(u)`` / ``loglikelihood_for_blackjax``
      - ``compute_maps_for_theta(theta)``
    """

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def build_pipeline(cfg: Config) -> Pipeline:
    """Build the forward model, projector, u-space transform, prior and likelihood.

    Pure-ish: no file IO, no inference. Safe to call from tests. Requires that the
    live JAX x64 state matches ``cfg.use_x64`` (set the env var before import).
    """
    validate_config(cfg)
    if bool(jax.config.jax_enable_x64) != bool(cfg.use_x64):
        raise RuntimeError(
            f"JAX x64 state ({bool(jax.config.jax_enable_x64)}) != cfg.use_x64 ({bool(cfg.use_x64)}). "
            "Set SWAMPE_JAX_ENABLE_X64 (and JAX_ENABLE_X64) in the environment BEFORE importing pipeline."
        )

    dtype = float_dtype()
    npdtype = np_float_dtype()

    specs = specs_from_config(cfg)
    param_names = [s.name for s in specs]
    param_labels = [s.label for s in specs]
    param_prior_lo = np.array([s.lo for s in specs], dtype=np.float64)
    param_prior_hi = np.array([s.hi for s in specs], dtype=np.float64)
    param_truth = np.array([s.truth for s in specs], dtype=np.float64)
    n_dim = len(specs)

    flags = call_with_filtered_kwargs(
        RunFlags,
        dict(forcflag=cfg.forcflag, diffflag=cfg.diffflag, expflag=cfg.expflag, modalflag=cfg.modalflag,
             diagnostics=cfg.diagnostics, alpha=float(cfg.alpha), blowup_rms=float(cfg.blowup_rms)),
        name="RunFlags",
    )

    # ---- static builder (jit-safe after eager warm of the geometry cache) ----
    def _build_static_from_values(*, taurad_s, taudrag_s, Phibar, DPhieq, K6, K6Phi, omega, a, g):
        static_kwargs = dict(
            M=int(cfg.M),
            dt=jnp.asarray(cfg.dt_seconds, dtype=dtype),
            a=jnp.asarray(a, dtype=dtype),
            omega=jnp.asarray(omega, dtype=dtype),
            g=jnp.asarray(g, dtype=dtype),
            Phibar=jnp.asarray(Phibar, dtype=dtype),
            taurad=jnp.asarray(taurad_s, dtype=dtype),
            taudrag=jnp.asarray(taudrag_s, dtype=dtype),
            DPhieq=jnp.asarray(DPhieq, dtype=dtype),
            K6=jnp.asarray(K6, dtype=dtype),
            K6Phi=(None if K6Phi is None else jnp.asarray(K6Phi, dtype=dtype)),
            test=None,
        )
        return call_with_filtered_kwargs(build_static, static_kwargs, name="build_static")

    static_base = _build_static_from_values(
        taurad_s=tau_hours_to_seconds(cfg.taurad_true_hours),
        taudrag_s=tau_hours_to_seconds(cfg.taudrag_true_hours),
        Phibar=cfg.Phibar, DPhieq=cfg.DPhieq, K6=cfg.K6, K6Phi=cfg.K6Phi,
        omega=cfg.omega_rad_s, a=cfg.a_planet_m, g=cfg.g_m_s2,
    )

    I = int(getattr(static_base, "I", -1))
    J = int(getattr(static_base, "J", -1))
    n_steps = compute_n_steps(cfg.model_days, cfg.dt_seconds)
    t_seq = jnp.arange(cfg.starttime_index, cfg.starttime_index + n_steps, dtype=jnp.int32)

    # ---- initial conditions ----
    init_fn = getattr(swamp_model, "_init_state_from_fields", None) or getattr(swamp_model, "init_state_from_fields", None)
    if init_fn is None:
        raise RuntimeError("my_swamp.model._init_state_from_fields not found.")

    def init_rest_state(static):
        Jloc, Iloc = int(static.J), int(static.I)
        mus_ = getattr(static, "mus", None)
        omega_ = getattr(static, "omega", 0.0)
        if mus_ is None:
            eta0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
        else:
            mu_ = jnp.asarray(mus_, dtype=dtype)
            omega_ = jnp.asarray(omega_, dtype=dtype)
            eta0 = (2.0 * omega_ * mu_)[:, None] * jnp.ones((Jloc, Iloc), dtype=dtype)
        z = jnp.zeros((Jloc, Iloc), dtype=dtype)
        Phieq = jnp.asarray(getattr(static, "Phieq", 0.0), dtype=dtype)
        Phibar_ = jnp.asarray(getattr(static, "Phibar", 0.0), dtype=dtype)
        Phi0 = Phieq - Phibar_
        return eta0, z, Phi0, z, z

    def build_state0(static):
        eta0, delta0, Phi0, U0, V0 = init_rest_state(static)
        state0 = call_with_filtered_kwargs(
            init_fn,
            dict(static=static, flags=flags, test=None, eta0=eta0, delta0=delta0, Phi0=Phi0, U0=U0, V0=V0),
            name=init_fn.__name__,
        )
        return state0, U0, V0

    state0_base, U0_base, V0_base = build_state0(static_base)

    # ---- emission layer ----
    _temp_mode = str(cfg.emission_temp_mode).strip().lower()

    def _all_finite(x):
        return jnp.all(jnp.isfinite(x))

    def phi_to_temperature(phi):
        phi = jnp.asarray(phi, dtype=dtype)
        finite_phi = _all_finite(phi)

        def _ok(_):
            if _temp_mode == "geopotential":
                # paper / Perez-Becker 2013: T = (Phibar + Phi) / R_d
                T = (jnp.asarray(cfg.Phibar, dtype=dtype) + phi) / jnp.asarray(cfg.R_d, dtype=dtype)
            else:
                T = jnp.asarray(cfg.T_ref, dtype=dtype) + phi / jnp.asarray(cfg.phi_to_T_scale, dtype=dtype)
            return jnp.maximum(T, jnp.asarray(cfg.Tmin_K, dtype=dtype))

        def _bad(_):
            return jnp.full_like(phi, jnp.asarray(jnp.nan, dtype=dtype))

        return jax.lax.cond(finite_phi, _ok, _bad, operand=None)

    def planck_intensity_relative_lambda(T, wavelength_m):
        T = jnp.asarray(T, dtype=dtype)
        finite_T = _all_finite(T)

        def _ok(_):
            lam = jnp.asarray(wavelength_m, dtype=dtype)
            h = jnp.asarray(6.62607015e-34, dtype=dtype)
            c = jnp.asarray(299792458.0, dtype=dtype)
            kB = jnp.asarray(1.380649e-23, dtype=dtype)
            x = (h * c) / (lam * kB * T)
            x = jnp.clip(x, jnp.asarray(0.0, dtype=dtype), jnp.asarray(cfg.planck_x_clip, dtype=dtype))
            tiny = jnp.asarray(1.0e-30, dtype=dtype)
            return jnp.asarray(1.0, dtype=dtype) / (jnp.expm1(x) + tiny)

        def _bad(_):
            return jnp.full_like(T, jnp.asarray(jnp.nan, dtype=dtype))

        return jax.lax.cond(finite_T, _ok, _bad, operand=None)

    emission_mode = str(cfg.emission_model).strip().lower()

    def temperature_to_intensity(T):
        if emission_mode == "bolometric":
            return jnp.asarray(T, dtype=dtype) ** 4
        if emission_mode == "planck":
            return planck_intensity_relative_lambda(jnp.asarray(T, dtype=dtype), float(cfg.planck_wavelength_m))
        raise ValueError(f"Unknown cfg.emission_model={cfg.emission_model!r}.")

    # ---- pixel grid + weights ----
    lambdas = getattr(static_base, "lambdas", None)
    mus = getattr(static_base, "mus", None)
    w_lat = getattr(static_base, "w", None)
    if lambdas is None or mus is None:
        raise RuntimeError("static_base.lambdas and static_base.mus are required to build the starry projector.")

    lon = jnp.asarray(lambdas, dtype=dtype)
    mu = jnp.clip(jnp.asarray(mus, dtype=dtype), jnp.asarray(-1.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    lat = jnp.arcsin(mu)
    lon2d = jnp.broadcast_to(lon[None, :], (lat.shape[0], lon.shape[0]))
    lat2d = jnp.broadcast_to(lat[:, None], (lat.shape[0], lon.shape[0]))
    lon_flat = lon2d.reshape(-1)
    lat_flat = lat2d.reshape(-1)

    if w_lat is None:
        w_pix = jnp.ones_like(lat_flat)
    else:
        w_lat = jnp.asarray(w_lat, dtype=dtype)
        if not np.all(np.isfinite(np.asarray(w_lat))):
            raise RuntimeError("static_base.w contains non-finite values.")
        if np.any(np.asarray(w_lat) < 0.0):
            raise RuntimeError("static_base.w contains negative values.")
        w_pix = jnp.repeat(w_lat, lon.shape[0])
    w_sqrt = jnp.sqrt(w_pix)

    # ---- starry design matrix + LSQ projector ----
    lm_list = [(ell, m) for ell in range(cfg.ydeg + 1) for m in range(-ell, ell + 1)]
    n_coeff = (cfg.ydeg + 1) ** 2
    n_pix = int(lat_flat.shape[0])

    def ylm_from_dense(y_dense, lm_list_):
        return Ylm({lm: y_dense[i] for i, lm in enumerate(lm_list_)})

    def surface_intensity(surf, latv, lonv):
        try:
            sig = inspect.signature(surf.intensity)
            if "theta" in sig.parameters:
                return surf.intensity(latv, lonv, theta=jnp.asarray(0.0, dtype=dtype))
            return surf.intensity(latv, lonv)
        except (TypeError, ValueError):
            return surf.intensity(latv, lonv)

    def _intensity_from_yvec(y_vec):
        ylm = ylm_from_dense(y_vec, lm_list)
        surf = Surface(y=ylm, u=(), inc=jnp.asarray(cfg.map_inc_rad, dtype=dtype),
                       obl=jnp.asarray(cfg.map_obl_rad, dtype=dtype),
                       amplitude=jnp.asarray(1.0, dtype=dtype), normalize=False)
        return surface_intensity(surf, lat_flat, lon_flat)

    _intensity_from_yvec_jit = jax.jit(_intensity_from_yvec)
    eye = jnp.eye(n_coeff, dtype=dtype)
    B = jax.vmap(_intensity_from_yvec_jit)(eye).T  # (n_pix, n_coeff)
    B.block_until_ready()
    if not np.isfinite(np.asarray(B)).all():
        raise RuntimeError("Design matrix B contains NaNs/Infs.")

    Bw = w_sqrt[:, None] * B
    ridge = jnp.asarray(cfg.projector_ridge, dtype=dtype)
    gram = Bw.T @ Bw + ridge * jnp.eye(n_coeff, dtype=dtype)
    projector = jnp.linalg.solve(gram, Bw.T)  # (n_coeff, n_pix)
    projector.block_until_ready()
    if not np.isfinite(np.asarray(projector)).all():
        raise RuntimeError("Projector contains NaNs/Infs.")

    def intensity_map_to_y_dense(I_map):
        I_flat = jnp.asarray(I_map, dtype=dtype).reshape(-1)
        finite_mask = jnp.isfinite(I_flat)
        all_finite_in = jnp.all(finite_mask)
        w_eff = jnp.asarray(w_pix, dtype=dtype) * finite_mask
        w_sum = jnp.sum(w_eff)
        I_safe = jnp.where(finite_mask, I_flat, jnp.asarray(0.0, dtype=dtype))

        def _mean_ok(_):
            return jnp.sum(w_eff * I_safe) / w_sum

        def _mean_bad(_):
            return jnp.asarray(jnp.nan, dtype=dtype)

        I_mean = jax.lax.cond(w_sum > jnp.asarray(0.0, dtype=dtype), _mean_ok, _mean_bad, operand=None)
        eps = jnp.asarray(1.0e-30, dtype=dtype)
        mean_ok = jnp.isfinite(I_mean) & (jnp.abs(I_mean) > eps)
        I_rel = jnp.where(mean_ok, I_safe / I_mean, jnp.asarray(jnp.nan, dtype=dtype))
        rhs = jnp.asarray(w_sqrt, dtype=dtype) * I_rel
        y = jnp.asarray(projector, dtype=dtype) @ rhs
        y0 = y[0]
        y0_ok = jnp.isfinite(y0) & (jnp.abs(y0) > eps)
        y = jnp.where(y0_ok, y / y0, jnp.full_like(y, jnp.asarray(jnp.nan, dtype=dtype)))
        out_ok = all_finite_in & mean_ok & y0_ok & _all_finite(y)
        y = jax.lax.cond(out_ok, lambda _: y, lambda _: jnp.full_like(y, jnp.asarray(jnp.nan, dtype=dtype)), operand=None)
        return y

    # ---- observation times ----
    orbital_period_days_base = (
        float(cfg.orbital_period_override_days)
        if cfg.orbital_period_override_days is not None
        else float((2.0 * math.pi / cfg.omega_rad_s) / 86400.0)
    )
    times_days = np.linspace(
        cfg.time_transit_days, cfg.time_transit_days + cfg.n_orbits_observed * orbital_period_days_base,
        cfg.n_times, endpoint=False,
    ).astype(npdtype)
    times_days_jax = jnp.asarray(times_days, dtype=dtype)

    # ---- SWAMP forward (terminal Phi) ----
    _fast_path_ok = (not cfg.force_rebuild_static) and not (
        cfg.infer_Phibar or cfg.infer_DPhieq or cfg.infer_K6 or cfg.infer_K6Phi
        or cfg.infer_omega or cfg.infer_a_planet or cfg.infer_g
    )

    def swamp_terminal_phi(taurad_s, taudrag_s, *, Phibar, DPhieq, K6, K6Phi, omega, a, g):
        if _fast_path_ok:
            static = replace(static_base, taurad=taurad_s, taudrag=taudrag_s)
            state0, U0, V0 = state0_base, U0_base, V0_base
        else:
            static = _build_static_from_values(
                taurad_s=taurad_s, taudrag_s=taudrag_s, Phibar=Phibar, DPhieq=DPhieq,
                K6=K6, K6Phi=(None if K6Phi is None else K6Phi), omega=omega, a=a, g=g)
            state0, U0, V0 = build_state0(static)

        sim_last = getattr(swamp_model, "simulate_scan_last", None) or getattr(swamp_model, "run_model_scan_final", None)
        if sim_last is not None:
            kwargs = dict(static=static, flags=flags, state0=state0, t_seq=t_seq, test=None,
                          Uic=U0, Vic=V0, remat_step=False, jit_scan=True, return_history=False)
            out = call_with_filtered_kwargs(sim_last, kwargs, name=getattr(sim_last, "__name__", "simulate_scan_last"))
            last_state = out
            if isinstance(out, dict) and "last_state" in out:
                last_state = out["last_state"]
            return getattr(last_state, "Phi_curr")

        step_fn = getattr(swamp_model, "_step_once_state_only", None)
        if step_fn is None:
            raise RuntimeError("No simulate_scan_last/_step_once_state_only in my_swamp.model.")

        def body(i, st):
            return step_fn(st, t_seq[i], static, flags, None, U0, V0)

        state_f = jax.lax.fori_loop(0, int(t_seq.shape[0]), body, state0)
        return getattr(state_f, "Phi_curr")

    # ---- starry phase curve ----
    central = Central(radius=cfg.star_radius_rsun, mass=cfg.star_mass_msun)
    star_surface = Surface(amplitude=jnp.asarray(0.0, dtype=dtype), normalize=False)

    def _theta_vector_to_model_kwargs(theta):
        params = dict(
            tau_rad_hours=jnp.asarray(cfg.taurad_true_hours, dtype=dtype),
            tau_drag_hours=jnp.asarray(cfg.taudrag_true_hours, dtype=dtype),
            planet_radius_rjup=jnp.asarray(cfg.planet_radius_rjup, dtype=dtype),
            planet_fpfs=jnp.asarray(cfg.planet_fpfs, dtype=dtype),
            Phibar=jnp.asarray(cfg.Phibar, dtype=dtype),
            DPhieq=jnp.asarray(cfg.DPhieq, dtype=dtype),
            K6=jnp.asarray(cfg.K6, dtype=dtype),
            K6Phi=(None if cfg.K6Phi is None else jnp.asarray(cfg.K6Phi, dtype=dtype)),
            omega_rad_s=jnp.asarray(cfg.omega_rad_s, dtype=dtype),
            a_planet_m=jnp.asarray(cfg.a_planet_m, dtype=dtype),
            g_m_s2=jnp.asarray(cfg.g_m_s2, dtype=dtype),
        )
        for i, spec in enumerate(specs):
            params[spec.name] = theta[i]
        return params

    def phase_curve_model(theta):
        p = _theta_vector_to_model_kwargs(theta)
        taurad_s = jnp.asarray(tau_hours_to_seconds(p["tau_rad_hours"]), dtype=dtype)
        taudrag_s = jnp.asarray(tau_hours_to_seconds(p["tau_drag_hours"]), dtype=dtype)
        k6phi_val = p["K6Phi"]
        if k6phi_val is not None:
            k6phi_val = jnp.asarray(k6phi_val, dtype=dtype)
        phi = swamp_terminal_phi(
            taurad_s, taudrag_s, Phibar=jnp.asarray(p["Phibar"], dtype=dtype),
            DPhieq=jnp.asarray(p["DPhieq"], dtype=dtype), K6=jnp.asarray(p["K6"], dtype=dtype),
            K6Phi=k6phi_val, omega=jnp.asarray(p["omega_rad_s"], dtype=dtype),
            a=jnp.asarray(p["a_planet_m"], dtype=dtype), g=jnp.asarray(p["g_m_s2"], dtype=dtype))

        def _bad(_):
            return jnp.full((times_days_jax.shape[0],), jnp.asarray(jnp.nan, dtype=dtype), dtype=dtype)

        def _ok(_):
            T = phi_to_temperature(phi)
            I_map = temperature_to_intensity(T)
            y_dense = intensity_map_to_y_dense(I_map)
            ylm = ylm_from_dense(y_dense, lm_list)
            if cfg.orbital_period_override_days is not None:
                orbital_period_days = jnp.asarray(cfg.orbital_period_override_days, dtype=dtype)
            else:
                orbital_period_days = jnp.asarray(orbital_period_days_from_omega(p["omega_rad_s"]), dtype=dtype)
            planet_surface = Surface(
                y=ylm, u=(), inc=jnp.asarray(cfg.map_inc_rad, dtype=dtype),
                obl=jnp.asarray(cfg.map_obl_rad, dtype=dtype), period=orbital_period_days,
                phase=jnp.asarray(cfg.phase_at_transit_rad, dtype=dtype),
                amplitude=jnp.asarray(p["planet_fpfs"], dtype=dtype), normalize=False)
            planet = Body(
                radius=jnp.asarray(p["planet_radius_rjup"], dtype=dtype) * jnp.asarray(RJUP_TO_RSUN, dtype=dtype),
                period=orbital_period_days, time_transit=jnp.asarray(cfg.time_transit_days, dtype=dtype),
                impact_param=jnp.asarray(cfg.impact_param, dtype=dtype))
            system = SurfaceSystem(central=central, central_surface=star_surface, bodies=((planet, planet_surface),))
            lc = starry_light_curve(system)(times_days_jax)
            return lc[:, 1]

        return jax.lax.cond(_all_finite(phi), _ok, _bad, operand=None)

    phase_curve_model_jit = jax.jit(phase_curve_model)

    # ---- u-space parameterization ----
    prior_lo = jnp.asarray(param_prior_lo, dtype=dtype)
    prior_hi = jnp.asarray(param_prior_hi, dtype=dtype)

    def theta_from_u(u):
        u = jnp.asarray(u, dtype=dtype)
        z = jax.nn.sigmoid(u)
        out = []
        for i, spec in enumerate(specs):
            lo, hi, zi = prior_lo[i], prior_hi[i], z[i]
            if spec.prior_type == "uniform":
                out.append(lo + (hi - lo) * zi)
            elif spec.prior_type == "log10_uniform":
                lo_log, hi_log = jnp.log10(lo), jnp.log10(hi)
                out.append(10.0 ** (lo_log + (hi_log - lo_log) * zi))
            else:
                raise ValueError(f"Unknown prior_type {spec.prior_type!r}")
        return jnp.stack(out, axis=0)

    def log_prior_u(u):
        u = jnp.asarray(u, dtype=dtype)
        return jnp.sum(jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))

    def sample_prior_u(rng_key, n_particles):
        eps = jnp.asarray(1e-6, dtype=dtype)
        z = jax.random.uniform(rng_key, shape=(n_particles, n_dim), minval=eps, maxval=1.0 - eps)
        return jnp.log(z) - jnp.log1p(-z)

    theta_truth = jnp.stack([jnp.asarray(s.truth, dtype=dtype) for s in specs], axis=0)

    # ---- likelihood (needs observations injected later via .set_observations) ----
    pipe = Pipeline()

    def log_likelihood_u(u):
        theta = theta_from_u(u)
        mu_pred = phase_curve_model_jit(theta)
        finite = jnp.all(jnp.isfinite(mu_pred))
        flux_obs_jax = pipe.flux_obs_jax
        # Per-point sigma: scalar (white) or vector (heteroscedastic photon noise),
        # broadcast to the data shape so the same code path handles both.
        sig = jnp.broadcast_to(pipe.obs_sigma_jax, mu_pred.shape)

        def _ok():
            resid = (flux_obs_jax - mu_pred) / sig
            n = mu_pred.size
            return (-0.5 * jnp.sum(resid * resid) - jnp.sum(jnp.log(sig))
                    - 0.5 * n * jnp.log(jnp.asarray(2.0 * math.pi, dtype=dtype)))

        def _bad():
            return jnp.asarray(-1.0e30, dtype=dtype)

        return jax.lax.cond(finite, _ok, _bad)

    # custom forward-mode VJP for the likelihood (low-dim, memory stable)
    def _value_and_grad_fwd(fun, x):
        x = jnp.asarray(x)
        n = int(x.shape[0])
        eye_ = jnp.eye(n, dtype=x.dtype)
        y0, dy0 = jax.jvp(fun, (x,), (eye_[0],))
        if n == 1:
            return y0, jnp.atleast_1d(dy0)

        def jvp_dir(v):
            _, dy = jax.jvp(fun, (x,), (v,))
            return dy

        dy_rest = jax.vmap(jvp_dir)(eye_[1:])
        return y0, jnp.concatenate([jnp.atleast_1d(dy0), dy_rest], axis=0)

    use_custom_grads = bool(cfg.smc_use_custom_gradients) and (n_dim <= int(cfg.smc_custom_grad_max_dim))
    if use_custom_grads:
        @jax.custom_vjp
        def log_likelihood_u_for_grad(u):
            return log_likelihood_u(u)

        def _ll_fwd(u):
            val, grad = _value_and_grad_fwd(log_likelihood_u, u)
            return val, grad

        def _ll_bwd(grad, g):
            return (g * grad,)

        log_likelihood_u_for_grad.defvjp(_ll_fwd, _ll_bwd)
        loglikelihood_for_blackjax = log_likelihood_u_for_grad
    else:
        loglikelihood_for_blackjax = log_likelihood_u

    def compute_maps_for_theta(theta):
        p = _theta_vector_to_model_kwargs(theta)
        taurad_s = jnp.asarray(tau_hours_to_seconds(p["tau_rad_hours"]), dtype=dtype)
        taudrag_s = jnp.asarray(tau_hours_to_seconds(p["tau_drag_hours"]), dtype=dtype)
        k6phi_val = p["K6Phi"]
        if k6phi_val is not None:
            k6phi_val = jnp.asarray(k6phi_val, dtype=dtype)
        phi = swamp_terminal_phi(
            taurad_s, taudrag_s, Phibar=jnp.asarray(p["Phibar"], dtype=dtype),
            DPhieq=jnp.asarray(p["DPhieq"], dtype=dtype), K6=jnp.asarray(p["K6"], dtype=dtype),
            K6Phi=k6phi_val, omega=jnp.asarray(p["omega_rad_s"], dtype=dtype),
            a=jnp.asarray(p["a_planet_m"], dtype=dtype), g=jnp.asarray(p["g_m_s2"], dtype=dtype))
        T = phi_to_temperature(phi)
        I_map = temperature_to_intensity(T)
        y_dense = intensity_map_to_y_dense(I_map)
        return {"phi": np.asarray(phi), "T": np.asarray(T), "I": np.asarray(I_map), "y_dense": np.asarray(y_dense)}

    # ---- assemble pipeline ----
    pipe.__dict__.update(dict(
        cfg=cfg, dtype=dtype, npdtype=npdtype,
        specs=specs, param_names=param_names, param_labels=param_labels,
        param_prior_lo=param_prior_lo, param_prior_hi=param_prior_hi, param_truth=param_truth, n_dim=n_dim,
        flags=flags, static_base=static_base, I=I, J=J, n_steps=n_steps, t_seq=t_seq,
        state0_base=state0_base, U0_base=U0_base, V0_base=V0_base,
        lon=lon, lat=lat, lon_flat=lon_flat, lat_flat=lat_flat, w_pix=w_pix, w_sqrt=w_sqrt,
        lm_list=lm_list, n_coeff=n_coeff, n_pix=n_pix, B=B, projector=projector,
        times_days=times_days, times_days_jax=times_days_jax,
        orbital_period_days_base=orbital_period_days_base, _fast_path_ok=_fast_path_ok,
        swamp_terminal_phi=swamp_terminal_phi,
        phi_to_temperature=phi_to_temperature, temperature_to_intensity=temperature_to_intensity,
        intensity_map_to_y_dense=intensity_map_to_y_dense, ylm_from_dense=ylm_from_dense,
        phase_curve_model=phase_curve_model, phase_curve_model_jit=phase_curve_model_jit,
        theta_from_u=theta_from_u, log_prior_u=log_prior_u, sample_prior_u=sample_prior_u,
        log_likelihood_u=log_likelihood_u, loglikelihood_for_blackjax=loglikelihood_for_blackjax,
        use_custom_grads=use_custom_grads, theta_truth=theta_truth,
        compute_maps_for_theta=compute_maps_for_theta,
        # observations (set later)
        flux_obs_jax=None, obs_sigma_jax=jnp.asarray(cfg.obs_sigma, dtype=dtype),
        flux_obs=None, flux_true=None,
    ))

    def set_observations(flux_obs, obs_sigma=None):
        pipe.flux_obs = np.asarray(flux_obs)
        pipe.flux_obs_jax = jnp.asarray(flux_obs, dtype=dtype)
        if obs_sigma is not None:
            pipe.obs_sigma_jax = jnp.asarray(obs_sigma, dtype=dtype)

    pipe.set_observations = set_observations
    return pipe


# =============================================================================
# Observations
# =============================================================================


def per_point_sigma(pipe: Pipeline, flux_true: np.ndarray) -> np.ndarray:
    """Per-point observational sigma for the chosen noise model (length n_times).

    - "white"  : constant cfg.obs_sigma.
    - "photon" : heteroscedastic photon noise sigma_i = sigma_phot / sqrt(F_tot_i),
                 with total relative flux F_tot_i = 1 (star) + planet flux_i. Photon
                 count scales with total flux, so brighter (dayside) points have
                 slightly smaller fractional error. Sigma floored at a tiny positive.
    """
    cfg = pipe.cfg
    mode = str(cfg.noise_model).strip().lower()
    if mode == "white":
        return np.full(flux_true.shape, float(cfg.obs_sigma), dtype=pipe.npdtype)
    if mode == "photon":
        F_tot = 1.0 + np.asarray(flux_true, dtype=np.float64)        # star (=1) + planet
        F_tot = np.clip(F_tot, 1e-6, None)
        sig = float(cfg.sigma_phot) / np.sqrt(F_tot)
        return sig.astype(pipe.npdtype)
    raise ValueError(f"Unknown cfg.noise_model={cfg.noise_model!r} (use 'white' or 'photon').")


def generate_observations(pipe: Pipeline, *, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Generate synthetic noisy observations from the truth theta and inject them
    into ``pipe`` (sets ``pipe.flux_true/flux_obs`` and the per-point sigma).

    The per-point sigma is computed once from the truth (the data's photon budget)
    and held fixed in the likelihood, as real per-point uncertainties are.
    """
    cfg = pipe.cfg
    seed = cfg.seed if seed is None else seed
    flux_true = np.asarray(pipe.phase_curve_model_jit(pipe.theta_truth)).astype(pipe.npdtype)
    if not np.isfinite(flux_true).any():
        raise RuntimeError("Truth flux is all non-finite; forward model invalid at truth.")
    sigma_vec = per_point_sigma(pipe, flux_true)
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(flux_true.shape) * sigma_vec).astype(pipe.npdtype)
    flux_obs = flux_true + noise
    pipe.flux_true = flux_true
    pipe.set_observations(flux_obs, obs_sigma=sigma_vec)
    return dict(times_days=pipe.times_days, flux_true=flux_true, flux_obs=flux_obs,
                obs_sigma=sigma_vec, obs_sigma_mean=float(np.mean(sigma_vec)),
                noise_model=str(cfg.noise_model),
                orbital_period_days=float(pipe.orbital_period_days_base))


# =============================================================================
# SMC machinery
# =============================================================================

import blackjax  # noqa: E402
from blackjax.smc import adaptive_tempered as smc_adaptive_tempered  # noqa: E402
from blackjax.smc import resampling as smc_resampling_mod  # noqa: E402


def _extract_acceptance_rate(info: Any):
    for attr in ("acceptance_rate", "acceptance_probability", "accept_prob", "prob_accept"):
        if hasattr(info, attr):
            return jnp.asarray(getattr(info, attr), dtype=float_dtype())
    if hasattr(info, "is_accepted"):
        return jnp.asarray(getattr(info, "is_accepted"), dtype=float_dtype())
    if hasattr(info, "update_info"):
        return _extract_acceptance_rate(getattr(info, "update_info"))
    raise AttributeError("Could not extract acceptance statistic from BlackJAX info object.")


def _mala_step_one(rng_key, state, logdensity_fn, step_size):
    kernel = blackjax.mala.build_kernel()
    try:
        return kernel(rng_key, state, logdensity_fn, step_size)
    except TypeError:
        return kernel(rng_key, state, logdensity_fn, step_size=step_size)


def _hmc_step_one(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps):
    kernel = blackjax.hmc.build_kernel()
    try:
        return kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
    except TypeError:
        return kernel(rng_key, state, logdensity_fn, step_size=step_size,
                      inverse_mass_matrix=inverse_mass_matrix, num_integration_steps=num_integration_steps)


def tune_mcmc_step_size(pipe: Pipeline, rng_key) -> float:
    """Auto-tune MALA/HMC step size in u-space with a small pilot run (Robbins-Monro)."""
    cfg = pipe.cfg
    dtype = pipe.dtype
    if not bool(cfg.mcmc_auto_tune):
        return float(cfg.mala_step_size) if str(cfg.smc_mcmc_kernel).strip().lower() == "mala" else float(cfg.hmc_step_size)

    kernel = str(cfg.smc_mcmc_kernel).strip().lower()
    beta = jnp.asarray(float(cfg.mcmc_tune_beta), dtype=dtype)
    n_particles, n_steps, n_iters = int(cfg.mcmc_tune_particles), int(cfg.mcmc_tune_steps), int(cfg.mcmc_tune_iters)
    step_min, step_max, gain = float(cfg.mcmc_step_size_min), float(cfg.mcmc_step_size_max), float(cfg.mcmc_tune_gain)

    if n_particles <= 0 or n_steps <= 0 or n_iters <= 0:
        raise ValueError("MCMC tuning counts must be positive.")
    if not (0.0 < float(beta) <= 1.0):
        raise ValueError(f"cfg.mcmc_tune_beta must be in (0,1], got {float(beta)}")
    if not (0.0 < step_min < step_max):
        raise ValueError(f"Invalid step size bounds: min={step_min}, max={step_max}")

    if kernel == "mala":
        target, step0 = float(cfg.mcmc_target_accept_mala), float(cfg.mala_step_size)
    else:
        target, step0 = float(cfg.mcmc_target_accept_hmc), float(cfg.hmc_step_size)

    def logdensity(u):
        return pipe.log_prior_u(u) + beta * pipe.loglikelihood_for_blackjax(u)

    rng_key, subkey = jax.random.split(rng_key)
    u0 = pipe.sample_prior_u(subkey, n_particles)

    if kernel == "mala":
        state = jax.vmap(lambda uu: blackjax.mala.init(uu, logdensity))(u0)

        @jax.jit
        def _run_block(key_in, state_in, step_size_scalar):
            def one_step(carry, _):
                key, st = carry
                key, sub = jax.random.split(key)
                keys = jax.random.split(sub, n_particles)
                st_new, info = jax.vmap(lambda kk, ss: _mala_step_one(kk, ss, logdensity, step_size_scalar))(keys, st)
                return (key, st_new), jnp.mean(_extract_acceptance_rate(info))
            (key_out, state_out), accs = jax.lax.scan(one_step, (key_in, state_in), None, length=n_steps)
            return key_out, state_out, jnp.mean(accs)
    else:
        inv_mass = jnp.ones((pipe.n_dim,), dtype=dtype)
        n_leapfrog = jnp.asarray(int(cfg.hmc_num_integration_steps), dtype=jnp.int32)
        state = jax.vmap(lambda uu: blackjax.hmc.init(uu, logdensity))(u0)

        @jax.jit
        def _run_block(key_in, state_in, step_size_scalar):
            def one_step(carry, _):
                key, st = carry
                key, sub = jax.random.split(key)
                keys = jax.random.split(sub, n_particles)
                st_new, info = jax.vmap(
                    lambda kk, ss: _hmc_step_one(kk, ss, logdensity, step_size_scalar, inv_mass, n_leapfrog))(keys, st)
                return (key, st_new), jnp.mean(_extract_acceptance_rate(info))
            (key_out, state_out), accs = jax.lax.scan(one_step, (key_in, state_in), None, length=n_steps)
            return key_out, state_out, jnp.mean(accs)

    log_step = math.log(min(max(step0, step_min), step_max))
    tol = 0.05
    for it in range(n_iters):
        step_jax = jnp.asarray(math.exp(log_step), dtype=dtype)
        rng_key, state, acc = _run_block(rng_key, state, step_jax)
        acc_f = float(jax.device_get(acc))
        log_step = log_step + gain * (acc_f - target)
        log_step = math.log(min(max(math.exp(log_step), step_min), step_max))
        if it >= 2 and abs(acc_f - target) < tol:
            break
    tuned = float(math.exp(log_step))
    logger.info(f"Auto-tuned {kernel.upper()} step size (u-space): {tuned:.4g} (target_accept={target:.2f})")
    return tuned


def build_smc_algorithm(pipe: Pipeline, *, step_size_override: Optional[float] = None,
                        inverse_mass_diag_override: Optional[Any] = None):
    """Build the BlackJAX adaptive-tempered-SMC top-level API for this pipeline."""
    cfg = pipe.cfg
    dtype = pipe.dtype
    kernel = str(cfg.smc_mcmc_kernel).strip().lower()
    N = int(cfg.smc_num_particles)

    if kernel == "mala":
        mcmc_step_fn = blackjax.mala.build_kernel()
        mcmc_init_fn = blackjax.mala.init
        step = float(cfg.mala_step_size) if step_size_override is None else float(step_size_override)
        mcmc_parameters = {"step_size": jnp.full((N,), step, dtype=dtype)}
    elif kernel == "hmc":
        mcmc_step_fn = blackjax.hmc.build_kernel()
        mcmc_init_fn = blackjax.hmc.init
        dim = int(pipe.n_dim)
        step = float(cfg.hmc_step_size) if step_size_override is None else float(step_size_override)
        inv_mass_diag = (jnp.ones((dim,), dtype=dtype) if inverse_mass_diag_override is None
                         else jnp.asarray(inverse_mass_diag_override, dtype=dtype))
        if tuple(inv_mass_diag.shape) != (dim,):
            raise ValueError(f"inverse_mass_diag_override must have shape ({dim},)")
        mcmc_parameters = {
            "step_size": jnp.full((N,), step, dtype=dtype),
            "inverse_mass_matrix": jnp.tile(inv_mass_diag[None, :], (N, 1)),
            "num_integration_steps": jnp.full((N,), int(cfg.hmc_num_integration_steps), dtype=jnp.int32),
        }
    else:
        raise ValueError(f"Unknown cfg.smc_mcmc_kernel={cfg.smc_mcmc_kernel!r}")

    resampling_name = str(cfg.smc_resampling).strip().lower()
    resampling_fn = {"systematic": smc_resampling_mod.systematic,
                     "stratified": smc_resampling_mod.stratified,
                     "multinomial": smc_resampling_mod.multinomial}[resampling_name]

    target_ess_frac = float(cfg.smc_target_ess_frac)
    if (not math.isfinite(target_ess_frac)) or target_ess_frac <= 0.0 or target_ess_frac > 1.0:
        raise ValueError(f"cfg.smc_target_ess_frac must be in (0, 1]. Got {cfg.smc_target_ess_frac!r}.")

    logger.info(f"Building adaptive tempered SMC: kernel={kernel}, N={N}, target_ess_frac={target_ess_frac:.3f}, "
                f"num_mcmc_steps={cfg.smc_num_mcmc_steps}, step_size={step:.4g}")

    return smc_adaptive_tempered.as_top_level_api(
        logprior_fn=pipe.log_prior_u, loglikelihood_fn=pipe.loglikelihood_for_blackjax,
        mcmc_step_fn=mcmc_step_fn, mcmc_init_fn=mcmc_init_fn, mcmc_parameters=mcmc_parameters,
        resampling_fn=resampling_fn, num_mcmc_steps=int(cfg.smc_num_mcmc_steps), target_ess=target_ess_frac)


def run_smc_loop(pipe: Pipeline, *, key, progress: bool = True) -> Dict[str, Any]:
    """Run adaptive-tempered SMC to beta=1; returns particles, weights, diagnostics, draws.

    No file IO. Tuning (if enabled) happens here.
    """
    cfg = pipe.cfg
    key, subkey = jax.random.split(key)
    particles0 = pipe.sample_prior_u(subkey, int(cfg.smc_num_particles))

    tuned_step_size: Optional[float] = None
    if bool(cfg.mcmc_auto_tune):
        key, tune_key = jax.random.split(key)
        tuned_step_size = tune_mcmc_step_size(pipe, tune_key)

    kernel_name = str(cfg.smc_mcmc_kernel).strip().lower()
    step_used = (float(cfg.mala_step_size) if kernel_name == "mala" else float(cfg.hmc_step_size)) \
        if tuned_step_size is None else float(tuned_step_size)

    smc = build_smc_algorithm(pipe, step_size_override=step_used)
    smc_step = jax.jit(smc.step)
    state = smc.init(particles0)

    key, subkey = jax.random.split(key)
    try:
        smc_step.lower(subkey, state).compile()
    except Exception:
        _s, _i = smc_step(subkey, state)
        jax.block_until_ready(_s)
        state = smc.init(particles0)

    betas: List[float] = [0.0]
    ess_hist, acc_hist, logz_inc_hist = [], [], []

    iterator = range(int(cfg.smc_max_steps))
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Adaptive tempered SMC", leave=True)
        except Exception:
            pass

    for i in iterator:
        key, subkey = jax.random.split(key)
        state, info = smc_step(subkey, state)
        jax.block_until_ready(state)
        beta = float(jax.device_get(state.tempering_param))
        w = state.weights
        ess = float(jax.device_get(1.0 / jnp.sum(w * w)))
        if (not math.isfinite(beta)) or (not math.isfinite(ess)):
            raise RuntimeError(f"Non-finite SMC diagnostics at step {i:03d}: beta={beta}, ess={ess}.")
        acc = float("nan")
        try:
            acc = float(jax.device_get(jnp.mean(info.update_info.acceptance_rate)))
        except Exception:
            pass
        betas.append(beta)
        ess_hist.append(ess)
        acc_hist.append(acc)
        logz_inc = float("nan")
        for kn in ("log_likelihood_increment", "logZ_increment", "log_normalizer_increment"):
            if hasattr(info, kn):
                try:
                    logz_inc = float(jax.device_get(getattr(info, kn)))
                    break
                except Exception:
                    pass
        logz_inc_hist.append(logz_inc)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(beta=f"{beta:.3f}", ess=f"{ess:.1f}", acc=f"{acc:.3f}")
        logger.info(f"SMC step {i:03d}: beta={beta:.6f}, ESS={ess:.1f}/{cfg.smc_num_particles}, mean_accept={acc:.3f}")
        if beta >= 1.0 - 1e-8:
            break

    final_beta = float(jax.device_get(state.tempering_param))
    reached = math.isfinite(final_beta) and final_beta >= 1.0 - 1e-6

    n_draws_total = int(cfg.num_chains) * int(cfg.num_samples)
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, int(cfg.smc_num_particles), shape=(n_draws_total,), p=state.weights, replace=True)
    u_draws = state.particles[idx]
    theta_draws = jax.vmap(pipe.theta_from_u)(u_draws)
    theta_np = np.asarray(theta_draws, dtype=np.float64).reshape((int(cfg.num_chains), int(cfg.num_samples), pipe.n_dim))

    return dict(
        state=state, reached_beta1=reached, final_beta=final_beta, step_size_used=step_used,
        betas=np.asarray(betas, dtype=np.float64),
        ess=np.asarray(ess_hist, dtype=np.float64),
        acceptance_rate=np.asarray(acc_hist, dtype=np.float64),
        logZ_increment=np.asarray(logz_inc_hist, dtype=np.float64),
        final_weights=np.asarray(jax.device_get(state.weights), dtype=np.float64),
        theta_draws=theta_np, u_draws=np.asarray(jax.device_get(u_draws), dtype=np.float64),
    )


def fast_cpu_config(**overrides: Any) -> Config:
    """A fast local-CPU preset: ~2-day spin-up, fewer particles/steps, float32.

    Override any field via kwargs, e.g. ``fast_cpu_config(model_days=3.0)``.
    """
    base = dict(
        use_x64=False, M=42, dt_seconds=240.0, model_days=2.0,
        taurad_true_hours=8.0, taudrag_true_hours=8.0,
        obs_sigma=80e-6, n_times=200,
        smc_num_particles=24, smc_num_mcmc_steps=12, smc_max_steps=24, smc_target_ess_frac=0.6,
        mcmc_tune_particles=8, mcmc_tune_steps=6, mcmc_tune_iters=6,
        num_samples=24, num_chains=2, ppc_draws=48, ppc_chunk_size=16,
    )
    base.update(overrides)
    return Config(**base)


def gpu_config(**overrides: Any) -> Config:
    """Full-retrieval GPU preset (A100/H100), paper-aligned: a 64-particle SMC swarm,
    20-day spin-up, heteroscedastic photon noise, float64, paper temperature mapping.

    The SMC mutation kernel is ``jax.vmap``-ed over particles, so the whole swarm
    (``smc_num_particles``) advances simultaneously on the device. Per the SWAMPE-JAX
    paper, A100 throughput SATURATES at a few dozen simultaneous trajectories, so
    **N=64 is the efficient sweet spot** — larger swarms (256/512) fit in memory but
    just queue (no throughput gain) and multiply wall-time. The likelihood uses a
    custom FORWARD-MODE-JVP gradient (the paper's stated approach), so there is NO
    reverse-mode tape through the 7200-step scan: peak memory is O(n_particles*J*I),
    not O(n_steps*...).

    Paper alignment (parity-figure caption + speed section):
    - dt=240 s, model_days=20 -> 7200 steps (= the paper's 10-day-at-120s benchmark
      step count; dt=120 gives an IDENTICAL phase curve at 2x the cost, so 240 is used).
    - truth tau_rad=10 h, tau_drag=6 h; Phibar=3e5, DPhieq=1e6, omega=3.2e-5, a=8.2e7, M=42.
    - emission_temp_mode="geopotential": T=(Phibar+Phi)/R_d (R_d=3.78e3).
    - float64 for robustness over the long integration (modest cost on A100).

    Notes
    -----
    - The forward is already converged by ~5 days; model_days=20 is conservative (~4x
      the cost of 5 days for ~identical results) — lower it to save time if desired.
    - 64 particles is a sparse cloud for a 2D corner; the plots use KDE smoothing.
      Raise smc_num_mcmc_steps for more mixing, or N for more samples (diminishing
      returns past ~a few dozen per the paper's saturation).
    """
    base = dict(
        use_x64=True, M=42, dt_seconds=240.0, model_days=20.0,
        taurad_true_hours=10.0, taudrag_true_hours=6.0,
        infer_tau_rad=True, infer_tau_drag=True,
        emission_temp_mode="geopotential", R_d=3.78e3,
        noise_model="photon", sigma_phot=50e-6, n_times=300, n_orbits_observed=1.0,
        smc_num_particles=64, smc_num_mcmc_steps=20, smc_max_steps=40, smc_target_ess_frac=0.70,
        mcmc_tune_particles=32, mcmc_tune_steps=8, mcmc_tune_iters=6,
        num_samples=64, num_chains=2, ppc_draws=64, ppc_chunk_size=64,
    )
    base.update(overrides)
    return Config(**base)
