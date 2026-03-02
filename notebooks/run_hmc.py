#!/usr/bin/env python3
"""
run_exposed.py

Differentiable end-to-end SWAMP ➜ starry/jaxoplanet phase-curve retrieval using
parallel Hamiltonian Monte Carlo (BlackJAX NUTS).

Key points
----------
- The SWAMP + starry forward model is unchanged and JIT compiled.
- Synthetic data generation is preserved (and can be reused from disk).
- Inference uses NUTS (HMC) with per-chain warmup (dual averaging + diagonal mass matrix),
  with warmup & sampling loops executed as JAX `lax.scan` (GPU-friendly).
- All *continuous* SWAMP parameters that can be passed through `build_static` (and the
  continuous `alpha` RunFlag) are exposed as *optional* inference parameters. Each
  parameter has a boolean `infer_*` switch in Config.

Outputs written to cfg.out_dir
------------------------------
- config.json
- run.log
- observations.npz
- posterior_samples.npz
- hmc_diagnostics.npz
- posterior_predictive.npz              (optional)
- posterior_predictive_quantiles.npz    (optional)
- maps_truth_and_posterior_summary.npz

Run:
  python run_exposed.py

Notes
-----
1) Integration length:
   - cfg.model_days is in *Earth days*.
   - If you want to specify "planet days" (i.e. orbital/rotation periods), set
     cfg.model_orbits instead. The script converts orbits -> days using omega_rad_s.
   - IMPORTANT: The number of SWAMP time steps is static (compile-time). Do not infer
     any parameter that would change dt or the integration length.

2) Parameterization:
   - All inferred parameters are sampled in an unconstrained space `u` with a sigmoid
     transform to a bounded range in *log10(parameter)*. This yields stable inference
     for wide dynamic ranges and keeps all inferred parameters strictly positive.

3) Custom gradients:
   - cfg.use_custom_gradients=True uses forward-mode JVPs via a custom VJP for stable
     memory on long scans, but scales ~O(dim^2). It is auto-disabled when the number of
     inferred parameters exceeds cfg.custom_grad_max_dim.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm


# =============================================================================
# Configuration (edit here; no CLI args by design)
# =============================================================================


@dataclass(frozen=True)
class Config:
    # -----------------------
    # I/O & reproducibility
    # -----------------------
    out_dir: Path = Path("swamp_jaxoplanet_retrieval_outputs")
    seed: int = 7
    log_level: str = "INFO"
    overwrite: bool = True

    # -----------------------
    # Numeric precision / XLA behavior
    # -----------------------
    use_x64: bool = False
    xla_preallocate: bool = False

    # -----------------------
    # SWAMP numerical params (STATIC; do not infer)
    # -----------------------
    M: int = 42
    dt_seconds: float = 360.0

    # Choose ONE of:
    #   - model_days: integrate for this many Earth days
    #   - model_orbits: integrate for this many orbital/rotation periods ("planet days")
    model_days: Optional[float] = 10.0
    model_orbits: Optional[float] = None

    starttime_index: int = 2  # leapfrog start index (>=2)

    # -----------------------
    # SWAMP physical params (defaults also used as synthetic "truth")
    # -----------------------
    a_planet_m: float = 8.2e7
    omega_rad_s: float = 3.2e-5
    g_m_s2: float = 9.8
    Phibar: float = 3.0e5
    DPhieq: float = 1.0e6
    K6: float = 1.24e33
    K6Phi: Optional[float] = None

    # RunFlags (treated as fixed unless infer_alpha=True)
    forcflag: bool = True
    diffflag: bool = True
    expflag: bool = False
    modalflag: bool = True
    alpha: float = 0.01
    diagnostics: bool = False  # keep False for clean JAX
    blowup_rms: float = 1.0e30

    # -----------------------
    # Phi -> temperature -> intensity
    # -----------------------
    T_ref: float = 300.0
    phi_to_T_scale: float = 4.0e4
    Tmin_K: float = 1.0

    # emission_model="bolometric": I ∝ T^4
    # emission_model="planck":     I ∝ B_λ(T) at planck_wavelength_m (up to a constant factor)
    emission_model: str = "bolometric"  # {"bolometric", "planck"}
    planck_wavelength_m: float = 4.5e-6
    planck_x_clip: float = 80.0

    # -----------------------
    # starry / map projection
    # -----------------------
    ydeg: int = 10
    projector_ridge: float = 1.0e-6

    # Map orientation (known)
    map_inc_rad: float = math.pi / 2
    map_obl_rad: float = 0.0
    # Phase convention: at transit, observer sees nightside for tidally locked planet.
    phase_at_transit_rad: float = math.pi

    # -----------------------
    # Orbit/system geometry (known)
    # -----------------------
    star_mass_msun: float = 1.0
    star_radius_rsun: float = 1.0
    planet_radius_rjup: float = 1.0
    impact_param: float = 0.0
    time_transit_days: float = 0.0
    orbital_period_override_days: Optional[float] = None  # if None, derived from omega_rad_s at startup

    # Flux scaling: planet/star flux ratio (fixed here; not part of SWAMP)
    planet_fpfs: float = 1500e-6

    # -----------------------
    # Synthetic observations
    # -----------------------
    generate_synthetic_data: bool = True
    n_times: int = 250
    n_orbits_observed: float = 1.0
    obs_sigma: float = 80e-6

    # Radiative & drag timescale "truth" (and default fixed values if not inferred)
    taurad_true_hours: float = 10.0
    taudrag_true_hours: float = 6.0

    # -----------------------
    # Inference: BlackJAX NUTS (parallel chains via vmap)
    # -----------------------
    run_inference: bool = True

    # Which continuous parameters are inferred?
    infer_taurad: bool = True
    infer_taudrag: bool = True
    infer_a_planet_m: bool = False
    infer_omega_rad_s: bool = False
    infer_g_m_s2: bool = False
    infer_Phibar: bool = False
    infer_DPhieq: bool = False
    infer_K6: bool = False
    infer_K6Phi: bool = False
    infer_alpha: bool = False

    # Prior on log10(tau_hours) (applied to any inferred tau parameter)
    prior_log10_tau_hours_min: float = -1.0  # 0.1 hours
    prior_log10_tau_hours_max: float = 2.0   # 100 hours

    # For other positive parameters: uniform prior in log10(parameter) on a bounded range.
    # If min/max are None, bounds are set to log10(base_value) ± other_param_log10_half_width_dex.
    other_param_log10_half_width_dex: float = 1.0

    prior_log10_a_planet_m_min: Optional[float] = None
    prior_log10_a_planet_m_max: Optional[float] = None

    prior_log10_omega_rad_s_min: Optional[float] = None
    prior_log10_omega_rad_s_max: Optional[float] = None

    prior_log10_g_m_s2_min: Optional[float] = None
    prior_log10_g_m_s2_max: Optional[float] = None

    prior_log10_Phibar_min: Optional[float] = None
    prior_log10_Phibar_max: Optional[float] = None

    prior_log10_DPhieq_min: Optional[float] = None
    prior_log10_DPhieq_max: Optional[float] = None

    prior_log10_K6_min: Optional[float] = None
    prior_log10_K6_max: Optional[float] = None

    prior_log10_K6Phi_min: Optional[float] = None
    prior_log10_K6Phi_max: Optional[float] = None

    prior_log10_alpha_min: Optional[float] = None
    prior_log10_alpha_max: Optional[float] = None

    # NUTS configuration
    num_chains: int = 1          # run in parallel via vmap on GPU
    num_warmup: int = 20         # warmup steps (split into phase1+phase2)
    warmup_phase1_frac: float = 0.75
    warmup_mass_matrix_collect_frac: float = 1.0 / 3.0  # collect after this fraction of phase1
    num_samples: int = 50       # post-warmup draws per chain

    nuts_initial_step_size: float = 0.1
    target_accept: float = 0.6
    nuts_max_num_doublings: int = 2

    # Dual averaging hyperparameters (Hoffman & Gelman 2014)
    da_gamma: float = 0.05
    da_t0: float = 10.0
    da_kappa: float = 0.75

    # Diagonal mass matrix estimation safety (u-space variance floor)
    mass_matrix_var_floor: float = 1e-6

    # Gradient strategy: forward-mode JVPs for small dim (stable memory for long scans)
    use_custom_gradients: bool = True
    custom_grad_max_dim: int = 4

    # -----------------------
    # Posterior predictive (optional)
    # -----------------------
    do_ppc: bool = True
    ppc_draws: int = 128
    ppc_chunk_size: int = 16

    # -----------------------
    # Misc
    # -----------------------
    cache_projector: bool = True  # saves projector to out_dir/projector_cache.npz

    # Plot-related config saved for the plot script
    fig_dpi: int = 160


cfg = Config()


# =============================================================================
# Validate config early (fail fast)
# =============================================================================


_valid_emission_models = {"bolometric", "planck"}
if str(cfg.emission_model).strip().lower() not in _valid_emission_models:
    raise ValueError(f"cfg.emission_model must be one of {_valid_emission_models}, got {cfg.emission_model!r}")

if cfg.dt_seconds <= 0:
    raise ValueError("cfg.dt_seconds must be > 0")
if cfg.starttime_index < 2:
    raise ValueError("cfg.starttime_index must be >= 2 for leapfrog startup")
if cfg.num_chains < 1:
    raise ValueError("cfg.num_chains must be >= 1")
if cfg.num_samples < 1:
    raise ValueError("cfg.num_samples must be >= 1")
if cfg.num_warmup < 0:
    raise ValueError("cfg.num_warmup must be >= 0")
if cfg.warmup_phase1_frac <= 0.0 or cfg.warmup_phase1_frac >= 1.0:
    raise ValueError("cfg.warmup_phase1_frac must be in (0,1)")
if cfg.mass_matrix_var_floor <= 0.0:
    raise ValueError("cfg.mass_matrix_var_floor must be > 0")
if cfg.other_param_log10_half_width_dex <= 0.0:
    raise ValueError("cfg.other_param_log10_half_width_dex must be > 0")
if cfg.custom_grad_max_dim < 1:
    raise ValueError("cfg.custom_grad_max_dim must be >= 1")

# Sanity: inferred parameters must be positive (by construction), so base values must be positive too.
for _name, _val in [
    ("a_planet_m", cfg.a_planet_m),
    ("omega_rad_s", cfg.omega_rad_s),
    ("g_m_s2", cfg.g_m_s2),
    ("Phibar", cfg.Phibar),
    ("DPhieq", cfg.DPhieq),
    ("K6", cfg.K6),
    ("alpha", cfg.alpha),
    ("taurad_true_hours", cfg.taurad_true_hours),
    ("taudrag_true_hours", cfg.taudrag_true_hours),
]:
    if not (isinstance(_val, (int, float)) and float(_val) > 0.0 and math.isfinite(float(_val))):
        raise ValueError(f"Config value for {_name} must be finite and > 0, got {_val!r}")

if bool(cfg.infer_K6Phi):
    if cfg.K6Phi is None:
        raise ValueError("cfg.infer_K6Phi=True requires cfg.K6Phi to be a positive float (not None).")
    if not (float(cfg.K6Phi) > 0.0 and math.isfinite(float(cfg.K6Phi))):
        raise ValueError(f"cfg.K6Phi must be finite and > 0 when inferred; got {cfg.K6Phi!r}")

_n_infer = int(
    bool(cfg.infer_taurad)
    + bool(cfg.infer_taudrag)
    + bool(cfg.infer_a_planet_m)
    + bool(cfg.infer_omega_rad_s)
    + bool(cfg.infer_g_m_s2)
    + bool(cfg.infer_Phibar)
    + bool(cfg.infer_DPhieq)
    + bool(cfg.infer_K6)
    + bool(cfg.infer_K6Phi)
    + bool(cfg.infer_alpha)
)
if bool(cfg.run_inference) and _n_infer < 1:
    raise ValueError("cfg.run_inference=True requires at least one infer_* switch to be True.")


# =============================================================================
# Environment + logging (set env vars BEFORE importing JAX/my_swamp)
# =============================================================================


os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1" if cfg.use_x64 else "0")
if not cfg.xla_preallocate:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

cfg.out_dir.mkdir(parents=True, exist_ok=True)
log_path = cfg.out_dir / "run.log"

logging.basicConfig(
    level=getattr(logging, str(cfg.log_level).upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w" if cfg.overwrite else "a"),
    ],
    force=True,
)
logger = logging.getLogger("swamp_run_hmc")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

(cfg.out_dir / "config_hmc.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
logger.info(f"Wrote config to: {cfg.out_dir / 'config_hmc.json'}")


# =============================================================================
# Imports (after env vars)
# =============================================================================


import jax

jax.config.update("jax_enable_x64", bool(cfg.use_x64))
import jax.numpy as jnp

Array = jax.Array

import my_swamp.model as swamp_model
from my_swamp.model import RunFlags
from my_swamp.model import build_static

from jaxoplanet.orbits.keplerian import Body
from jaxoplanet.orbits.keplerian import Central
from jaxoplanet.starry.light_curves import light_curve as starry_light_curve
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

try:
    import blackjax  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "BlackJAX is required for this script. Install with `pip install blackjax` "
        "(and ensure your JAX install matches your accelerator)."
    ) from e

logger.info(f"JAX backend: {jax.default_backend()}")
logger.info(f"JAX devices: {jax.devices()}")
logger.info(f"BlackJAX version: {getattr(blackjax, '__version__', 'unknown')}")


# =============================================================================
# Utility helpers
# =============================================================================


def float_dtype() -> Any:
    return jnp.float64 if cfg.use_x64 else jnp.float32


def np_float_dtype() -> Any:
    return np.float64 if cfg.use_x64 else np.float32


def tau_hours_to_seconds(x: Any) -> Any:
    return 3600.0 * x


def orbital_period_days_from_omega(omega_rad_s: float) -> float:
    return float((2.0 * math.pi / float(omega_rad_s)) / 86400.0)


def compute_n_steps(model_days: float, dt_seconds: float) -> int:
    n = int(np.round(float(model_days) * 86400.0 / float(dt_seconds)))
    return max(n, 1)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def call_with_filtered_kwargs(func, kwargs: Dict[str, Any], *, name: Optional[str] = None):
    """Call func(**kwargs), filtering out unexpected kwargs using inspect.signature."""
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


def _replace_dataclass(base: Any, updates: Dict[str, Any], *, name: str) -> Any:
    """JIT-safe dataclass update.

    `dataclasses.replace` executes at trace time and is compatible with JAX pytrees.
    This MUST work for inference parameters; rebuilding inside the model is not viable.
    """
    if not updates:
        return base
    try:
        return dc_replace(base, **updates)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"dataclasses.replace failed for {name} with updates={list(updates.keys())}. "
            "For HMC/NUTS this must be supported under jit. Ensure the object is a dataclass "
            "with those field names and that the fields are consumed at step time."
        ) from e


def phi_to_temperature(phi: jnp.ndarray) -> jnp.ndarray:
    dtype = float_dtype()
    T = jnp.asarray(cfg.T_ref, dtype=dtype) + phi / jnp.asarray(cfg.phi_to_T_scale, dtype=dtype)
    return jnp.maximum(T, jnp.asarray(cfg.Tmin_K, dtype=dtype))


def planck_intensity_relative_lambda(T: jnp.ndarray, wavelength_m: float) -> jnp.ndarray:
    dtype = float_dtype()
    T = jnp.asarray(T, dtype=dtype)
    lam = jnp.asarray(float(wavelength_m), dtype=dtype)

    h = jnp.asarray(6.62607015e-34, dtype=dtype)
    c = jnp.asarray(299792458.0, dtype=dtype)
    kB = jnp.asarray(1.380649e-23, dtype=dtype)

    x = (h * c) / (lam * kB * T)
    x = jnp.clip(x, a_min=jnp.asarray(0.0, dtype=dtype), a_max=jnp.asarray(cfg.planck_x_clip, dtype=dtype))

    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    return jnp.asarray(1.0, dtype=dtype) / (jnp.expm1(x) + tiny)


def temperature_to_intensity(T: jnp.ndarray) -> jnp.ndarray:
    mode = str(cfg.emission_model).strip().lower()
    if mode == "bolometric":
        return T**4
    if mode == "planck":
        return planck_intensity_relative_lambda(T, float(cfg.planck_wavelength_m))
    raise ValueError(f"Unknown cfg.emission_model={cfg.emission_model!r}")


def build_lm_list(ydeg: int) -> List[Tuple[int, int]]:
    return [(ell, m) for ell in range(ydeg + 1) for m in range(-ell, ell + 1)]


def ylm_from_dense(y_dense: jnp.ndarray, lm_list: Sequence[Tuple[int, int]]) -> Ylm:
    data = {lm: y_dense[i] for i, lm in enumerate(lm_list)}
    return Ylm(data)


def surface_intensity(surf: Surface, latv: jnp.ndarray, lonv: jnp.ndarray) -> jnp.ndarray:
    """Call Surface.intensity with signature safety across jaxoplanet versions."""
    try:
        sig = inspect.signature(surf.intensity)
        if "theta" in sig.parameters:
            return surf.intensity(latv, lonv, theta=jnp.asarray(0.0, dtype=float_dtype()))
        return surf.intensity(latv, lonv)
    except (TypeError, ValueError):
        return surf.intensity(latv, lonv)


# =============================================================================
# Derived orbital period + integration length
# =============================================================================


orbital_period_days = (
    float(cfg.orbital_period_override_days)
    if cfg.orbital_period_override_days is not None
    else orbital_period_days_from_omega(cfg.omega_rad_s)
)
logger.info(f"Orbital period used for light curve (days): {orbital_period_days:.6f}")

if cfg.model_orbits is not None:
    if cfg.model_orbits <= 0:
        raise ValueError("cfg.model_orbits must be > 0 when provided")
    model_days_effective = float(cfg.model_orbits) * orbital_period_days
    logger.info(f"SWAMP integration length: model_orbits={cfg.model_orbits} -> model_days={model_days_effective:.6f}")
else:
    if cfg.model_days is None or cfg.model_days <= 0:
        raise ValueError("cfg.model_days must be > 0 when cfg.model_orbits is None")
    model_days_effective = float(cfg.model_days)
    logger.info(f"SWAMP integration length: model_days={model_days_effective:.6f} (Earth days)")

n_steps = compute_n_steps(model_days_effective, float(cfg.dt_seconds))
logger.info(f"SWAMP integration steps: n_steps={n_steps} (dt={cfg.dt_seconds}s)")

# Time indices used by SWAMP stepper (STATIC shape)
t_seq = jnp.arange(int(cfg.starttime_index), int(cfg.starttime_index) + int(n_steps), dtype=jnp.int32)


# =============================================================================
# Build SWAMP static + RunFlags (once)
# =============================================================================


static_kwargs: Dict[str, Any] = dict(
    M=int(cfg.M),
    dt=float(cfg.dt_seconds),
    a=float(cfg.a_planet_m),
    omega=float(cfg.omega_rad_s),
    g=float(cfg.g_m_s2),
    Phibar=float(cfg.Phibar),
    taurad=float(tau_hours_to_seconds(cfg.taurad_true_hours)),
    taudrag=float(tau_hours_to_seconds(cfg.taudrag_true_hours)),
    DPhieq=float(cfg.DPhieq),
    K6=float(cfg.K6),
    K6Phi=(None if cfg.K6Phi is None else float(cfg.K6Phi)),
    test=None,
)
static_base = call_with_filtered_kwargs(build_static, static_kwargs, name="build_static")

flags_kwargs = dict(
    forcflag=bool(cfg.forcflag),
    diffflag=bool(cfg.diffflag),
    expflag=bool(cfg.expflag),
    modalflag=bool(cfg.modalflag),
    diagnostics=bool(cfg.diagnostics),
    alpha=float(cfg.alpha),
    blowup_rms=float(cfg.blowup_rms),
)
flags_base = call_with_filtered_kwargs(RunFlags, flags_kwargs, name="RunFlags")

I = int(getattr(static_base, "I", -1))
J = int(getattr(static_base, "J", -1))
logger.info(f"SWAMP grid: I={I}, J={J}, M={getattr(static_base,'M','?')}, N={getattr(static_base,'N','?')}")


# =============================================================================
# Init-state wrapper (we will create state0 inside the forward model so inferred
# params can affect ICs)
# =============================================================================


_init_fn = getattr(swamp_model, "_init_state_from_fields", None) or getattr(swamp_model, "init_state_from_fields", None)
if _init_fn is None:
    raise RuntimeError(
        "Could not find my_swamp.model._init_state_from_fields or init_state_from_fields. "
        "This pipeline requires initializing a State from fields without allocating history."
    )

_init_sig = None
try:
    _init_sig = inspect.signature(_init_fn)
except (TypeError, ValueError):
    _init_sig = None

_INIT_ACCEPTS_TEST = bool(_init_sig is not None and "test" in _init_sig.parameters)

def init_state_from_fields(
    *,
    static: Any,
    flags: Any,
    eta0: jnp.ndarray,
    delta0: jnp.ndarray,
    Phi0: jnp.ndarray,
    U0: jnp.ndarray,
    V0: jnp.ndarray,
) -> Any:
    kwargs: Dict[str, Any] = dict(
        static=static,
        flags=flags,
        eta0=eta0,
        delta0=delta0,
        Phi0=Phi0,
        U0=U0,
        V0=V0,
    )
    if _INIT_ACCEPTS_TEST:
        kwargs["test"] = None
    return _init_fn(**kwargs)


def init_rest_fields(static: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rest initial conditions (JAX-friendly; evaluated inside the forward model)."""
    dtype = float_dtype()
    Jloc = int(getattr(static, "J"))
    Iloc = int(getattr(static, "I"))

    mus = getattr(static, "mus", None)
    if mus is None:
        raise RuntimeError("static.mus not found; required for rest initial conditions.")
    omega = getattr(static, "omega", 0.0)

    eta1d = jnp.asarray(omega, dtype=dtype) * jnp.asarray(mus, dtype=dtype)
    eta0 = eta1d[:, None] * jnp.ones((Jloc, Iloc), dtype=dtype)

    delta0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
    U0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
    V0 = jnp.zeros((Jloc, Iloc), dtype=dtype)

    Phieq = getattr(static, "Phieq", None)
    if Phieq is None:
        raise RuntimeError("static.Phieq not found; required for Phi0.")
    Phi0 = jnp.asarray(Phieq, dtype=dtype)
    return eta0, delta0, U0, V0, Phi0


# =============================================================================
# Precompute SWAMP pixel grid + weights (once)
# =============================================================================


lambdas = getattr(static_base, "lambdas", None)
mus = getattr(static_base, "mus", None)
w_lat = getattr(static_base, "w", None)

if lambdas is None or mus is None:
    raise RuntimeError("static_base.lambdas and static_base.mus are required to build the starry projector.")

lon = jnp.asarray(lambdas, dtype=float_dtype())  # (I,)
lat = jnp.arcsin(jnp.asarray(mus, dtype=float_dtype()))  # (J,)

lon2d = jnp.broadcast_to(lon[None, :], (lat.shape[0], lon.shape[0]))
lat2d = jnp.broadcast_to(lat[:, None], (lat.shape[0], lon.shape[0]))
lon_flat = lon2d.reshape(-1)
lat_flat = lat2d.reshape(-1)

if w_lat is None:
    logger.warning("static_base.w not found; using uniform weights for LSQ projector.")
    w_pix = jnp.ones_like(lat_flat)
else:
    w_lat = jnp.asarray(w_lat, dtype=float_dtype())  # (J,)
    w_pix = jnp.repeat(w_lat, lon.shape[0])  # (J*I,)

w_sqrt = jnp.sqrt(w_pix)


# =============================================================================
# Build / load starry design matrix projector (once)
# =============================================================================


lm_list = build_lm_list(int(cfg.ydeg))
n_coeff = (int(cfg.ydeg) + 1) ** 2
n_pix = int(lat_flat.shape[0])

logger.info(f"starry projector: n_pix={n_pix}, n_coeff={n_coeff} (ydeg={cfg.ydeg})")

projector_cache_path = cfg.out_dir / "projector_cache_hmc.npz"
projector: jnp.ndarray

def _build_projector() -> jnp.ndarray:
    logger.info("Building starry design matrix (this is a one-time cost per grid/ydeg)...")

    def _intensity_from_yvec(y_vec: jnp.ndarray) -> jnp.ndarray:
        ylm = ylm_from_dense(y_vec, lm_list)
        surf = Surface(
            y=ylm,
            u=(),
            inc=jnp.asarray(cfg.map_inc_rad, dtype=float_dtype()),
            obl=jnp.asarray(cfg.map_obl_rad, dtype=float_dtype()),
            amplitude=jnp.asarray(1.0, dtype=float_dtype()),
            normalize=False,
        )
        return surface_intensity(surf, lat_flat, lon_flat)

    _intensity_from_yvec_jit = jax.jit(_intensity_from_yvec)

    eye = jnp.eye(n_coeff, dtype=float_dtype())
    t0_B = time.perf_counter()
    B = jax.vmap(_intensity_from_yvec_jit)(eye).T  # (n_pix, n_coeff)
    _ = B.block_until_ready()
    logger.info(f"Design matrix built in {time.perf_counter() - t0_B:.2f} s; shape={tuple(B.shape)}")

    Bw = w_sqrt[:, None] * B
    ridge = jnp.asarray(cfg.projector_ridge, dtype=float_dtype())
    gram = Bw.T @ Bw + ridge * jnp.eye(n_coeff, dtype=float_dtype())

    t0_proj = time.perf_counter()
    proj = jnp.linalg.solve(gram, Bw.T)  # (n_coeff, n_pix)
    _ = proj.block_until_ready()
    logger.info(f"Projector built in {time.perf_counter() - t0_proj:.2f} s")
    return proj


if cfg.cache_projector and projector_cache_path.exists():
    try:
        cache = np.load(projector_cache_path)
        ok = (
            int(cache["ydeg"]) == int(cfg.ydeg)
            and int(cache["n_pix"]) == int(n_pix)
            and int(cache["n_coeff"]) == int(n_coeff)
        )
        if ok:
            proj_np = np.asarray(cache["projector"])
            projector = jnp.asarray(proj_np, dtype=float_dtype())
            logger.info(f"Loaded cached projector from: {projector_cache_path}")
        else:
            raise ValueError("projector cache metadata mismatch")
    except Exception as e:
        logger.warning(f"Failed to load projector cache ({e}); rebuilding.")
        projector = _build_projector()
else:
    projector = _build_projector()

if cfg.cache_projector and (not projector_cache_path.exists()):
    try:
        save_npz(
            projector_cache_path,
            ydeg=np.asarray(int(cfg.ydeg), dtype=np.int32),
            n_pix=np.asarray(int(n_pix), dtype=np.int32),
            n_coeff=np.asarray(int(n_coeff), dtype=np.int32),
            projector=np.asarray(projector, dtype=np_float_dtype()),
        )
        logger.info(f"Saved projector cache to: {projector_cache_path}")
    except Exception as e:
        logger.warning(f"Could not save projector cache: {e}")


def intensity_map_to_y_dense(I_map: jnp.ndarray) -> jnp.ndarray:
    """Convert SWAMP-grid intensity map (J,I) to dense starry coefficients."""
    I_flat = I_map.reshape(-1)
    I_mean = (w_pix * I_flat).sum() / w_pix.sum()
    I_rel = I_flat / I_mean

    y = projector @ (w_sqrt * I_rel)
    y = y / y[0]
    return y


# =============================================================================
# Orbit/system setup (once)
# =============================================================================


central = Central(radius=cfg.star_radius_rsun, mass=cfg.star_mass_msun)

RJUP_TO_RSUN = 0.10045  # adequate for toy demo

planet = Body(
    radius=cfg.planet_radius_rjup * RJUP_TO_RSUN,
    period=orbital_period_days,
    time_transit=cfg.time_transit_days,
    impact_param=cfg.impact_param,
)

# Planet-only emission: star amplitude 0, but star still exists for occultation.
# Some jaxoplanet versions require an explicit spherical harmonic map `y=...`
# even if amplitude=0 (the star contributes no flux).
try:
    star_surface = Surface(amplitude=jnp.asarray(0.0, dtype=float_dtype()), normalize=False)
except Exception:
    star_surface = Surface(
        y=Ylm({(0, 0): jnp.asarray(1.0, dtype=float_dtype())}),
        u=(),
        amplitude=jnp.asarray(0.0, dtype=float_dtype()),
        normalize=False,
    )

times_days = np.linspace(
    cfg.time_transit_days,
    cfg.time_transit_days + float(cfg.n_orbits_observed) * orbital_period_days,
    int(cfg.n_times),
    endpoint=False,
).astype(np_float_dtype())
times_days_jax = jnp.asarray(times_days, dtype=float_dtype())


# =============================================================================
# Inferred-parameter specification (build once, affects model + plots)
# =============================================================================


@dataclass(frozen=True)
class InferredParam:
    name: str
    label: str
    base_value: float
    log10_min: float
    log10_max: float
    apply_to: str          # "static" or "flags"
    field_name: str        # resolved dataclass field name
    scale_to_field: float  # multiply (linear param) by this before writing to dataclass field


def _log10_bounds_from_cfg(
    *,
    name: str,
    base_value: float,
    explicit_min: Optional[float],
    explicit_max: Optional[float],
    half_width_dex: float,
) -> Tuple[float, float]:
    if explicit_min is not None or explicit_max is not None:
        if explicit_min is None or explicit_max is None:
            raise ValueError(f"Prior bounds for {name} must set both min and max or neither.")
        lo = float(explicit_min)
        hi = float(explicit_max)
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError(f"Invalid prior bounds for {name}: min={explicit_min}, max={explicit_max}")
        return lo, hi

    if not (base_value > 0.0 and math.isfinite(base_value)):
        raise ValueError(f"Base value for {name} must be finite and > 0 to set default log10 prior.")
    c = math.log10(float(base_value))
    return (c - float(half_width_dex), c + float(half_width_dex))


def _resolve_field(obj: Any, candidates: Sequence[str], logical_name: str) -> str:
    for c in candidates:
        if hasattr(obj, c):
            return c
    raise RuntimeError(
        f"Could not resolve field for {logical_name}. Tried {list(candidates)} but "
        f"{type(obj).__name__} has none of them."
    )


# Build the list of inferred parameters in a fixed order.
_INFERRED: List[InferredParam] = []

# Tau parameters (hours) -> static fields (seconds)
_tau_lo = float(cfg.prior_log10_tau_hours_min)
_tau_hi = float(cfg.prior_log10_tau_hours_max)
if not (math.isfinite(_tau_lo) and math.isfinite(_tau_hi) and _tau_hi > _tau_lo):
    raise ValueError("cfg.prior_log10_tau_hours_min/max must be finite with max > min")

if bool(cfg.infer_taurad):
    _INFERRED.append(
        InferredParam(
            name="taurad_hours",
            label=r"$\tau_{\rm rad}$ [hours]",
            base_value=float(cfg.taurad_true_hours),
            log10_min=_tau_lo,
            log10_max=_tau_hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["taurad"], "taurad"),
            scale_to_field=float(tau_hours_to_seconds(1.0)),
        )
    )

if bool(cfg.infer_taudrag):
    _INFERRED.append(
        InferredParam(
            name="taudrag_hours",
            label=r"$\tau_{\rm drag}$ [hours]",
            base_value=float(cfg.taudrag_true_hours),
            log10_min=_tau_lo,
            log10_max=_tau_hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["taudrag"], "taudrag"),
            scale_to_field=float(tau_hours_to_seconds(1.0)),
        )
    )

# Other SWAMP static parameters (positive; log10-uniform)
if bool(cfg.infer_a_planet_m):
    lo, hi = _log10_bounds_from_cfg(
        name="a_planet_m",
        base_value=float(cfg.a_planet_m),
        explicit_min=cfg.prior_log10_a_planet_m_min,
        explicit_max=cfg.prior_log10_a_planet_m_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="a_planet_m",
            label=r"$a$ [m]",
            base_value=float(cfg.a_planet_m),
            log10_min=lo,
            log10_max=hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["a", "a_planet_m"], "a_planet_m"),
            scale_to_field=1.0,
        )
    )

if bool(cfg.infer_omega_rad_s):
    lo, hi = _log10_bounds_from_cfg(
        name="omega_rad_s",
        base_value=float(cfg.omega_rad_s),
        explicit_min=cfg.prior_log10_omega_rad_s_min,
        explicit_max=cfg.prior_log10_omega_rad_s_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="omega_rad_s",
            label=r"$\omega$ [rad s$^{-1}$]",
            base_value=float(cfg.omega_rad_s),
            log10_min=lo,
            log10_max=hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["omega", "omega_rad_s"], "omega_rad_s"),
            scale_to_field=1.0,
        )
    )

if bool(cfg.infer_g_m_s2):
    lo, hi = _log10_bounds_from_cfg(
        name="g_m_s2",
        base_value=float(cfg.g_m_s2),
        explicit_min=cfg.prior_log10_g_m_s2_min,
        explicit_max=cfg.prior_log10_g_m_s2_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="g_m_s2",
            label=r"$g$ [m s$^{-2}$]",
            base_value=float(cfg.g_m_s2),
            log10_min=lo,
            log10_max=hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["g", "g_m_s2"], "g_m_s2"),
            scale_to_field=1.0,
        )
    )

if bool(cfg.infer_Phibar):
    lo, hi = _log10_bounds_from_cfg(
        name="Phibar",
        base_value=float(cfg.Phibar),
        explicit_min=cfg.prior_log10_Phibar_min,
        explicit_max=cfg.prior_log10_Phibar_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="Phibar",
            label=r"$\bar{\Phi}$",
            base_value=float(cfg.Phibar),
            log10_min=lo,
            log10_max=hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["Phibar"], "Phibar"),
            scale_to_field=1.0,
        )
    )

if bool(cfg.infer_DPhieq):
    lo, hi = _log10_bounds_from_cfg(
        name="DPhieq",
        base_value=float(cfg.DPhieq),
        explicit_min=cfg.prior_log10_DPhieq_min,
        explicit_max=cfg.prior_log10_DPhieq_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="DPhieq",
            label=r"$\Delta \Phi_{\rm eq}$",
            base_value=float(cfg.DPhieq),
            log10_min=lo,
            log10_max=hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["DPhieq"], "DPhieq"),
            scale_to_field=1.0,
        )
    )

if bool(cfg.infer_K6):
    lo, hi = _log10_bounds_from_cfg(
        name="K6",
        base_value=float(cfg.K6),
        explicit_min=cfg.prior_log10_K6_min,
        explicit_max=cfg.prior_log10_K6_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="K6",
            label=r"$K_6$",
            base_value=float(cfg.K6),
            log10_min=lo,
            log10_max=hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["K6"], "K6"),
            scale_to_field=1.0,
        )
    )

if bool(cfg.infer_K6Phi):
    if cfg.K6Phi is None:
        raise ValueError("cfg.infer_K6Phi=True requires cfg.K6Phi to be set (not None).")
    lo, hi = _log10_bounds_from_cfg(
        name="K6Phi",
        base_value=float(cfg.K6Phi),
        explicit_min=cfg.prior_log10_K6Phi_min,
        explicit_max=cfg.prior_log10_K6Phi_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="K6Phi",
            label=r"$K_{6\Phi}$",
            base_value=float(cfg.K6Phi),
            log10_min=lo,
            log10_max=hi,
            apply_to="static",
            field_name=_resolve_field(static_base, ["K6Phi"], "K6Phi"),
            scale_to_field=1.0,
        )
    )

# RunFlag alpha (positive; log10-uniform)
if bool(cfg.infer_alpha):
    lo, hi = _log10_bounds_from_cfg(
        name="alpha",
        base_value=float(cfg.alpha),
        explicit_min=cfg.prior_log10_alpha_min,
        explicit_max=cfg.prior_log10_alpha_max,
        half_width_dex=float(cfg.other_param_log10_half_width_dex),
    )
    _INFERRED.append(
        InferredParam(
            name="alpha",
            label=r"$\alpha$",
            base_value=float(cfg.alpha),
            log10_min=lo,
            log10_max=hi,
            apply_to="flags",
            field_name=_resolve_field(flags_base, ["alpha"], "alpha"),
            scale_to_field=1.0,
        )
    )

infer_dim = int(len(_INFERRED))
if bool(cfg.run_inference) and infer_dim < 1:
    raise RuntimeError("No inferred parameters were constructed; check infer_* switches and priors.")

logger.info(f"Inference parameter dim: {infer_dim}")
if infer_dim > 0:
    logger.info("Inferred parameters (order): " + ", ".join([p.name for p in _INFERRED]))

# Pre-build prior arrays in log10 space
_prior_log10_lo = jnp.asarray([p.log10_min for p in _INFERRED], dtype=float_dtype())
_prior_log10_w = jnp.asarray([p.log10_max - p.log10_min for p in _INFERRED], dtype=float_dtype())

_param_names = [p.name for p in _INFERRED]
_param_labels = [p.label for p in _INFERRED]

_static_bindings: List[Tuple[str, int, float]] = [
    (p.field_name, i, float(p.scale_to_field)) for i, p in enumerate(_INFERRED) if p.apply_to == "static"
]
_flags_bindings: List[Tuple[str, int, float]] = [
    (p.field_name, i, float(p.scale_to_field)) for i, p in enumerate(_INFERRED) if p.apply_to == "flags"
]

# Custom-gradient decision
_use_custom_gradients = bool(cfg.use_custom_gradients) and (infer_dim <= int(cfg.custom_grad_max_dim))
if bool(cfg.use_custom_gradients) and not _use_custom_gradients:
    logger.warning(
        f"cfg.use_custom_gradients=True but infer_dim={infer_dim} > cfg.custom_grad_max_dim={cfg.custom_grad_max_dim}; "
        "disabling custom gradients (using reverse-mode)."
    )


# =============================================================================
# SWAMP forward model (terminal Phi) driven by inferred parameters
# =============================================================================


def log10_theta_from_u(u: jnp.ndarray) -> jnp.ndarray:
    """Map unconstrained u -> log10(theta) within configured bounds."""
    z = jax.nn.sigmoid(u)
    return _prior_log10_lo + _prior_log10_w * z


def theta_from_u(u: jnp.ndarray) -> jnp.ndarray:
    """Map unconstrained u -> theta (linear units, positive)."""
    log10_th = log10_theta_from_u(u)
    ten = jnp.asarray(10.0, dtype=float_dtype())
    return jnp.power(ten, log10_th)


def log_prior_u(u: jnp.ndarray) -> jnp.ndarray:
    # Uniform prior in log10(theta) under the sigmoid transform:
    # p(u) ∝ sigmoid(u) * (1 - sigmoid(u)), independent per-dimension.
    return jnp.sum(jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))


def _static_and_flags_from_theta(theta: jnp.ndarray) -> Tuple[Any, Any]:
    static_updates: Dict[str, Any] = {k: theta[i] * s for (k, i, s) in _static_bindings}
    flags_updates: Dict[str, Any] = {k: theta[i] * s for (k, i, s) in _flags_bindings}

    static = _replace_dataclass(static_base, static_updates, name="static")
    flags = _replace_dataclass(flags_base, flags_updates, name="flags")
    return static, flags


def swamp_terminal_phi_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """Run SWAMP for the configured integration length and return terminal Phi map (J,I)."""
    static, flags = _static_and_flags_from_theta(theta)
    eta0, delta0, U0, V0, Phi0 = init_rest_fields(static)
    state0 = init_state_from_fields(static=static, flags=flags, eta0=eta0, delta0=delta0, Phi0=Phi0, U0=U0, V0=V0)

    sim_last = getattr(swamp_model, "simulate_scan_last", None)
    if sim_last is not None:
        kwargs = dict(
            static=static,
            flags=flags,
            state0=state0,
            t_seq=t_seq,
            test=None,
            Uic=U0,
            Vic=V0,
            remat_step=False,
        )
        last_state = call_with_filtered_kwargs(sim_last, kwargs, name="simulate_scan_last")
        return getattr(last_state, "Phi_curr")

    # Fallback: fori_loop stepping (slower, but avoids storing full trajectory)
    step_fn = getattr(swamp_model, "_step_once_state_only", None)
    if step_fn is None:
        raise RuntimeError("Neither simulate_scan_last nor _step_once_state_only found; cannot run SWAMP forward.")

    def body(i: int, st: Any) -> Any:
        t = t_seq[i]
        return step_fn(st, t, static, flags, None, U0, V0)

    state_f = jax.lax.fori_loop(0, int(t_seq.shape[0]), body, state0)
    return getattr(state_f, "Phi_curr")


def phase_curve_model_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    """Full forward model: theta -> phase curve (planet flux only)."""
    phi = swamp_terminal_phi_from_theta(theta)

    T = phi_to_temperature(phi)
    I_map = temperature_to_intensity(T)

    y_dense = intensity_map_to_y_dense(I_map)
    ylm = ylm_from_dense(y_dense, lm_list)

    planet_surface = Surface(
        y=ylm,
        u=(),
        inc=jnp.asarray(cfg.map_inc_rad, dtype=float_dtype()),
        obl=jnp.asarray(cfg.map_obl_rad, dtype=float_dtype()),
        period=jnp.asarray(orbital_period_days, dtype=float_dtype()),
        phase=jnp.asarray(cfg.phase_at_transit_rad, dtype=float_dtype()),
        amplitude=jnp.asarray(cfg.planet_fpfs, dtype=float_dtype()),
        normalize=False,
    )

    system = SurfaceSystem(
        central=central,
        central_surface=star_surface,
        bodies=((planet, planet_surface),),
    )

    lc = starry_light_curve(system)(times_days_jax)  # (n_times, 2): [star, planet]
    return lc[:, 1]


phase_curve_model_theta_jit = jax.jit(phase_curve_model_from_theta)

# Compile forward model once (on "truth" theta for the inferred subset)
if infer_dim > 0:
    theta_truth = jnp.asarray([p.base_value for p in _INFERRED], dtype=float_dtype())
else:
    theta_truth = jnp.zeros((0,), dtype=float_dtype())

logger.info("JIT compiling forward model (first call can be slow)...")
t0_compile = time.perf_counter()
_ = phase_curve_model_theta_jit(theta_truth).block_until_ready()
logger.info(f"Forward model compiled in {time.perf_counter() - t0_compile:.2f} s")


# =============================================================================
# Observations (synthetic by default)
# =============================================================================


obs_path = cfg.out_dir / "observations_hmc.npz"

if bool(cfg.generate_synthetic_data) or (not obs_path.exists()):
    logger.info("Generating synthetic observations from SWAMP+starry truth...")

    flux_true = np.asarray(phase_curve_model_theta_jit(theta_truth)).astype(np_float_dtype())

    rng = np.random.default_rng(int(cfg.seed))
    noise = rng.normal(0.0, float(cfg.obs_sigma), size=flux_true.shape).astype(np_float_dtype())
    flux_obs = flux_true + noise

    save_npz(
        obs_path,
        times_days=times_days,
        flux_true=flux_true,
        flux_obs=flux_obs,
        obs_sigma=float(cfg.obs_sigma),
        orbital_period_days=float(orbital_period_days),
    )
    logger.info(f"Saved observations to: {obs_path}")
else:
    d = np.load(obs_path)
    times_days = np.asarray(d["times_days"])
    times_days_jax = jnp.asarray(times_days, dtype=float_dtype())
    flux_true = np.asarray(d["flux_true"])
    flux_obs = np.asarray(d["flux_obs"])
    logger.info(f"Loaded observations from: {obs_path}")

imin = int(np.argmin(flux_true))
imax = int(np.argmax(flux_true))
logger.info(
    f"Truth flux: min={flux_true[imin]:.3e} at t={times_days[imin]:.5f} d, "
    f"max={flux_true[imax]:.3e} at t={times_days[imax]:.5f} d"
)

flux_obs_jax = jnp.asarray(flux_obs, dtype=float_dtype())
obs_sigma_jax = jnp.asarray(float(cfg.obs_sigma), dtype=float_dtype())


# =============================================================================
# Posterior: logdensity + custom gradients (optional)
# =============================================================================


samples_path = cfg.out_dir / "posterior_samples_hmc.npz"
diag_path = cfg.out_dir / "diagnostics_hmc.npz"


def _log_likelihood_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
    mu = phase_curve_model_theta_jit(theta)
    finite = jnp.all(jnp.isfinite(mu))

    def _ok() -> jnp.ndarray:
        resid = (flux_obs_jax - mu) / obs_sigma_jax
        n = mu.size
        return -0.5 * jnp.sum(resid * resid) - n * jnp.log(obs_sigma_jax) - 0.5 * n * jnp.log(
            jnp.asarray(2.0 * math.pi, dtype=float_dtype())
        )

    def _bad() -> jnp.ndarray:
        return jnp.asarray(-1.0e30, dtype=float_dtype())

    return jax.lax.cond(finite, _ok, _bad)


def log_likelihood_u(u: jnp.ndarray) -> jnp.ndarray:
    theta = theta_from_u(u)
    return _log_likelihood_from_theta(theta)


def log_posterior_u(u: jnp.ndarray) -> jnp.ndarray:
    return log_prior_u(u) + log_likelihood_u(u)


def _value_and_grad_fwd(fun, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (fun(x), grad fun(x)) using forward-mode JVPs (good for tiny dim)."""
    x = jnp.asarray(x)
    n = int(x.shape[0])
    eye = jnp.eye(n, dtype=x.dtype)

    y0, dy0 = jax.jvp(fun, (x,), (eye[0],))
    if n == 1:
        return y0, jnp.atleast_1d(dy0)

    def jvp_dir(v: jnp.ndarray) -> jnp.ndarray:
        _, dy = jax.jvp(fun, (x,), (v,))
        return dy

    dy_rest = jax.vmap(jvp_dir)(eye[1:])
    grad = jnp.concatenate([jnp.atleast_1d(dy0), dy_rest], axis=0)
    return y0, grad


"""
def _value_and_grad_fwd(fun, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Compute (fun(x), grad fun(x)) in one batched JVP pass (often memory-friendlier than linearize).
    x = jnp.asarray(x)
    eye = jnp.eye(x.shape[0], dtype=x.dtype)

    def jvp_dir(v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        y, dy = jax.jvp(fun, (x,), (v,))
        return y, dy

    y_batched, grad = jax.vmap(jvp_dir)(eye)
    return y_batched[0], grad
"""


if _use_custom_gradients:

    @jax.custom_vjp
    def logdensity_fn(u: jnp.ndarray) -> jnp.ndarray:
        return log_posterior_u(u)

    def _lp_fwd(u: jnp.ndarray):
        val, grad = _value_and_grad_fwd(log_posterior_u, u)
        return val, grad

    def _lp_bwd(grad: jnp.ndarray, g: jnp.ndarray):
        return (g * grad,)

    logdensity_fn.defvjp(_lp_fwd, _lp_bwd)
    logger.info("Using custom VJP (forward-mode JVP) for log-posterior gradients.")
else:
    logdensity_fn = log_posterior_u
    logger.info("Using default reverse-mode gradients for log-posterior.")


def sample_prior_u(rng_key: jax.Array, n: int) -> jax.Array:
    """Sample u ~ logistic(Uniform(0,1)) which corresponds to bounded-uniform priors in log10(theta)."""
    eps = jnp.asarray(1e-6, dtype=float_dtype())
    z = jax.random.uniform(rng_key, shape=(int(n), int(infer_dim)), minval=eps, maxval=1.0 - eps)
    return jnp.log(z) - jnp.log1p(-z)


# =============================================================================
# NUTS kernel + warmup/sampling utilities (all JAX scans)
# =============================================================================


_nuts_kernel = blackjax.nuts.build_kernel()


def _nuts_step_single(rng_key: jax.Array, state: Any, step_size: jnp.ndarray, mass_matrix_diag: jnp.ndarray):
    # BlackJAX expects the "inverse_mass_matrix" argument, but the diagonal metric it uses is
    # the *covariance-like* scaling (what window_adaptation calls inverse_mass_matrix).
    # For a diagonal metric, provide an estimate of Var(u) (not 1/Var(u)).
    return _nuts_kernel(
        rng_key,
        state,
        logdensity_fn,
        step_size,
        mass_matrix_diag,
        max_num_doublings=int(cfg.nuts_max_num_doublings),
    )


def _init_nuts_state(position_u: jnp.ndarray) -> Any:
    return blackjax.nuts.init(position_u, logdensity_fn)


@dataclass(frozen=True)
class DualAveragingParams:
    target_accept: float
    gamma: float
    t0: float
    kappa: float


_DA_PARAMS = DualAveragingParams(
    target_accept=float(cfg.target_accept),
    gamma=float(cfg.da_gamma),
    t0=float(cfg.da_t0),
    kappa=float(cfg.da_kappa),
)


@jax.jit
def _da_init(step_size0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize dual averaging state (batched over chains).

    Returns:
      log_step_size, log_step_size_bar, h_bar, step, mu
    """
    step_size0 = jnp.asarray(step_size0, dtype=float_dtype())
    log_eps0 = jnp.log(step_size0)
    return (
        log_eps0,
        jnp.zeros_like(log_eps0),
        jnp.zeros_like(log_eps0),
        jnp.zeros_like(log_eps0, dtype=jnp.int32),
        jnp.log(jnp.asarray(10.0, dtype=float_dtype()) * step_size0),
    )


@jax.jit
def _da_update(
    da_state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    accept_rate: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    log_eps, log_eps_bar, h_bar, step, mu = da_state

    step = step + jnp.asarray(1, dtype=jnp.int32)
    m = step.astype(float_dtype())

    w = 1.0 / (m + jnp.asarray(_DA_PARAMS.t0, dtype=float_dtype()))
    h_bar = (1.0 - w) * h_bar + w * (jnp.asarray(_DA_PARAMS.target_accept, dtype=float_dtype()) - accept_rate)

    log_eps = mu - (jnp.sqrt(m) / jnp.asarray(_DA_PARAMS.gamma, dtype=float_dtype())) * h_bar

    m_kappa = m ** (-jnp.asarray(_DA_PARAMS.kappa, dtype=float_dtype()))
    log_eps_bar = m_kappa * log_eps + (1.0 - m_kappa) * log_eps_bar

    return log_eps, log_eps_bar, h_bar, step, mu


@jax.jit
def _welford_init(init_positions: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-safe Welford initialization.

    init_positions: (num_chains, dim)
    Returns:
        count: (num_chains,)
        mean:  (num_chains, dim)
        m2:    (num_chains, dim)
    """
    if init_positions.ndim != 2:
        raise ValueError(f"_welford_init expects (num_chains, dim), got shape {init_positions.shape}")

    num_chains = init_positions.shape[0]
    dim = init_positions.shape[1]
    dtype = init_positions.dtype

    count = jnp.zeros((num_chains,), dtype=jnp.int32)
    mean = jnp.zeros((num_chains, dim), dtype=dtype)
    m2 = jnp.zeros((num_chains, dim), dtype=dtype)
    return (count, mean, m2)


@jax.jit
def _welford_update(
    w_state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    x: jnp.ndarray,  # (num_chains, dim)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    count, mean, m2 = w_state
    count_new = count + jnp.asarray(1, dtype=jnp.int32)
    count_f = count_new.astype(float_dtype())[:, None]

    delta = x - mean
    mean_new = mean + delta / count_f
    delta2 = x - mean_new
    m2_new = m2 + delta * delta2
    return (count_new, mean_new, m2_new)


def _extract_info_field(info: Any, name: str, default: Any) -> Any:
    """Extract a field from a BlackJAX info pytree.

    BlackJAX info objects are typically NamedTuples/dataclasses, but in some
    versions they may be dict-like. This helper supports both.
    """
    if isinstance(info, Mapping):
        return info.get(name, default)
    return getattr(info, name, default)


def warmup_and_sample_nuts(
    rng_key: jax.Array,
    init_positions_u: jnp.ndarray,  # (num_chains, dim)
) -> Tuple[
    jnp.ndarray,  # u_samples: (num_chains, num_samples, dim)
    Dict[str, jnp.ndarray],  # diagnostics (device arrays)
]:
    num_chains = int(init_positions_u.shape[0])
    dim = int(init_positions_u.shape[1])

    # --- init states (batched) ---
    init_states = jax.vmap(_init_nuts_state)(init_positions_u)

    # --- warmup split ---
    n_warmup = int(cfg.num_warmup)
    n1 = int(max(0, min(n_warmup, int(round(n_warmup * float(cfg.warmup_phase1_frac))))))
    n2 = int(n_warmup - n1)

    collect_start = int(max(0, min(n1, int(round(n1 * float(cfg.warmup_mass_matrix_collect_frac))))))

    # --- phase 1: step-size adaptation + mass matrix estimation ---
    step_size0 = jnp.full((num_chains,), float(cfg.nuts_initial_step_size), dtype=float_dtype())
    da = _da_init(step_size0)
    welford = _welford_init(init_positions_u)

    def phase1_body(carry, xs):
        states, da_state, w_state = carry
        key_t, i_t = xs

        step_sizes = jnp.exp(da_state[0])  # current log_step_size
        mass_diag = jnp.ones((num_chains, dim), dtype=float_dtype())

        keys = jax.random.split(key_t, num_chains)
        states, infos = jax.vmap(_nuts_step_single)(keys, states, step_sizes, mass_diag)

        acc = _extract_info_field(infos, "acceptance_rate", jnp.zeros((num_chains,), dtype=float_dtype()))
        acc = jnp.asarray(acc, dtype=float_dtype())
        da_state = _da_update(da_state, acc)

        def _do_update(ws):
            return _welford_update(ws, states.position)

        w_state = jax.lax.cond(i_t >= collect_start, _do_update, lambda ws: ws, w_state)
        return (states, da_state, w_state), None

    if n1 > 0:
        keys1 = jax.random.split(rng_key, n1 + 1)
        rng_key_phase2 = keys1[-1]
        keys1 = keys1[:-1]
        idx1 = jnp.arange(n1, dtype=jnp.int32)
        (states1, da1, welford1), _ = jax.lax.scan(phase1_body, (init_states, da, welford), (keys1, idx1))
    else:
        states1, da1, welford1 = init_states, da, welford
        rng_key_phase2 = rng_key

    # Diagonal covariance estimate (u-space)
    count, mean, m2 = welford1
    count_f = count.astype(float_dtype())
    denom = jnp.maximum(count_f - 1.0, 1.0)
    var_raw = m2 / denom[:, None]
    # If we collected too few samples to estimate variance, fall back to unit variance.
    var = jnp.where(count_f[:, None] > 1.0, var_raw, jnp.ones_like(var_raw))
    var = jnp.maximum(var, jnp.asarray(float(cfg.mass_matrix_var_floor), dtype=float_dtype()))

    # BlackJAX's diagonal metric uses a covariance-like scaling. Use Var(u), not 1/Var(u).
    mass_diag_est = var

    # --- phase 2: final step-size tuning using estimated mass matrix ---
    step_size1 = jnp.exp(da1[1])  # log_step_size_bar
    step_size1 = jnp.maximum(step_size1, jnp.asarray(1e-8, dtype=float_dtype()))
    da2 = _da_init(step_size1)

    def phase2_body(carry, key_t):
        states, da_state = carry
        step_sizes = jnp.exp(da_state[0])
        keys = jax.random.split(key_t, num_chains)
        states, infos = jax.vmap(_nuts_step_single)(keys, states, step_sizes, mass_diag_est)

        acc = _extract_info_field(infos, "acceptance_rate", jnp.zeros((num_chains,), dtype=float_dtype()))
        acc = jnp.asarray(acc, dtype=float_dtype())
        da_state = _da_update(da_state, acc)
        return (states, da_state), None

    if n2 > 0:
        keys2 = jax.random.split(rng_key_phase2, n2 + 1)
        rng_key_sample = keys2[-1]
        keys2 = keys2[:-1]
        (states2, da2_final), _ = jax.lax.scan(phase2_body, (states1, da2), keys2)
    else:
        states2, da2_final = states1, da2
        rng_key_sample = rng_key_phase2

    step_sizes_final = jnp.exp(da2_final[1])  # averaged log_step_size_bar
    step_sizes_final = jnp.maximum(step_sizes_final, jnp.asarray(1e-8, dtype=float_dtype()))

    # --- sampling ---
    num_samples = int(cfg.num_samples)

    def sample_body(states, key_t):
        keys = jax.random.split(key_t, num_chains)
        states, infos = jax.vmap(_nuts_step_single)(keys, states, step_sizes_final, mass_diag_est)

        acc = _extract_info_field(infos, "acceptance_rate", jnp.full((num_chains,), jnp.nan, dtype=float_dtype()))
        div = _extract_info_field(infos, "is_divergent", jnp.zeros((num_chains,), dtype=jnp.bool_))
        n_leap = _extract_info_field(infos, "num_integration_steps", jnp.full((num_chains,), -1, dtype=jnp.int32))

        return states, (states.position, states.logdensity, acc, div, n_leap)

    keys_s = jax.random.split(rng_key_sample, num_samples)
    _, (pos_s, logd_s, acc_s, div_s, nleap_s) = jax.lax.scan(sample_body, states2, keys_s)

    # pos_s: (num_samples, num_chains, dim) -> (num_chains, num_samples, dim)
    u_samples = jnp.transpose(pos_s, (1, 0, 2))
    diag = {
        "step_size": step_sizes_final,
        "inverse_mass_matrix": mass_diag_est,
        "logdensity": jnp.transpose(logd_s, (1, 0)),
        "acceptance_rate": jnp.transpose(acc_s, (1, 0)),
        "is_divergent": jnp.transpose(div_s, (1, 0)),
        "num_integration_steps": jnp.transpose(nleap_s, (1, 0)),
    }
    return u_samples, diag


warmup_and_sample_nuts_jit = jax.jit(warmup_and_sample_nuts)


# =============================================================================
# Inference run
# =============================================================================


if bool(cfg.run_inference):
    logger.info(
        f"Running NUTS inference: {cfg.num_chains} chains × ({cfg.num_warmup} warmup + {cfg.num_samples} samples)"
    )

    key = jax.random.PRNGKey(int(cfg.seed))
    key, subkey = jax.random.split(key)
    init_positions_u = sample_prior_u(subkey, int(cfg.num_chains))  # (num_chains, dim)

    # Compile the full warmup+sampling function once (best walltime behavior on GPU)
    logger.info("Compiling NUTS warmup+sampling function (first call triggers XLA compile)...")
    t0 = time.perf_counter()
    did_lower = False
    compiled_output = None
    try:
        warmup_and_sample_nuts_jit.lower(rng_key=key, init_positions_u=init_positions_u).compile()
        did_lower = True
    except Exception:
        # Fallback for older JAX: execute once to trigger compilation and reuse the result.
        compiled_output = warmup_and_sample_nuts_jit(key, init_positions_u)
        jax.block_until_ready(compiled_output[0])
    logger.info(
        f"NUTS warmup+sampling compilation finished in {time.perf_counter() - t0:.1f} s "
        f"(lower={did_lower})"
    )

    # Execute inference (or reuse the fallback run)
    if compiled_output is None:
        logger.info("Running NUTS warmup+sampling...")
        t1 = time.perf_counter()
        u_samples, diag = warmup_and_sample_nuts_jit(key, init_positions_u)
        jax.block_until_ready(u_samples)
        logger.info(f"Inference finished in {time.perf_counter() - t1:.1f} s")
    else:
        u_samples, diag = compiled_output
        logger.info("Warmup+sampling executed once during compilation fallback; reusing results.")

    # Convert to theta space (log10 units and linear units)
    theta_log10 = jax.vmap(jax.vmap(log10_theta_from_u))(u_samples)  # (chains, samples, dim)
    theta_lin = jnp.power(jnp.asarray(10.0, dtype=float_dtype()), theta_log10)

    theta_log10_np = np.asarray(theta_log10, dtype=np.float64)
    theta_lin_np = np.asarray(theta_lin, dtype=np.float64)

    # Always save tau arrays for convenience/compatibility, even if not inferred.
    # If not inferred, they are constant arrays at the configured truth.
    taurad_hours_arr = np.full((int(cfg.num_chains), int(cfg.num_samples)), float(cfg.taurad_true_hours), dtype=np.float64)
    taudrag_hours_arr = np.full((int(cfg.num_chains), int(cfg.num_samples)), float(cfg.taudrag_true_hours), dtype=np.float64)

    for i, p in enumerate(_INFERRED):
        if p.name == "taurad_hours":
            taurad_hours_arr = theta_lin_np[..., i]
        elif p.name == "taudrag_hours":
            taudrag_hours_arr = theta_lin_np[..., i]

    out_samples: Dict[str, Any] = dict(
        param_names=np.asarray(_param_names, dtype="<U64"),
        param_labels=np.asarray(_param_labels, dtype="<U128"),
        log10_param_samples=theta_log10_np,
        param_samples=theta_lin_np,
        u_samples=np.asarray(u_samples, dtype=np.float64),
        taurad_hours=taurad_hours_arr,
        taudrag_hours=taudrag_hours_arr,
    )

    # Also save each inferred parameter as its own array for convenience
    for i, name in enumerate(_param_names):
        out_samples[name] = theta_lin_np[..., i]
        out_samples[f"log10_{name}"] = theta_log10_np[..., i]

    save_npz(samples_path, **out_samples)
    logger.info(f"Saved posterior samples to: {samples_path}")

    # Save diagnostics (host arrays)
    diag_np = {k: np.asarray(jax.device_get(v)) for k, v in diag.items()}
    n_div = int(np.sum(diag_np["is_divergent"]))
    mean_acc = float(np.nanmean(diag_np["acceptance_rate"]))
    mean_steps = float(np.mean(diag_np["num_integration_steps"][diag_np["num_integration_steps"] > 0])) if np.any(
        diag_np["num_integration_steps"] > 0
    ) else float("nan")

    logger.info(f"Divergent transitions: {n_div}/{int(cfg.num_chains) * int(cfg.num_samples)}")
    logger.info(f"Mean acceptance rate: {mean_acc:.3f}")
    logger.info(f"Mean num_integration_steps: {mean_steps:.1f}")

    save_npz(
        diag_path,
        inference_method=np.asarray("nuts", dtype="<U16"),
        num_chains=np.asarray(int(cfg.num_chains), dtype=np.int32),
        num_warmup=np.asarray(int(cfg.num_warmup), dtype=np.int32),
        num_samples=np.asarray(int(cfg.num_samples), dtype=np.int32),
        step_size=np.asarray(diag_np["step_size"], dtype=np.float64),
        inverse_mass_matrix=np.asarray(diag_np["inverse_mass_matrix"], dtype=np.float64),
        logdensity=np.asarray(diag_np["logdensity"], dtype=np.float64),
        acceptance_rate=np.asarray(diag_np["acceptance_rate"], dtype=np.float64),
        is_divergent=np.asarray(diag_np["is_divergent"], dtype=np.bool_),
        num_integration_steps=np.asarray(diag_np["num_integration_steps"], dtype=np.int32),
        u_samples=np.asarray(np.asarray(u_samples, dtype=np.float64), dtype=np.float64),
        param_names=np.asarray(_param_names, dtype="<U64"),
        param_samples=theta_lin_np,
    )
    logger.info(f"Saved HMC diagnostics to: {diag_path}")

else:
    logger.info("cfg.run_inference=False; skipping inference.")
    if not samples_path.exists():
        raise FileNotFoundError(f"posterior_samples_hmc.npz not found at {samples_path}")


# =============================================================================
# Posterior predictive (optional)
# =============================================================================


ppc_path = cfg.out_dir / "posterior_predictive_hmc.npz"
ppc_quant_path = cfg.out_dir / "posterior_predictive_quantiles_hmc.npz"

if bool(cfg.do_ppc):
    logger.info("Computing posterior predictive phase curves (subset of draws)...")

    s = np.load(samples_path)
    theta = np.asarray(s["param_samples"], dtype=np_float_dtype())  # (chains, samples, dim)
    theta_flat = theta.reshape(-1, theta.shape[-1])
    n_available = theta_flat.shape[0]

    rng = np.random.default_rng(int(cfg.seed) + 1)
    n_take = min(int(cfg.ppc_draws), int(n_available))
    take_idx = rng.choice(n_available, size=n_take, replace=False)

    theta_sel = theta_flat[take_idx].astype(np_float_dtype())  # (n_take, dim)
    theta_sel_jax = jnp.asarray(theta_sel, dtype=float_dtype())

    @jax.jit
    def _batch_forward(theta_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(phase_curve_model_theta_jit)(theta_batch)

    preds: List[np.ndarray] = []
    for i0 in tqdm(range(0, n_take, int(cfg.ppc_chunk_size)), desc="PPC batches"):
        i1 = min(i0 + int(cfg.ppc_chunk_size), n_take)
        preds.append(np.asarray(_batch_forward(theta_sel_jax[i0:i1])))

    ppc_draws = np.concatenate(preds, axis=0)  # (n_take, n_times)
    ppc_q = {
        "p05": np.quantile(ppc_draws, 0.05, axis=0),
        "p50": np.quantile(ppc_draws, 0.50, axis=0),
        "p95": np.quantile(ppc_draws, 0.95, axis=0),
    }

    save_npz(ppc_path, ppc_draws=ppc_draws, theta_sel=theta_sel, times_days=times_days, param_names=np.asarray(_param_names))
    save_npz(ppc_quant_path, **ppc_q, times_days=times_days)
    logger.info(f"Saved PPC draws to: {ppc_path}")
    logger.info(f"Saved PPC quantiles to: {ppc_quant_path}")
else:
    logger.info("cfg.do_ppc=False; skipping posterior predictive.")


# =============================================================================
# Save truth + posterior-summary maps (so plotting does NOT rerun SWAMP)
# =============================================================================


maps_path = cfg.out_dir / "maps_truth_and_posterior_summary_hmc.npz"


def compute_maps_for_theta(theta: jnp.ndarray) -> Dict[str, np.ndarray]:
    phi = swamp_terminal_phi_from_theta(theta)
    T = phi_to_temperature(phi)
    I_map = temperature_to_intensity(T)
    y_dense = intensity_map_to_y_dense(I_map)

    return {
        "phi": np.asarray(phi),
        "T": np.asarray(T),
        "I": np.asarray(I_map),
        "y_dense": np.asarray(y_dense),
    }


logger.info("Computing truth and posterior-summary terminal maps...")

truth_maps = compute_maps_for_theta(theta_truth)

# Posterior median in inferred-parameter space
s = np.load(samples_path)
theta_samps = np.asarray(s["param_samples"], dtype=np.float64).reshape(-1, infer_dim)
theta_med = np.median(theta_samps, axis=0).astype(np.float64)
theta_med_jax = jnp.asarray(theta_med, dtype=float_dtype())

post_maps = compute_maps_for_theta(theta_med_jax)

save_npz(
    maps_path,
    lon=np.asarray(lon),
    lat=np.asarray(lat),
    phi_truth=truth_maps["phi"],
    T_truth=truth_maps["T"],
    I_truth=truth_maps["I"],
    y_truth=truth_maps["y_dense"],
    phi_post=post_maps["phi"],
    T_post=post_maps["T"],
    I_post=post_maps["I"],
    y_post=post_maps["y_dense"],
    param_names=np.asarray(_param_names, dtype="<U64"),
    theta_truth=np.asarray(np.asarray(theta_truth, dtype=np.float64), dtype=np.float64),
    theta_post_median=np.asarray(theta_med, dtype=np.float64),
)
logger.info(f"Saved truth + posterior-median maps to: {maps_path}")

logger.info("DONE.")
logger.info(f"Outputs saved to: {cfg.out_dir}")
