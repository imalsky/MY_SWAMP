#!/usr/bin/env python3
"""
nss.py

End-to-end differentiable SWAMP (my_swamp) + starry (jaxoplanet) retrieval with
BlackJAX Nested Sampling (NSS), including optional gradient-informed inner kernels.

What this script does
---------------------
1) Builds a SWAMP static model (grid, spectral basis, etc.) once to define the pixel grid.
2) Precomputes a robust linear projector from the SWAMP pixel grid -> starry spherical harmonics,
   using starry's own Surface.intensity() evaluation to avoid convention mismatches.
3) Defines a JAX-traceable forward model mapping a parameter vector -> SWAMP terminal Phi
   -> temperature proxy -> intensity map -> starry phase curve.
4) Generates synthetic observations (optional) and runs Bayesian inference with NSS.
5) Writes everything needed for plotting into cfg.out_dir; plot.py never reruns SWAMP.

Important design choices
------------------------
- The map projection can preserve physically meaningful amplitude information (recommended default)
  rather than forcing purely shape-only maps.
- NSS can run with either the standard slice inner sampler or a gradient-informed inner sampler
  (MALA/HMC) when your BlackJAX NS build exposes the adaptive NS APIs.
- Priors and inference toggles are parameter-by-parameter; non-inferred parameters remain fixed.

Notes on "infer all SWAMP inputs"
---------------------------------
This script supports inferring a superset of the continuous scalar SWAMP inputs (tau_rad, tau_drag,
Phibar, DPhieq, K6, K6Phi, omega, a, g) and several system parameters (planet radius, planet_fpfs).
However, shape-changing parameters (e.g., resolution M, time step dt, grid size) are intentionally
NOT supported for inference: changing them would change array shapes and invalidate JIT-compiled
functions and the precomputed starry projector.

The my_swamp README (document version 2026-02-13) notes that the forward simulation is differentiable
w.r.t these continuous scalars. This script assumes that the my_swamp build you are using preserves
that property.

Outputs written to cfg.out_dir
------------------------------
- config.json
- run.log
- observations.npz
- posterior_samples.npz
- mcmc_extra_fields.npz                  (NSS diagnostics; name kept for plot compatibility)
- posterior_predictive.npz               (optional)
- posterior_predictive_quantiles.npz     (optional)
- maps_truth_and_posterior_summary.npz   (truth + posterior-median maps; for plotting without reruns)

Run `plot.py` after this completes.
"""

from __future__ import annotations

import inspect
from functools import partial
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
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
    # NOTE: my_swamp defaults to enabling x64 at import time (see README). We keep an explicit switch.
    use_x64: bool = True
    xla_preallocate: bool = False

    # -----------------------
    # SWAMP numerical params (SHAPES) - NOT inferred
    # -----------------------
    M: int = 42
    dt_seconds: float = 240.0
    model_days: float = 50.0
    starttime_index: int = 2  # leapfrog start index (>=2)

    # -----------------------
    # SWAMP physical params (defaults / truth)
    # -----------------------
    # These can optionally be inferred via the infer_* flags below.
    a_planet_m: float = 8.2e7
    omega_rad_s: float = 3.2e-5
    g_m_s2: float = 9.8
    Phibar: float = 3.0e5
    DPhieq: float = 1.0e6
    K6: float = 1.24e33
    K6Phi: Optional[float] = None  # additional Phi diffusion; None disables

    # If True, always rebuild SWAMP static/state0 inside the forward model instead of using the
    # dataclasses.replace() fast path. This is a correctness/debug switch: if my_swamp caches
    # tau-dependent derived fields inside `static`, the fast path can silently freeze tau dependence.
    force_rebuild_static: bool = False

    # RunFlags (kept fixed; could be exposed if desired but usually not identifiable from a phase curve)
    forcflag: bool = True
    diffflag: bool = True
    expflag: bool = False
    modalflag: bool = True
    alpha: float = 0.01
    diagnostics: bool = False  # must be False for clean JAX
    blowup_rms: float = 1.0e30

    # -----------------------
    # Phi -> temperature -> intensity (toy emission layer)
    # -----------------------
    T_ref: float = 300.0
    phi_to_T_scale: float = 4.0e4
    Tmin_K: float = 1.0

    # Optional spectral dependence:
    #   emission_model="bolometric": I ∝ T^4
    #   emission_model="planck":     I ∝ B_λ(T) at planck_wavelength_m (up to a constant factor)
    emission_model: str = "bolometric"  # {"bolometric", "planck"}
    planck_wavelength_m: float = 4.5e-6  # meters (e.g., 4.5 µm)
    planck_x_clip: float = 80.0  # clip x = hc/(λ k T) to avoid overflow in expm1

    # -----------------------
    # starry / map projection
    # -----------------------
    ydeg: int = 10
    projector_ridge: float = 1.0e-6
    # "shape_only": historical behavior (normalize away global map amplitude)
    # "shape_plus_amplitude": preserve map monopole as an amplitude factor (recommended)
    map_projection_mode: str = "shape_plus_amplitude"  # {"shape_only", "shape_plus_amplitude"}

    # Map orientation (known)
    map_inc_rad: float = math.pi / 2
    map_obl_rad: float = 0.0

    # IMPORTANT phase convention:
    # At transit, observer sees nightside for a tidally locked planet.
    phase_at_transit_rad: float = math.pi

    # -----------------------
    # Orbit/system geometry (defaults / truth)
    # -----------------------
    star_mass_msun: float = 1.0
    star_radius_rsun: float = 1.0
    planet_radius_rjup: float = 1.0
    impact_param: float = 0.0
    time_transit_days: float = 0.0

    # If None: derived from omega_rad_s (synchronous assumption).
    # If not None: orbital period is fixed and omega controls only SWAMP dynamics.
    orbital_period_override_days: Optional[float] = None

    # Flux scaling: planet/star flux ratio (can optionally be inferred)
    planet_fpfs: float = 1500e-6

    # -----------------------
    # Synthetic observations
    # -----------------------
    generate_synthetic_data: bool = True
    n_times: int = 250
    n_orbits_observed: float = 1.0
    obs_sigma: float = 80e-6

    # True tau values used for synthetic data generation (hours)
    taurad_true_hours: float = 10.0
    taudrag_true_hours: float = 6.0

    # -----------------------
    # Inference toggles (requested feature)
    # -----------------------
    # Each parameter has:
    #   - infer_* bool
    #   - prior_* bounds (independent priors)
    #
    # IMPORTANT: avoid inferring parameters that change array shapes.
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

    # -----------------------
    # Priors
    # -----------------------
    # Tau priors (requested): Uniform in linear HOURS, independently configurable.
    prior_tau_rad_hours_min: float = 1.0
    prior_tau_rad_hours_max: float = 30.0
    prior_tau_drag_hours_min: float = 1.00
    prior_tau_drag_hours_max: float = 30.0

    # Other priors: default to broad but finite bounds.
    # NOTE: For wide positive parameters it is usually better to use a log prior, but for simplicity
    # we implement "uniform in log10(param)" via prior_type="log10_uniform".
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

    # Prior types for non-tau parameters:
    #   "uniform"       => Uniform(param)
    #   "log10_uniform" => Uniform(log10(param))  (i.e., log-uniform in param)
    prior_type_planet_radius: str = "uniform"
    prior_type_planet_fpfs: str = "log10_uniform"

    prior_type_Phibar: str = "log10_uniform"
    prior_type_DPhieq: str = "log10_uniform"
    prior_type_K6: str = "log10_uniform"
    prior_type_K6Phi: str = "log10_uniform"

    prior_type_omega: str = "log10_uniform"
    prior_type_a_planet: str = "log10_uniform"
    prior_type_g: str = "log10_uniform"

    # -----------------------
    # Inference
    # -----------------------
    run_inference: bool = True

    # Gradient handling:
    # If True, use a custom VJP for the likelihood based on forward-mode JVPs.
    use_custom_likelihood_gradients: bool = True
    custom_grad_max_dim: int = 8

    # Posterior draws to save after NS importance resampling.
    num_samples: int = 256
    num_chains: int = 4

    # -----------------------
    # Inference: BlackJAX Nested Sampling
    # -----------------------
    # Inner-kernel choice:
    # "slice": default BlackJAX nested slice sampler.
    # "mala"/"hmc": gradient-informed inner kernels via blackjax.ns.adaptive APIs (if available).
    ns_inner_kernel: str = "hmc"  # {"slice", "mala", "hmc"}

    # NOTE: Nested sampling typically needs many likelihood evaluations.
    ns_num_live: int = 768
    # If None, derived as round(ns_num_delete_frac * ns_num_live) (clipped to [1, ns_num_live-1])
    ns_num_delete: Optional[int] = None
    ns_num_delete_frac: float = 0.15
    # If None, derived as round(ns_inner_steps_mult * n_dim)
    ns_num_inner_steps: Optional[int] = None
    ns_inner_steps_mult: float = 20.0
    # Terminate when log(Z_live / Z_dead) < -ns_stop_delta_logz; default corresponds to 1e-3.
    ns_stop_delta_logz: float = 5.0
    # Hard safety caps
    ns_max_steps: int = 4096
    ns_max_dead_points: Optional[int] = None
    ns_max_seconds: Optional[float] = None
    # Bootstrap draws for rough logZ uncertainty estimate
    ns_logz_bootstrap: int = 256
    # Store dead points on host to reduce device memory pressure (can be slower)
    ns_store_dead_on_host: bool = False
    # Progress logging for NSS (outer-loop steps)
    ns_log_every: int = 5
    ns_eta_window: int = 5

    # Gradient inner-kernel parameters (used when ns_inner_kernel in {"mala","hmc"}).
    ns_mala_step_size: float = 0.18
    ns_hmc_step_size: float = 0.05
    ns_hmc_num_integration_steps: int = 12
    ns_hmc_adapt_mass_from_live: bool = True
    ns_hmc_mass_jitter: float = 1.0e-3
    ns_hmc_mass_min: float = 1.0e-3
    ns_hmc_mass_max: float = 1.0e3

    # -----------------------
    # Posterior predictive (optional)
    # -----------------------
    do_ppc: bool = True
    ppc_draws: int = 128
    ppc_chunk_size: int = 16

    # -----------------------
    # Plot-related config saved for the plot script
    # -----------------------
    fig_dpi: int = 160
    render_res: int = 250
    render_phases: Tuple[float, ...] = (0.0, 0.25, 0.49, 0.51, 0.75)

    # Plot preference: "many orders of magnitude" threshold for log axes
    log_axis_orders_threshold: float = 3.0


cfg = Config()


# =============================================================================
# Validate config early (fail fast)
# =============================================================================


_valid_emission_models = {"bolometric", "planck"}
if str(cfg.emission_model).strip().lower() not in _valid_emission_models:
    raise ValueError(f"cfg.emission_model must be one of {_valid_emission_models}, got {cfg.emission_model!r}")

_valid_map_projection_modes = {"shape_only", "shape_plus_amplitude"}
if str(cfg.map_projection_mode).strip().lower() not in _valid_map_projection_modes:
    raise ValueError(
        f"cfg.map_projection_mode must be one of {_valid_map_projection_modes}, got {cfg.map_projection_mode!r}"
    )

_valid_ns_inner_kernels = {"slice", "mala", "hmc"}
if str(cfg.ns_inner_kernel).strip().lower() not in _valid_ns_inner_kernels:
    raise ValueError(f"cfg.ns_inner_kernel must be one of {_valid_ns_inner_kernels}, got {cfg.ns_inner_kernel!r}")

if cfg.model_days <= 0:
    raise ValueError("cfg.model_days must be > 0")
if cfg.dt_seconds <= 0:
    raise ValueError("cfg.dt_seconds must be > 0")
if cfg.starttime_index < 2:
    raise ValueError("cfg.starttime_index must be >= 2 for leapfrog startup")
if cfg.num_chains <= 0 or cfg.num_samples <= 0:
    raise ValueError("cfg.num_chains and cfg.num_samples must be > 0")
if cfg.ns_max_steps <= 0:
    raise ValueError("cfg.ns_max_steps must be > 0")
if cfg.ns_log_every <= 0:
    raise ValueError("cfg.ns_log_every must be > 0")
if cfg.ns_eta_window <= 0:
    raise ValueError("cfg.ns_eta_window must be > 0")
if cfg.ns_num_live < 4:
    raise ValueError("cfg.ns_num_live must be >= 4")
if cfg.ns_hmc_num_integration_steps <= 0:
    raise ValueError("cfg.ns_hmc_num_integration_steps must be > 0")
if cfg.ns_mala_step_size <= 0.0 or cfg.ns_hmc_step_size <= 0.0:
    raise ValueError("cfg.ns_mala_step_size and cfg.ns_hmc_step_size must be > 0")


# =============================================================================
# Environment + logging
# =============================================================================


# my_swamp reads this at import time (per README).
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1" if cfg.use_x64 else "0")
if not cfg.xla_preallocate:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

cfg.out_dir.mkdir(parents=True, exist_ok=True)
log_path = cfg.out_dir / "run.log"

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w" if cfg.overwrite else "a"),
    ],
    force=True,
)
logger = logging.getLogger("swamp_run")


# =============================================================================
# Imports (after env vars)
# =============================================================================


import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", cfg.use_x64)

# my_swamp
import my_swamp.model as swamp_model
from my_swamp.model import RunFlags, build_static

# starry via jaxoplanet
from jaxoplanet.orbits.keplerian import Body, Central
from jaxoplanet.starry.light_curves import light_curve as starry_light_curve
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

# blackjax
try:
    import blackjax  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "BlackJAX is required for this script. Install with `pip install blackjax` "
        "(and ensure your JAX install matches your accelerator, if any)."
    ) from e

logger.info(f"JAX backend: {jax.default_backend()}")
logger.info(f"JAX devices: {jax.devices()}")
logger.info(f"BlackJAX version: {getattr(blackjax, '__version__', 'unknown')}")


# =============================================================================
# Utility helpers
# =============================================================================


def float_dtype() -> Any:
    """Local dtype helper (do NOT depend on my_swamp.dtypes, which may not exist)."""
    return jnp.float64 if cfg.use_x64 else jnp.float32


def np_float_dtype() -> Any:
    """NumPy dtype matching float_dtype()."""
    return np.float64 if cfg.use_x64 else np.float32


def tau_hours_to_seconds(x_hours: Any) -> Any:
    return 3600.0 * x_hours


def orbital_period_days_from_omega(omega_rad_s: Any) -> Any:
    """Synchronous period implied by omega (can be JAX scalar)."""
    return (2.0 * jnp.pi / omega_rad_s) / 86400.0


def compute_n_steps(model_days: float, dt_seconds: float) -> int:
    n = int(np.round(model_days * 86400.0 / dt_seconds))
    return max(n, 1)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def call_with_filtered_kwargs(func, kwargs: Dict[str, Any], *, name: Optional[str] = None):
    """Call func(**kwargs), filtering out unexpected kwargs using inspect.signature.

    This is critical when supporting multiple my_swamp / jaxoplanet versions whose call
    signatures may differ slightly.
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
# Parameter registry (requested: inference over multiple params via bool toggles)
# =============================================================================


@dataclass(frozen=True)
class ParamSpec:
    """A single inferred parameter specification.

    We sample in an unconstrained u-space using a sigmoid transform:
      z = sigmoid(u) in (0, 1)

    The interpolation coordinate depends on prior_type:

      - prior_type="uniform":
          x = lo + (hi - lo) * z
          theta = x

      - prior_type="log10_uniform":
          x = log10(lo) + (log10(hi) - log10(lo)) * z
          theta = 10**x    (so x is Uniform in log10(theta), i.e. log-uniform in theta)

    In both cases, the prior in x is Uniform(lo, hi), independent across parameters.
    Under x = lo + width * sigmoid(u), the induced density in u is:
      p(u) ∝ sigmoid(u) * (1 - sigmoid(u))
    so:
      log p(u) = log_sigmoid(u) + log_sigmoid(-u) + const
    """

    name: str
    label: str
    prior_type: str  # {"uniform", "log10_uniform"}
    lo: float
    hi: float
    truth: float


def _specs_from_config(cfg: Config) -> List[ParamSpec]:
    """Build the active parameter list based on cfg.infer_* toggles."""
    specs: List[ParamSpec] = []

    def add(name: str, label: str, prior_type: str, lo: float, hi: float, truth: float) -> None:
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"Invalid prior bounds for {name}: lo={lo}, hi={hi}")
        if prior_type not in {"uniform", "log10_uniform"}:
            raise ValueError(f"Invalid prior_type for {name}: {prior_type!r}")
        if prior_type == "log10_uniform" and (lo <= 0.0 or hi <= 0.0):
            raise ValueError(
                f"log10_uniform prior requires strictly positive bounds for {name}: lo={lo}, hi={hi}. "
                "Either choose prior_type=\"uniform\" or set positive min/max."
            )
        specs.append(ParamSpec(name=name, label=label, prior_type=prior_type, lo=lo, hi=hi, truth=truth))

    # --- Taus (requested uniform priors in linear hours; independently configurable)
    if cfg.infer_tau_rad:
        add(
            "tau_rad_hours",
            "tau_rad [h]",
            "uniform",
            float(cfg.prior_tau_rad_hours_min),
            float(cfg.prior_tau_rad_hours_max),
            float(cfg.taurad_true_hours),
        )
    if cfg.infer_tau_drag:
        add(
            "tau_drag_hours",
            "tau_drag [h]",
            "uniform",
            float(cfg.prior_tau_drag_hours_min),
            float(cfg.prior_tau_drag_hours_max),
            float(cfg.taudrag_true_hours),
        )

    # --- System / starry params
    if cfg.infer_planet_radius:
        add(
            "planet_radius_rjup",
            "R_p [Rjup]",
            str(cfg.prior_type_planet_radius).strip().lower(),
            float(cfg.prior_planet_radius_rjup_min),
            float(cfg.prior_planet_radius_rjup_max),
            float(cfg.planet_radius_rjup),
        )
    if cfg.infer_planet_fpfs:
        add(
            "planet_fpfs",
            "F_p/F_s",
            str(cfg.prior_type_planet_fpfs).strip().lower(),
            float(cfg.prior_planet_fpfs_min),
            float(cfg.prior_planet_fpfs_max),
            float(cfg.planet_fpfs),
        )

    # --- SWAMP physical params
    if cfg.infer_Phibar:
        add(
            "Phibar",
            "Phibar",
            str(cfg.prior_type_Phibar).strip().lower(),
            float(cfg.prior_Phibar_min),
            float(cfg.prior_Phibar_max),
            float(cfg.Phibar),
        )
    if cfg.infer_DPhieq:
        add(
            "DPhieq",
            "DPhieq",
            str(cfg.prior_type_DPhieq).strip().lower(),
            float(cfg.prior_DPhieq_min),
            float(cfg.prior_DPhieq_max),
            float(cfg.DPhieq),
        )
    if cfg.infer_K6:
        add(
            "K6",
            "K6",
            str(cfg.prior_type_K6).strip().lower(),
            float(cfg.prior_K6_min),
            float(cfg.prior_K6_max),
            float(cfg.K6),
        )
    if cfg.infer_K6Phi:
        # If cfg.K6Phi is None (disabled) but infer_K6Phi=True, treat truth as 0.0 for synthetic data.
        truth_k6phi = 0.0 if cfg.K6Phi is None else float(cfg.K6Phi)
        add(
            "K6Phi",
            "K6Phi",
            str(cfg.prior_type_K6Phi).strip().lower(),
            float(cfg.prior_K6Phi_min),
            float(cfg.prior_K6Phi_max),
            truth_k6phi,
        )

    if cfg.infer_omega:
        add(
            "omega_rad_s",
            "omega [rad/s]",
            str(cfg.prior_type_omega).strip().lower(),
            float(cfg.prior_omega_min),
            float(cfg.prior_omega_max),
            float(cfg.omega_rad_s),
        )
    if cfg.infer_a_planet:
        add(
            "a_planet_m",
            "a [m]",
            str(cfg.prior_type_a_planet).strip().lower(),
            float(cfg.prior_a_planet_min),
            float(cfg.prior_a_planet_max),
            float(cfg.a_planet_m),
        )
    if cfg.infer_g:
        add(
            "g_m_s2",
            "g [m/s^2]",
            str(cfg.prior_type_g).strip().lower(),
            float(cfg.prior_g_min),
            float(cfg.prior_g_max),
            float(cfg.g_m_s2),
        )

    if len(specs) == 0:
        raise ValueError(
            "No parameters enabled for inference. Set at least one cfg.infer_* = True, "
            "or disable cfg.run_inference."
        )
    return specs


param_specs: List[ParamSpec] = _specs_from_config(cfg)
param_names: List[str] = [s.name for s in param_specs]
param_labels: List[str] = [s.label for s in param_specs]
param_prior_types: List[str] = [s.prior_type for s in param_specs]
param_prior_lo: np.ndarray = np.array([s.lo for s in param_specs], dtype=np.float64)
param_prior_hi: np.ndarray = np.array([s.hi for s in param_specs], dtype=np.float64)
param_truth: np.ndarray = np.array([s.truth for s in param_specs], dtype=np.float64)
n_dim = len(param_specs)

logger.info(f"Inferred parameters (n_dim={n_dim}): {param_names}")

# Save a richer config.json for plot.py (includes inferred-parameter metadata).
cfg_dict: Dict[str, Any] = asdict(cfg)
cfg_dict.update(
    dict(
        inferred_param_names=param_names,
        inferred_param_labels=param_labels,
        inferred_param_prior_types=param_prior_types,
        inferred_param_prior_lo=param_prior_lo.tolist(),
        inferred_param_prior_hi=param_prior_hi.tolist(),
        inferred_param_truth=param_truth.tolist(),
    )
)
(cfg.out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
logger.info(f"Wrote config to: {cfg.out_dir / 'config.json'}")


# =============================================================================
# Build SWAMP static + flags ONCE for grid/projection and for the fast path
# =============================================================================


def _build_static_from_values(
    *,
    taurad_s: Any,
    taudrag_s: Any,
    Phibar: Any,
    DPhieq: Any,
    K6: Any,
    K6Phi: Any,
    omega: Any,
    a: Any,
    g: Any,
) -> Any:
    """Build my_swamp static object.

    This is called both:
      - once outside JIT (with Python floats) to define grid/projection, and
      - potentially inside JIT (with JAX scalars) if you infer parameters that
        require rebuilding static for internal consistency.

    IMPORTANT: We avoid casting inputs to Python floats here, because doing so would
    break JAX tracing if these are tracers.
    """
    static_kwargs: Dict[str, Any] = dict(
        M=int(cfg.M),
        dt=jnp.asarray(cfg.dt_seconds, dtype=float_dtype()),
        a=jnp.asarray(a, dtype=float_dtype()),
        omega=jnp.asarray(omega, dtype=float_dtype()),
        g=jnp.asarray(g, dtype=float_dtype()),
        Phibar=jnp.asarray(Phibar, dtype=float_dtype()),
        taurad=jnp.asarray(taurad_s, dtype=float_dtype()),
        taudrag=jnp.asarray(taudrag_s, dtype=float_dtype()),
        DPhieq=jnp.asarray(DPhieq, dtype=float_dtype()),
        K6=jnp.asarray(K6, dtype=float_dtype()),
        # K6Phi=None disables diffusion in some my_swamp builds; we pass 0.0 if None.
        K6Phi=(None if K6Phi is None else jnp.asarray(K6Phi, dtype=float_dtype())),
        test=None,
    )
    return call_with_filtered_kwargs(build_static, static_kwargs, name="build_static")


# Baseline / truth values used to build the projector grid.
static_base = _build_static_from_values(
    taurad_s=tau_hours_to_seconds(cfg.taurad_true_hours),
    taudrag_s=tau_hours_to_seconds(cfg.taudrag_true_hours),
    Phibar=cfg.Phibar,
    DPhieq=cfg.DPhieq,
    K6=cfg.K6,
    K6Phi=cfg.K6Phi,
    omega=cfg.omega_rad_s,
    a=cfg.a_planet_m,
    g=cfg.g_m_s2,
)

flags = call_with_filtered_kwargs(
    RunFlags,
    dict(
        forcflag=cfg.forcflag,
        diffflag=cfg.diffflag,
        expflag=cfg.expflag,
        modalflag=cfg.modalflag,
        diagnostics=cfg.diagnostics,
        alpha=float(cfg.alpha),
        blowup_rms=float(cfg.blowup_rms),
    ),
    name="RunFlags",
)

I = int(getattr(static_base, "I", -1))
J = int(getattr(static_base, "J", -1))
logger.info(f"SWAMP grid: I={I}, J={J}, M={getattr(static_base,'M','?')}, N={getattr(static_base,'N','?')}")

n_steps = compute_n_steps(cfg.model_days, cfg.dt_seconds)
logger.info(f"SWAMP integration length: model_days={cfg.model_days} -> n_steps={n_steps} (dt={cfg.dt_seconds}s)")

t_seq = jnp.arange(cfg.starttime_index, cfg.starttime_index + n_steps, dtype=jnp.int32)


# =============================================================================
# Initial conditions helper
# =============================================================================


def init_rest_state(static: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Rest-like initial conditions consistent with SWAMPE variable conventions.

    - Phi in the model is a *perturbation* geopotential, so equilibrium perturbation is Phieq - Phibar.
    - U,V = 0; delta = 0.
    - eta: use solid-body absolute vorticity ~ 2*omega*sin(lat) = 2*omega*mu.
    """
    dtype = float_dtype()
    Jloc = int(getattr(static, "J"))
    Iloc = int(getattr(static, "I"))

    mus = getattr(static, "mus", None)
    omega = getattr(static, "omega", 0.0)

    if mus is None:
        logger.warning("static.mus not found; using eta0=0.")
        eta0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
    else:
        mu = jnp.asarray(mus, dtype=dtype)  # (J,)
        omega = jnp.asarray(omega, dtype=dtype)
        eta1d = 2.0 * omega * mu            # (J,)
        eta0 = eta1d[:, None] * jnp.ones((Jloc, Iloc), dtype=dtype)

    delta0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
    U0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
    V0 = jnp.zeros((Jloc, Iloc), dtype=dtype)

    Phieq = jnp.asarray(getattr(static, "Phieq", 0.0), dtype=dtype)       # total equilibrium
    Phibar = jnp.asarray(getattr(static, "Phibar", 0.0), dtype=dtype)     # mean
    Phi0 = Phieq - Phibar                                                 # perturbation equilibrium

    return eta0, delta0, U0, V0, Phi0



# State initialization: requires a my_swamp build that exposes this initializer.
init_fn = getattr(swamp_model, "_init_state_from_fields", None) or getattr(swamp_model, "init_state_from_fields", None)
if init_fn is None:
    raise RuntimeError(
        "Could not find my_swamp.model._init_state_from_fields or init_state_from_fields. "
        "This pipeline requires initializing a State from fields without allocating history."
    )


def build_state0(static: Any) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
    """Build (state0, U0, V0) for a given static object."""
    eta0, delta0, U0, V0, Phi0 = init_rest_state(static)
    state0 = call_with_filtered_kwargs(
        init_fn,
        dict(
            static=static,
            flags=flags,
            test=None,
            eta0=eta0,
            delta0=delta0,
            Phi0=Phi0,
            U0=U0,
            V0=V0,
        ),
        name=init_fn.__name__,
    )
    return state0, U0, V0


# Precompute ICs for the "fast path" (only taurad/taudrag inferred).
state0_base, U0_base, V0_base = build_state0(static_base)


# =============================================================================
# Phi -> temperature -> intensity (toy emission layer)
# =============================================================================

def _all_finite(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(jnp.isfinite(x))

def phi_to_temperature(phi: jnp.ndarray) -> jnp.ndarray:
    """Toy Phi -> temperature mapping with finiteness protection.

    If phi contains any non-finite values, return all-NaN to mark the forward model invalid.
    """
    dtype = float_dtype()
    phi = jnp.asarray(phi, dtype=dtype)

    finite_phi = _all_finite(phi)

    def _ok(_: None) -> jnp.ndarray:
        T = jnp.asarray(cfg.T_ref, dtype=dtype) + phi / jnp.asarray(cfg.phi_to_T_scale, dtype=dtype)
        T = jnp.maximum(T, jnp.asarray(cfg.Tmin_K, dtype=dtype))
        return T

    def _bad(_: None) -> jnp.ndarray:
        return jnp.full_like(phi, jnp.asarray(jnp.nan, dtype=dtype))

    return jax.lax.cond(finite_phi, _ok, _bad, operand=None)


def planck_intensity_relative_lambda(T: jnp.ndarray, wavelength_m: float) -> jnp.ndarray:
    """Relative Planck spectral radiance factor B_λ(T) (shape only), guarded.

    If T has any non-finite values, return all-NaN (invalid).
    """
    dtype = float_dtype()
    T = jnp.asarray(T, dtype=dtype)

    finite_T = _all_finite(T)

    def _ok(_: None) -> jnp.ndarray:
        lam = jnp.asarray(wavelength_m, dtype=dtype)

        # Physical constants (exact in SI by definition; cast to dtype)
        h = jnp.asarray(6.62607015e-34, dtype=dtype)  # [J s]
        c = jnp.asarray(299792458.0, dtype=dtype)     # [m/s]
        kB = jnp.asarray(1.380649e-23, dtype=dtype)   # [J/K]

        x = (h * c) / (lam * kB * T)  # dimensionless exponent
        x = jnp.clip(
            x,
            a_min=jnp.asarray(0.0, dtype=dtype),
            a_max=jnp.asarray(cfg.planck_x_clip, dtype=dtype),
        )

        tiny = jnp.asarray(1.0e-30, dtype=dtype)
        return jnp.asarray(1.0, dtype=dtype) / (jnp.expm1(x) + tiny)

    def _bad(_: None) -> jnp.ndarray:
        return jnp.full_like(T, jnp.asarray(jnp.nan, dtype=dtype))

    return jax.lax.cond(finite_T, _ok, _bad, operand=None)

def temperature_to_intensity(T: jnp.ndarray) -> jnp.ndarray:
    """Temperature -> intensity mapping, guarded."""
    mode = str(getattr(cfg, "emission_model", "bolometric")).strip().lower()
    if mode == "bolometric":
        # If T has NaNs/Infs, this propagates; phi_to_temperature already gates.
        return jnp.asarray(T, dtype=float_dtype()) ** 4
    if mode == "planck":
        return planck_intensity_relative_lambda(jnp.asarray(T, dtype=float_dtype()), float(cfg.planck_wavelength_m))
    raise ValueError(
        f"Unknown cfg.emission_model={cfg.emission_model!r}. Valid options are {'bolometric','planck'}."
    )


# =============================================================================
# Precompute SWAMP pixel grid + weights (once)
# =============================================================================


lambdas = getattr(static_base, "lambdas", None)
mus = getattr(static_base, "mus", None)
w_lat = getattr(static_base, "w", None)

if lambdas is None or mus is None:
    raise RuntimeError("static_base.lambdas and static_base.mus are required to build the starry projector.")




lon = jnp.asarray(lambdas, dtype=float_dtype())  # (I,)

mu = jnp.asarray(mus, dtype=float_dtype())  # (J,)
# Critical: protect arcsin from slight float overshoot outside [-1, 1]
mu = jnp.clip(mu, jnp.asarray(-1.0, dtype=float_dtype()), jnp.asarray(1.0, dtype=float_dtype()))
lat = jnp.arcsin(mu)  # (J,)

lon2d = jnp.broadcast_to(lon[None, :], (lat.shape[0], lon.shape[0]))
lat2d = jnp.broadcast_to(lat[:, None], (lat.shape[0], lon.shape[0]))
lon_flat = lon2d.reshape(-1)
lat_flat = lat2d.reshape(-1)

if w_lat is None:
    logger.warning("static_base.w not found; using uniform weights for LSQ projector.")
    w_pix = jnp.ones_like(lat_flat)
else:
    w_lat = jnp.asarray(w_lat, dtype=float_dtype())  # (J,)
    # Basic weight sanity: finite and non-negative (sqrt happens next)
    if not np.all(np.isfinite(np.asarray(w_lat))):
        raise RuntimeError("static_base.w contains non-finite values; cannot build projector weights.")
    if np.any(np.asarray(w_lat) < 0.0):
        raise RuntimeError("static_base.w contains negative values; cannot take sqrt for LSQ weights.")
    w_pix = jnp.repeat(w_lat, lon.shape[0])  # (J*I,)

w_sqrt = jnp.sqrt(w_pix)







# =============================================================================
# Build starry design matrix + LSQ projector (once)
# =============================================================================


def build_lm_list(ydeg: int) -> List[Tuple[int, int]]:
    return [(ell, m) for ell in range(ydeg + 1) for m in range(-ell, ell + 1)]


lm_list = build_lm_list(cfg.ydeg)
n_coeff = (cfg.ydeg + 1) ** 2
n_pix = int(lat_flat.shape[0])

logger.info(f"Building starry design matrix: n_pix={n_pix}, n_coeff={n_coeff} (ydeg={cfg.ydeg})")


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


def _intensity_from_yvec(y_vec: jnp.ndarray) -> jnp.ndarray:
    ylm = ylm_from_dense(y_vec, lm_list)
    surf = Surface(
        y=ylm,
        u=(),  # no limb darkening for thermal emission
        inc=jnp.asarray(cfg.map_inc_rad, dtype=float_dtype()),
        obl=jnp.asarray(cfg.map_obl_rad, dtype=float_dtype()),
        amplitude=jnp.asarray(1.0, dtype=float_dtype()),
        normalize=False,
    )
    return surface_intensity(surf, lat_flat, lon_flat)


_intensity_from_yvec_jit = jax.jit(_intensity_from_yvec)

eye = jnp.eye(n_coeff, dtype=float_dtype())
t0_B = time.time()
B = jax.vmap(_intensity_from_yvec_jit)(eye).T  # (n_pix, n_coeff)
_ = B.block_until_ready()

B_np = np.asarray(B)
if not np.isfinite(B_np).all():
    raise RuntimeError("Design matrix B contains NaNs/Infs. Most likely lat/ lon grid contains NaNs.")


logger.info(f"Design matrix built in {time.time() - t0_B:.2f} s; shape={tuple(B.shape)}")

# Weighted ridge LSQ projector: y = (Bᵀ W B + λI)^(-1) Bᵀ W
Bw = w_sqrt[:, None] * B
ridge = jnp.asarray(cfg.projector_ridge, dtype=float_dtype())
gram = Bw.T @ Bw + ridge * jnp.eye(n_coeff, dtype=float_dtype())

t0_proj = time.time()
projector = jnp.linalg.solve(gram, Bw.T)  # (n_coeff, n_pix)
_ = projector.block_until_ready()

P_np = np.asarray(projector)
if not np.isfinite(P_np).all():
    raise RuntimeError("Projector contains NaNs/Infs. Check weights, ridge term, and design matrix conditioning.")


logger.info(f"Projector built in {time.time() - t0_proj:.2f} s")


def intensity_map_to_y_dense_and_monopole(I_map: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert SWAMP intensity map -> starry coefficients and return map monopole proxy.

    Returns
    -------
    y_shape : jnp.ndarray
        Dense Ylm coefficients normalized by y0 (shape information only).
    y0_raw : jnp.ndarray
        Raw monopole coefficient before shape normalization (amplitude carrier).
    """
    dtype = float_dtype()
    I_flat = jnp.asarray(I_map, dtype=dtype).reshape(-1)

    finite_mask = jnp.isfinite(I_flat)
    all_finite_in = jnp.all(finite_mask)
    I_safe = jnp.where(finite_mask, I_flat, jnp.asarray(0.0, dtype=dtype))

    # Historical behavior: remove global weighted mean before projection.
    if str(cfg.map_projection_mode).strip().lower() == "shape_only":
        w_eff = jnp.asarray(w_pix, dtype=dtype) * finite_mask
        w_sum = jnp.sum(w_eff)

        def _mean_ok(_: None) -> jnp.ndarray:
            return jnp.sum(w_eff * I_safe) / w_sum

        def _mean_bad(_: None) -> jnp.ndarray:
            return jnp.asarray(jnp.nan, dtype=dtype)

        I_mean = jax.lax.cond(w_sum > jnp.asarray(0.0, dtype=dtype), _mean_ok, _mean_bad, operand=None)
        eps = jnp.asarray(1.0e-30, dtype=dtype)
        mean_ok = jnp.isfinite(I_mean) & (jnp.abs(I_mean) > eps)
        I_for_fit = jnp.where(mean_ok, I_safe / I_mean, jnp.asarray(jnp.nan, dtype=dtype))
    else:
        eps = jnp.asarray(1.0e-30, dtype=dtype)
        mean_ok = jnp.asarray(True)
        I_for_fit = I_safe

    rhs = jnp.asarray(w_sqrt, dtype=dtype) * I_for_fit
    y_raw = jnp.asarray(projector, dtype=dtype) @ rhs

    y0_raw = y_raw[0]
    y0_ok = jnp.isfinite(y0_raw) & (jnp.abs(y0_raw) > eps)
    y_shape = jnp.where(y0_ok, y_raw / y0_raw, jnp.full_like(y_raw, jnp.asarray(jnp.nan, dtype=dtype)))

    out_ok = all_finite_in & mean_ok & y0_ok & _all_finite(y_shape)

    y_shape = jax.lax.cond(
        out_ok,
        lambda _: y_shape,
        lambda _: jnp.full_like(y_shape, jnp.asarray(jnp.nan, dtype=dtype)),
        operand=None,
    )
    y0_raw = jax.lax.cond(
        out_ok,
        lambda _: y0_raw,
        lambda _: jnp.asarray(jnp.nan, dtype=dtype),
        operand=None,
    )
    return y_shape, y0_raw

# =============================================================================
# Observation times (fixed)
# =============================================================================


# NOTE: orbital period used to place observation times is fixed by cfg.omega_rad_s unless overridden.
orbital_period_days_base = (
    float(cfg.orbital_period_override_days)
    if cfg.orbital_period_override_days is not None
    else float((2.0 * math.pi / cfg.omega_rad_s) / 86400.0)
)
logger.info(f"Baseline orbital/rotation period (days): {orbital_period_days_base:.6f}")

times_days = np.linspace(
    cfg.time_transit_days,
    cfg.time_transit_days + cfg.n_orbits_observed * orbital_period_days_base,
    cfg.n_times,
    endpoint=False,
).astype(np_float_dtype())
times_days_jax = jnp.asarray(times_days, dtype=float_dtype())


# =============================================================================
# SWAMP forward model (terminal Phi)
# =============================================================================


# Determine whether we can use the fast static update path.
# If any parameter that impacts static/ICs is inferred, we rebuild static/state0 each call.
_fast_path_ok = (not cfg.force_rebuild_static) and not (
    cfg.infer_Phibar
    or cfg.infer_DPhieq
    or cfg.infer_K6
    or cfg.infer_K6Phi
    or cfg.infer_omega
    or cfg.infer_a_planet
    or cfg.infer_g
)


def swamp_terminal_phi(
    taurad_s: jnp.ndarray,
    taudrag_s: jnp.ndarray,
    *,
    Phibar: jnp.ndarray,
    DPhieq: jnp.ndarray,
    K6: jnp.ndarray,
    K6Phi: Optional[jnp.ndarray],
    omega: jnp.ndarray,
    a: jnp.ndarray,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """Run SWAMP for cfg.model_days and return terminal Phi map (J,I).

    Two modes:
      - Fast path: only taurad/taudrag vary; reuse static_base and state0_base.
      - General path: rebuild static + state0 each call for internal consistency.
    """
    if _fast_path_ok:
        # Only taurad/taudrag vary; static/ICs don't depend on those.
        try:
            from dataclasses import replace as dc_replace

            static = dc_replace(static_base, taurad=taurad_s, taudrag=taudrag_s)
            state0 = state0_base
            U0 = U0_base
            V0 = V0_base
        except Exception as e:  # pragma: no cover
            # If dataclasses.replace fails, we can still rebuild static/state0.
            logger.warning(f"Fast-path static update failed; rebuilding static. Error: {e}")
            static = _build_static_from_values(
                taurad_s=taurad_s,
                taudrag_s=taudrag_s,
                Phibar=Phibar,
                DPhieq=DPhieq,
                K6=K6,
                K6Phi=(None if K6Phi is None else K6Phi),
                omega=omega,
                a=a,
                g=g,
            )
            state0, U0, V0 = build_state0(static)
    else:
        static = _build_static_from_values(
            taurad_s=taurad_s,
            taudrag_s=taudrag_s,
            Phibar=Phibar,
            DPhieq=DPhieq,
            K6=K6,
            K6Phi=(None if K6Phi is None else K6Phi),
            omega=omega,
            a=a,
            g=g,
        )
        state0, U0, V0 = build_state0(static)

    # Prefer a terminal-only scan if available.
    sim_last = getattr(swamp_model, "simulate_scan_last", None) or getattr(swamp_model, "run_model_scan_final", None)
    if sim_last is not None:
        # We support both APIs by filtering kwargs.
        kwargs = dict(
            static=static,
            flags=flags,
            state0=state0,
            t_seq=t_seq,
            test=None,
            Uic=U0,
            Vic=V0,
            remat_step=False,
            # run_model_scan_final-style args (ignored if not present)
            jit_scan=True,
            return_history=False,
        )
        out = call_with_filtered_kwargs(sim_last, kwargs, name=getattr(sim_last, "__name__", "simulate_scan_last"))
        # simulate_scan_last returns last_state; run_model_scan_final returns dict with 'last_state'
        last_state = out
        if isinstance(out, dict) and "last_state" in out:
            last_state = out["last_state"]
        return getattr(last_state, "Phi_curr")

    # Fallback: fori_loop stepping
    step_fn = getattr(swamp_model, "_step_once_state_only", None)
    if step_fn is None:
        raise RuntimeError(
            "Could not find simulate_scan_last/run_model_scan_final nor _step_once_state_only in my_swamp.model; "
            "cannot run SWAMP forward."
        )

    def body(i: int, st: Any) -> Any:
        t = t_seq[i]
        return step_fn(st, t, static, flags, None, U0, V0)

    state_f = jax.lax.fori_loop(0, int(t_seq.shape[0]), body, state0)
    return getattr(state_f, "Phi_curr")


# =============================================================================
# Full forward model: theta -> phase curve
# =============================================================================


# Conversion Rjup -> Rsun (approx; acceptable for this toy demo)
RJUP_TO_RSUN = 0.10045

central = Central(radius=cfg.star_radius_rsun, mass=cfg.star_mass_msun)

# Planet-only emission: star amplitude 0, but star still exists for occultation.
star_surface = Surface(amplitude=jnp.asarray(0.0, dtype=float_dtype()), normalize=False)


def _theta_vector_to_model_kwargs(theta: jnp.ndarray) -> Dict[str, Any]:
    """Convert the inferred theta vector into a dict of model parameters.

    Parameters not being inferred are taken from cfg. This function is pure and JAX-friendly.
    """
    # Start with defaults (as JAX scalars for consistency under JIT).
    params: Dict[str, Any] = dict(
        tau_rad_hours=jnp.asarray(cfg.taurad_true_hours, dtype=float_dtype()),
        tau_drag_hours=jnp.asarray(cfg.taudrag_true_hours, dtype=float_dtype()),
        planet_radius_rjup=jnp.asarray(cfg.planet_radius_rjup, dtype=float_dtype()),
        planet_fpfs=jnp.asarray(cfg.planet_fpfs, dtype=float_dtype()),
        Phibar=jnp.asarray(cfg.Phibar, dtype=float_dtype()),
        DPhieq=jnp.asarray(cfg.DPhieq, dtype=float_dtype()),
        K6=jnp.asarray(cfg.K6, dtype=float_dtype()),
        K6Phi=(None if cfg.K6Phi is None else jnp.asarray(cfg.K6Phi, dtype=float_dtype())),
        omega_rad_s=jnp.asarray(cfg.omega_rad_s, dtype=float_dtype()),
        a_planet_m=jnp.asarray(cfg.a_planet_m, dtype=float_dtype()),
        g_m_s2=jnp.asarray(cfg.g_m_s2, dtype=float_dtype()),
    )

    # Override inferred entries.
    for i, spec in enumerate(param_specs):
        params[spec.name] = theta[i]

    return params


def _reference_map_monopole_from_truth() -> jnp.ndarray:
    """Reference y0 monopole used to keep `planet_fpfs` on a stable scale."""
    dtype = float_dtype()
    theta_ref = jnp.asarray(param_truth, dtype=dtype)
    p = _theta_vector_to_model_kwargs(theta_ref)

    k6phi_val = p["K6Phi"]
    if k6phi_val is not None:
        k6phi_val = jnp.asarray(k6phi_val, dtype=dtype)

    phi = swamp_terminal_phi(
        jnp.asarray(tau_hours_to_seconds(p["tau_rad_hours"]), dtype=dtype),
        jnp.asarray(tau_hours_to_seconds(p["tau_drag_hours"]), dtype=dtype),
        Phibar=jnp.asarray(p["Phibar"], dtype=dtype),
        DPhieq=jnp.asarray(p["DPhieq"], dtype=dtype),
        K6=jnp.asarray(p["K6"], dtype=dtype),
        K6Phi=k6phi_val,
        omega=jnp.asarray(p["omega_rad_s"], dtype=dtype),
        a=jnp.asarray(p["a_planet_m"], dtype=dtype),
        g=jnp.asarray(p["g_m_s2"], dtype=dtype),
    )
    T = phi_to_temperature(phi)
    I_map = temperature_to_intensity(T)
    _, y0_ref = intensity_map_to_y_dense_and_monopole(I_map)
    return jnp.asarray(y0_ref, dtype=dtype)


reference_map_monopole = _reference_map_monopole_from_truth()
_ref_map_monopole_f = float(np.asarray(reference_map_monopole))
if (not math.isfinite(_ref_map_monopole_f)) or abs(_ref_map_monopole_f) < 1.0e-30:
    logger.warning("Reference map monopole is invalid (non-finite or ~0); falling back to 1.0.")
    reference_map_monopole = jnp.asarray(1.0, dtype=float_dtype())
    _ref_map_monopole_f = 1.0
logger.info(
    "Map projection mode=%s, reference monopole=%.6g",
    str(cfg.map_projection_mode).strip().lower(),
    _ref_map_monopole_f,
)


def phase_curve_model(theta: jnp.ndarray) -> jnp.ndarray:
    """Complete forward model returning planet flux vs time (shape: n_times).

    Returns all-NaN if SWAMP terminal Phi or subsequent map projection becomes non-finite.
    """
    dtype = float_dtype()
    p = _theta_vector_to_model_kwargs(theta)

    taurad_s = jnp.asarray(tau_hours_to_seconds(p["tau_rad_hours"]), dtype=dtype)
    taudrag_s = jnp.asarray(tau_hours_to_seconds(p["tau_drag_hours"]), dtype=dtype)

    k6phi_val = p["K6Phi"]
    if k6phi_val is not None:
        k6phi_val = jnp.asarray(k6phi_val, dtype=dtype)

    phi = swamp_terminal_phi(
        taurad_s,
        taudrag_s,
        Phibar=jnp.asarray(p["Phibar"], dtype=dtype),
        DPhieq=jnp.asarray(p["DPhieq"], dtype=dtype),
        K6=jnp.asarray(p["K6"], dtype=dtype),
        K6Phi=k6phi_val,
        omega=jnp.asarray(p["omega_rad_s"], dtype=dtype),
        a=jnp.asarray(p["a_planet_m"], dtype=dtype),
        g=jnp.asarray(p["g_m_s2"], dtype=dtype),
    )

    def _bad(_: None) -> jnp.ndarray:
        return jnp.full((times_days_jax.shape[0],), jnp.asarray(jnp.nan, dtype=dtype), dtype=dtype)

    def _ok(_: None) -> jnp.ndarray:
        T = phi_to_temperature(phi)
        I_map = temperature_to_intensity(T)

        y_dense, y0_raw = intensity_map_to_y_dense_and_monopole(I_map)
        ylm = ylm_from_dense(y_dense, lm_list)

        if str(cfg.map_projection_mode).strip().lower() == "shape_plus_amplitude":
            amp_factor = y0_raw / jnp.asarray(reference_map_monopole, dtype=dtype)
        else:
            amp_factor = jnp.asarray(1.0, dtype=dtype)

        planet_amp = jnp.asarray(p["planet_fpfs"], dtype=dtype) * amp_factor

        if cfg.orbital_period_override_days is not None:
            orbital_period_days = jnp.asarray(cfg.orbital_period_override_days, dtype=dtype)
        else:
            orbital_period_days = jnp.asarray(orbital_period_days_from_omega(p["omega_rad_s"]), dtype=dtype)

        planet_surface = Surface(
            y=ylm,
            u=(),
            inc=jnp.asarray(cfg.map_inc_rad, dtype=dtype),
            obl=jnp.asarray(cfg.map_obl_rad, dtype=dtype),
            period=orbital_period_days,
            phase=jnp.asarray(cfg.phase_at_transit_rad, dtype=dtype),
            amplitude=planet_amp,
            normalize=False,
        )

        planet = Body(
            radius=jnp.asarray(p["planet_radius_rjup"], dtype=dtype)
            * jnp.asarray(RJUP_TO_RSUN, dtype=dtype),
            period=orbital_period_days,
            time_transit=jnp.asarray(cfg.time_transit_days, dtype=dtype),
            impact_param=jnp.asarray(cfg.impact_param, dtype=dtype),
        )

        system = SurfaceSystem(
            central=central,
            central_surface=star_surface,
            bodies=((planet, planet_surface),),
        )

        lc = starry_light_curve(system)(times_days_jax)  # (n_times, 2): [star, planet]
        return lc[:, 1]

    # Gate on Phi finiteness first; later stages also inject NaNs if invalid.
    return jax.lax.cond(_all_finite(phi), _ok, _bad, operand=None)



phase_curve_model_jit = jax.jit(phase_curve_model)


# =============================================================================
# u-space parameterization (NSS lives in unconstrained R^D)
# =============================================================================


prior_lo = jnp.asarray(param_prior_lo, dtype=float_dtype())
prior_hi = jnp.asarray(param_prior_hi, dtype=float_dtype())
prior_width = prior_hi - prior_lo


def theta_from_u(u: jnp.ndarray) -> jnp.ndarray:
    """Map unconstrained u -> physical theta vector.

    For prior_type="uniform":
        theta ~ Uniform(lo, hi)

    For prior_type="log10_uniform":
        log10(theta) ~ Uniform(log10(lo), log10(hi))   (i.e., log-uniform in theta)

    In both cases we use:
        z = sigmoid(u) in (0, 1)
    and linearly interpolate in the appropriate coordinate.
    """
    u = jnp.asarray(u, dtype=float_dtype())
    z = jax.nn.sigmoid(u)  # (0,1)

    out = []
    for i, spec in enumerate(param_specs):
        lo = prior_lo[i]
        hi = prior_hi[i]
        zi = z[i]
        if spec.prior_type == "uniform":
            out.append(lo + (hi - lo) * zi)
        elif spec.prior_type == "log10_uniform":
            # Bounds validated in _specs_from_config (strictly positive).
            lo_log = jnp.log10(lo)
            hi_log = jnp.log10(hi)
            x = lo_log + (hi_log - lo_log) * zi
            out.append(10.0 ** x)
        else:  # pragma: no cover
            raise ValueError(f"Unknown prior_type {spec.prior_type!r} for parameter {spec.name}")
    return jnp.stack(out, axis=0)



def log_prior_u(u: jnp.ndarray) -> jnp.ndarray:
    """Log prior density in u-space (up to an additive constant).

    The Uniform(lo, hi) prior in the chosen coordinate x induces the same p(u) ∝ sigmoid(u)*(1-sigmoid(u)),
    independent of lo/hi, so log p(u) is just the sum of logistic Jacobian terms.
    """
    u = jnp.asarray(u, dtype=float_dtype())
    return jnp.sum(jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))


# =============================================================================
# Observations (synthetic by default)
# =============================================================================


obs_path = cfg.out_dir / "observations.npz"

# Build the truth theta vector in *physical* space for synthetic data generation.
theta_truth = []
for spec in param_specs:
    # Use spec.truth (in physical units) for any inferred parameter.
    theta_truth.append(jnp.asarray(spec.truth, dtype=float_dtype()))
theta_truth = jnp.stack(theta_truth, axis=0)

# JIT compile forward model once early (first call can be slow).
logger.info("JIT compiling forward model (first call can be slow)...")
t0_compile = time.time()
_ = phase_curve_model_jit(theta_truth).block_until_ready()
logger.info(f"Forward model compiled in {time.time() - t0_compile:.2f} s")

if cfg.generate_synthetic_data or not obs_path.exists():
    logger.info("Generating synthetic observations from SWAMP+starry truth...")
    flux_true = np.asarray(phase_curve_model_jit(theta_truth)).astype(np_float_dtype())

    rng = np.random.default_rng(cfg.seed)
    noise = rng.normal(0.0, cfg.obs_sigma, size=flux_true.shape).astype(np_float_dtype())
    flux_obs = flux_true + noise

    save_npz(
        obs_path,
        times_days=times_days,
        flux_true=flux_true,
        flux_obs=flux_obs,
        obs_sigma=float(cfg.obs_sigma),
        orbital_period_days=float(orbital_period_days_base),
        inferred_param_names=np.asarray(param_names, dtype="<U64"),
        inferred_param_truth=np.asarray(param_truth, dtype=np.float64),
    )
    logger.info(f"Saved observations to: {obs_path}")
else:
    d = np.load(obs_path)
    times_days = d["times_days"]
    times_days_jax = jnp.asarray(times_days, dtype=float_dtype())
    flux_true = d["flux_true"]
    flux_obs = d["flux_obs"]
    logger.info(f"Loaded observations from: {obs_path}")


finite = np.isfinite(flux_true)
logger.info(f"Truth flux finite: {finite.sum()}/{finite.size}")
if not finite.any():
    raise RuntimeError(
        "Truth flux is all non-finite (NaN/Inf). Forward model is invalid. "
        "Common causes: SWAMP blow-up (Phi non-finite) or projector poisoning (lat=arcsin(mus) NaNs)."
    )

imin = int(np.nanargmin(flux_true))
imax = int(np.nanargmax(flux_true))
logger.info(
    f"Truth flux: min={flux_true[imin]:.3e} at t={times_days[imin]:.5f} d, "
    f"max={flux_true[imax]:.3e} at t={times_days[imax]:.5f} d"
)

flux_obs_jax = jnp.asarray(flux_obs, dtype=float_dtype())
obs_sigma_jax = jnp.asarray(cfg.obs_sigma, dtype=float_dtype())


# =============================================================================
# Posterior + inference (BlackJAX NSS)
# =============================================================================


samples_path = cfg.out_dir / "posterior_samples.npz"
extra_path = cfg.out_dir / "mcmc_extra_fields.npz"


def log_likelihood_u(u: jnp.ndarray) -> jnp.ndarray:
    theta = theta_from_u(u)
    mu = phase_curve_model_jit(theta)

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


# --- Custom gradients (optional) ---------------------------------------------


def _value_and_grad_fwd(fun, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (fun(x), grad fun(x)) using forward-mode JVPs.

    This is memory-stable for expensive scans but costs O(D) JVPs.
    """
    x = jnp.asarray(x)
    n = int(x.shape[0])
    eye = jnp.eye(n, dtype=x.dtype)

    y0, dy0 = jax.jvp(fun, (x,), (eye[0],))

    if n == 1:
        grad = jnp.atleast_1d(dy0)
        return y0, grad

    def jvp_dir(v: jnp.ndarray) -> jnp.ndarray:
        _, dy = jax.jvp(fun, (x,), (v,))
        return dy

    dy_rest = jax.vmap(jvp_dir)(eye[1:])
    grad = jnp.concatenate([jnp.atleast_1d(dy0), dy_rest], axis=0)
    return y0, grad


_use_custom_grads = bool(cfg.use_custom_likelihood_gradients) and (n_dim <= int(cfg.custom_grad_max_dim))
if cfg.use_custom_likelihood_gradients and not _use_custom_grads:
    logger.warning(
        "cfg.use_custom_likelihood_gradients=True but n_dim=%d > custom_grad_max_dim=%d. "
        "Disabling custom forward-mode gradients.",
        n_dim,
        int(cfg.custom_grad_max_dim),
    )

if _use_custom_grads:

    @jax.custom_vjp
    def log_likelihood_u_for_grad(u: jnp.ndarray) -> jnp.ndarray:
        return log_likelihood_u(u)

    def _ll_fwd(u: jnp.ndarray):
        val, grad = _value_and_grad_fwd(log_likelihood_u, u)
        return val, grad

    def _ll_bwd(grad: jnp.ndarray, g: jnp.ndarray):
        return (g * grad,)

    log_likelihood_u_for_grad.defvjp(_ll_fwd, _ll_bwd)

    loglikelihood_for_blackjax = log_likelihood_u_for_grad
    logger.info("Using custom VJP for likelihood gradients (forward-mode JVP-based).")
else:
    loglikelihood_for_blackjax = log_likelihood_u
    logger.info("Using default reverse-mode gradients for likelihood (jax.grad).")


# --- Prior sampling in u-space -----------------------------------------------


def sample_prior_u(rng_key: jax.Array, n_particles: int) -> jax.Array:
    """Sample u such that x = lo + width * sigmoid(u) is Uniform(lo, hi)."""
    eps = jnp.asarray(1e-6, dtype=float_dtype())
    z = jax.random.uniform(rng_key, shape=(n_particles, n_dim), minval=eps, maxval=1.0 - eps)
    return jnp.log(z) - jnp.log1p(-z)


def _extract_acceptance_rate(info: Any) -> jnp.ndarray:
    """Best-effort extraction of an acceptance statistic from a BlackJAX kernel info object."""
    # Common names across kernels/versions
    for attr in ("acceptance_rate", "acceptance_probability", "accept_prob", "prob_accept"):
        if hasattr(info, attr):
            return jnp.asarray(getattr(info, attr), dtype=float_dtype())

    # Some kernels expose a boolean "is_accepted"
    if hasattr(info, "is_accepted"):
        return jnp.asarray(getattr(info, "is_accepted"), dtype=float_dtype())

    # Some wrapper objects nest the kernel info
    if hasattr(info, "update_info"):
        return _extract_acceptance_rate(getattr(info, "update_info"))

    raise AttributeError("Could not extract acceptance statistic from BlackJAX info object.")


def _mala_init_one(u: jnp.ndarray, logdensity_fn):
    """Init wrapper tolerant to BlackJAX signature differences."""
    try:
        return blackjax.mala.init(u, logdensity_fn)
    except TypeError:
        return blackjax.mala.init(u)


def _mala_step_one(rng_key: jax.Array, state, logdensity_fn, step_size: jnp.ndarray):
    """Step wrapper tolerant to BlackJAX signature differences."""
    kernel = blackjax.mala.build_kernel()
    try:
        return kernel(rng_key, state, logdensity_fn, step_size)
    except TypeError:
        return kernel(rng_key, state, logdensity_fn, step_size=step_size)


def _hmc_init_one(u: jnp.ndarray, logdensity_fn):
    """Init wrapper tolerant to BlackJAX signature differences."""
    try:
        return blackjax.hmc.init(u, logdensity_fn)
    except TypeError:
        return blackjax.hmc.init(u)


def _hmc_step_one(
    rng_key: jax.Array,
    state,
    logdensity_fn,
    step_size: jnp.ndarray,
    inverse_mass_matrix: jnp.ndarray,
    num_integration_steps: jnp.ndarray,
):
    """Step wrapper tolerant to BlackJAX signature differences."""
    kernel = blackjax.hmc.build_kernel()
    try:
        return kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
    except TypeError:
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            num_integration_steps=num_integration_steps,
        )


def _build_nss_algorithm(
    *,
    n_delete: int,
    n_inner: int,
):
    """Build NSS algorithm object and return (algo, inner_kernel_used)."""
    inner_kernel_requested = str(cfg.ns_inner_kernel).strip().lower()

    if inner_kernel_requested == "slice":
        return (
            blackjax.nss(
                logprior_fn=log_prior_u,
                loglikelihood_fn=loglikelihood_for_blackjax,
                num_delete=n_delete,
                num_inner_steps=n_inner,
            ),
            "slice",
        )

    try:
        from blackjax.ns.adaptive import build_kernel as ns_adaptive_build_kernel  # type: ignore
        from blackjax.ns.adaptive import init as ns_adaptive_init  # type: ignore
        from blackjax.ns.base import delete_fn as ns_delete_fn  # type: ignore
        from blackjax.ns.base import new_state_and_info as ns_new_state_and_info  # type: ignore
        from blackjax.ns.utils import repeat_kernel as ns_repeat_kernel  # type: ignore
    except Exception as e:
        logger.warning(
            "Gradient-informed NSS requested (ns_inner_kernel=%s) but blackjax.ns.adaptive APIs "
            "are not available (%s). Falling back to slice inner kernel.",
            inner_kernel_requested,
            e,
        )
        return (
            blackjax.nss(
                logprior_fn=log_prior_u,
                loglikelihood_fn=loglikelihood_for_blackjax,
                num_delete=n_delete,
                num_inner_steps=n_inner,
            ),
            "slice_fallback",
        )

    dtype = float_dtype()

    def _init_inner_params():
        if inner_kernel_requested == "mala":
            return {"step_size": jnp.asarray(float(cfg.ns_mala_step_size), dtype=dtype)}
        inv_mass = jnp.ones((n_dim,), dtype=dtype)
        return {
            "step_size": jnp.asarray(float(cfg.ns_hmc_step_size), dtype=dtype),
            "inverse_mass_matrix": inv_mass,
            "num_integration_steps": jnp.asarray(int(cfg.ns_hmc_num_integration_steps), dtype=jnp.int32),
        }

    def _build_constrained_logprior(logprior_fn, loglikelihood_fn, loglikelihood_0):
        neg_inf = jnp.asarray(-jnp.inf, dtype=dtype)

        def _constrained(x: jnp.ndarray) -> jnp.ndarray:
            lp = logprior_fn(x)
            ll = loglikelihood_fn(x)
            ok = jnp.isfinite(lp) & jnp.isfinite(ll) & (ll > loglikelihood_0)
            return jnp.where(ok, lp, neg_inf)

        return _constrained

    def _mala_inner_kernel(rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params):
        step_size = jnp.asarray(params["step_size"], dtype=dtype)
        constrained = _build_constrained_logprior(logprior_fn, loglikelihood_fn, loglikelihood_0)
        m_state = _mala_init_one(state.position, constrained)
        m_state_new, m_info = _mala_step_one(rng_key, m_state, constrained, step_size)

        pos_new = m_state_new.position
        lp_new = logprior_fn(pos_new)
        ll_new = loglikelihood_fn(pos_new)
        valid = jnp.isfinite(lp_new) & jnp.isfinite(ll_new) & (ll_new > loglikelihood_0)

        ns_state_new, ns_info = ns_new_state_and_info(
            position=pos_new,
            logprior=lp_new,
            loglikelihood=ll_new,
            info=m_info,
        )
        ns_state = jax.lax.cond(valid, lambda _: ns_state_new, lambda _: state, operand=None)
        return ns_state, ns_info

    def _hmc_inner_kernel(rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params):
        step_size = jnp.asarray(params["step_size"], dtype=dtype)
        inv_mass = jnp.asarray(params["inverse_mass_matrix"], dtype=dtype)
        n_leap = jnp.asarray(params["num_integration_steps"], dtype=jnp.int32)
        constrained = _build_constrained_logprior(logprior_fn, loglikelihood_fn, loglikelihood_0)
        h_state = _hmc_init_one(state.position, constrained)
        h_state_new, h_info = _hmc_step_one(rng_key, h_state, constrained, step_size, inv_mass, n_leap)

        pos_new = h_state_new.position
        lp_new = logprior_fn(pos_new)
        ll_new = loglikelihood_fn(pos_new)
        valid = jnp.isfinite(lp_new) & jnp.isfinite(ll_new) & (ll_new > loglikelihood_0)

        ns_state_new, ns_info = ns_new_state_and_info(
            position=pos_new,
            logprior=lp_new,
            loglikelihood=ll_new,
            info=h_info,
        )
        ns_state = jax.lax.cond(valid, lambda _: ns_state_new, lambda _: state, operand=None)
        return ns_state, ns_info

    inner_base_kernel = _mala_inner_kernel if inner_kernel_requested == "mala" else _hmc_inner_kernel
    inner_kernel = ns_repeat_kernel(num_repeats=n_inner)(inner_base_kernel)
    delete = partial(ns_delete_fn, num_delete=n_delete)

    def update_inner_kernel_params(state, info, params):
        if params is None:
            params = _init_inner_params()
        if inner_kernel_requested != "hmc" or not bool(cfg.ns_hmc_adapt_mass_from_live):
            return params

        particles = getattr(state, "particles", None)
        if particles is None:
            return params
        pos = getattr(particles, "position", particles)
        pos = jnp.asarray(pos, dtype=dtype)
        if pos.ndim != 2 or pos.shape[1] != n_dim:
            return params

        var = jnp.var(pos, axis=0) + jnp.asarray(float(cfg.ns_hmc_mass_jitter), dtype=dtype)
        inv_mass = jnp.clip(
            var,
            jnp.asarray(float(cfg.ns_hmc_mass_min), dtype=dtype),
            jnp.asarray(float(cfg.ns_hmc_mass_max), dtype=dtype),
        )
        return {
            "step_size": jnp.asarray(params["step_size"], dtype=dtype),
            "inverse_mass_matrix": inv_mass,
            "num_integration_steps": jnp.asarray(params["num_integration_steps"], dtype=jnp.int32),
        }

    step_fn = ns_adaptive_build_kernel(
        log_prior_u,
        loglikelihood_for_blackjax,
        delete,
        inner_kernel,
        update_inner_kernel_params,
    )
    init_fn = partial(
        ns_adaptive_init,
        logprior_fn=log_prior_u,
        loglikelihood_fn=loglikelihood_for_blackjax,
        update_inner_kernel_params_fn=update_inner_kernel_params,
    )

    return SimpleNamespace(init=init_fn, step=step_fn), inner_kernel_requested


if cfg.run_inference:
    logger.info("Running inference with BlackJAX Nested Sampling (NSS)...")

    # BlackJAX nested sampling API landed after BlackJAX 1.0; fail loudly if unavailable.
    if not hasattr(blackjax, "nss"):
        raise ImportError(
            "This BlackJAX version does not expose blackjax.nss. "
            "Install a nested-sampling-capable BlackJAX build "
            "(for example handley-lab/blackjax@nested_sampling)."
        )

    # Import NS utilities lazily (older BlackJAX builds may not include them).
    try:
        from blackjax.ns.utils import finalise as ns_finalise  # type: ignore
        from blackjax.ns.utils import log_weights as ns_log_weights  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "blackjax.ns.utils not available. Install a nested-sampling-capable BlackJAX build, "
            "including blackjax.ns.utils."
        ) from e

    from jax.scipy.special import logsumexp

    n_live = int(cfg.ns_num_live)
    if n_live < 4:
        raise ValueError("cfg.ns_num_live must be >= 4 for NSS to be meaningful")

    if cfg.ns_num_delete is None:
        n_delete = int(round(float(cfg.ns_num_delete_frac) * float(n_live)))
    else:
        n_delete = int(cfg.ns_num_delete)
    n_delete = max(1, min(n_delete, n_live - 1))

    if cfg.ns_num_inner_steps is None:
        n_inner = int(round(float(cfg.ns_inner_steps_mult) * float(n_dim)))
    else:
        n_inner = int(cfg.ns_num_inner_steps)
    n_inner = max(1, n_inner)

    log_every = int(cfg.ns_log_every)
    eta_window = int(cfg.ns_eta_window)
    max_steps = int(cfg.ns_max_steps)
    target_dlogz = -float(cfg.ns_stop_delta_logz)

    logger.info(
        f"Building NSS: n_live={n_live}, num_delete={n_delete}, num_inner_steps={n_inner}, "
        f"stop_delta_logz={float(cfg.ns_stop_delta_logz):.2f}, log_every={log_every}, "
        f"inner_kernel_requested={str(cfg.ns_inner_kernel).strip().lower()}"
    )

    ns_algo, ns_inner_kernel_used = _build_nss_algorithm(n_delete=n_delete, n_inner=n_inner)
    logger.info(f"NSS inner kernel in use: {ns_inner_kernel_used}")

    rng_key = jax.random.PRNGKey(int(cfg.seed))
    rng_key, init_key = jax.random.split(rng_key)

    logger.info("Initializing NSS live points from the prior...")
    t_init0 = time.perf_counter()
    live_particles = sample_prior_u(init_key, n_live)
    live_state = ns_algo.init(live_particles)
    try:
        _ = float(jax.device_get(live_state.integrator.logZ_live - live_state.integrator.logZ))
    except Exception:
        pass
    logger.info(f"NSS state initialized in {time.perf_counter() - t_init0:.2f} s")

    ns_step = jax.jit(ns_algo.step)

    dead_points: List[Any] = []
    step_times: List[float] = []
    dlogz_history: List[float] = []
    t0 = time.time()
    n_steps_done = 0

    def _store_dead(dead_now: Any) -> None:
        if bool(cfg.ns_store_dead_on_host):
            dead_points.append(jax.device_get(dead_now))
        else:
            dead_points.append(dead_now)

    def _log_nss_progress(step_index: int, *, step_dt: float, state_now: Any, prefix: str = "") -> float:
        integrator_now = state_now.integrator
        logZ = float(jax.device_get(integrator_now.logZ))
        logZ_live = float(jax.device_get(integrator_now.logZ_live))
        dlogZ = logZ_live - logZ

        step_times.append(step_dt)
        dlogz_history.append(dlogZ)

        k_dt = min(eta_window, len(step_times))
        mean_dt = float(np.mean(step_times[-k_dt:]))
        eta_max_s = max(0.0, (max_steps - step_index) * mean_dt)

        eta_stop_s = float("nan")
        if len(dlogz_history) >= 2:
            k_dz = min(eta_window, len(dlogz_history))
            dz = np.diff(np.asarray(dlogz_history[-k_dz:], dtype=np.float64))
            mean_ddlogz = float(np.mean(dz))
            if mean_ddlogz < -1.0e-9 and dlogZ > target_dlogz:
                steps_left_stop = max(0.0, (target_dlogz - dlogZ) / mean_ddlogz)
                eta_stop_s = steps_left_stop * mean_dt

        dead_done = step_index * n_delete
        eta_bits = [f"eta_max≈{eta_max_s/60.0:.1f} min"]
        if math.isfinite(eta_stop_s):
            eta_bits.insert(0, f"eta_stop≈{eta_stop_s/60.0:.1f} min")
        eta_msg = ", ".join(eta_bits)

        prefix_msg = f"{prefix}: " if prefix else ""
        logger.info(
            f"{prefix_msg}NSS step {step_index:04d}: "
            f"logZ={logZ:.3f}, logZ_live={logZ_live:.3f}, dlogZ={dlogZ:.3f}, "
            f"dead≈{dead_done}, step_time={step_dt:.2f}s, mean_step_time={mean_dt:.2f}s, {eta_msg}"
        )
        return dlogZ

    logger.info("Running first NSS step (includes first-time XLA compile and can be slow)...")
    rng_key, subkey = jax.random.split(rng_key)
    t_step0 = time.perf_counter()
    live_state, dead = ns_step(subkey, live_state)
    first_dt = time.perf_counter() - t_step0
    _store_dead(dead)
    n_steps_done = 1
    dlogZ = _log_nss_progress(1, step_dt=first_dt, state_now=live_state, prefix="warmup")

    if dlogZ < target_dlogz:
        logger.info("NSS: termination criterion reached after the first step (remaining evidence is negligible).")
    else:
        for step_index in range(2, max_steps + 1):
            if cfg.ns_max_seconds is not None and (time.time() - t0) > float(cfg.ns_max_seconds):
                logger.info("NSS: reached cfg.ns_max_seconds; stopping.")
                break

            rng_key, subkey = jax.random.split(rng_key)
            t_step = time.perf_counter()
            live_state, dead = ns_step(subkey, live_state)
            step_dt = time.perf_counter() - t_step
            _store_dead(dead)
            n_steps_done = step_index

            should_log = (step_index <= 5) or (step_index % log_every == 0)
            integrator = live_state.integrator
            dlogZ = float(jax.device_get(integrator.logZ_live - integrator.logZ)) if should_log else float("nan")
            if should_log:
                dlogZ = _log_nss_progress(step_index, step_dt=step_dt, state_now=live_state)

            if (not math.isnan(dlogZ) and dlogZ < target_dlogz) or (
                should_log is False and float(jax.device_get(integrator.logZ_live - integrator.logZ)) < target_dlogz
            ):
                logger.info("NSS: termination criterion reached (remaining evidence is negligible).")
                break

            if cfg.ns_max_dead_points is not None and n_steps_done * n_delete >= int(cfg.ns_max_dead_points):
                logger.info("NSS: reached cfg.ns_max_dead_points; stopping.")
                break

    if n_steps_done == 0:
        raise RuntimeError("NSS terminated before producing any dead points.")

    # Finalise the NS run (collect dead+live points in a single structure for weighting/sampling).
    ns_run = ns_finalise(live_state, dead_points)

    rng_key, weight_key, draw_key = jax.random.split(rng_key, 3)

    # Bootstrap the evidence estimate from the NS importance weights.
    log_w = ns_log_weights(weight_key, ns_run, shape=int(cfg.ns_logz_bootstrap))
    logzs = logsumexp(log_w, axis=0)
    logZ_mean = float(np.mean(np.asarray(jax.device_get(logzs))))
    logZ_std = float(np.std(np.asarray(jax.device_get(logzs))))

    # Use the mean log-weights for deterministic posterior resampling and ESS reporting.
    log_w_mean = jnp.mean(log_w, axis=-1)
    probs = jax.nn.softmax(log_w_mean - jnp.max(log_w_mean))
    ess_val = float(np.asarray(jax.device_get(1.0 / jnp.sum(jnp.square(probs)))))

    try:
        evals_val = int(
            np.asarray(
                jax.device_get(ns_run.update_info.num_steps.sum() + ns_run.update_info.num_shrink.sum())
            )
        )
    except Exception:
        evals_val = -1

    runtime_val = float(time.time() - t0)

    n_total_ns_particles = int(ns_run.particles.loglikelihood.shape[0])
    ns_dead_points = int(max(0, n_total_ns_particles - n_live))

    inner_params_final = getattr(live_state, "inner_kernel_params", None)
    ns_inner_step_size = float("nan")
    ns_inner_num_integration_steps = -1
    if isinstance(inner_params_final, dict):
        if "step_size" in inner_params_final:
            try:
                ns_inner_step_size = float(np.asarray(jax.device_get(inner_params_final["step_size"])))
            except Exception:
                ns_inner_step_size = float("nan")
        if "num_integration_steps" in inner_params_final:
            try:
                ns_inner_num_integration_steps = int(
                    np.asarray(jax.device_get(inner_params_final["num_integration_steps"]))
                )
            except Exception:
                ns_inner_num_integration_steps = -1

    logger.info(
        f"NSS finished: logZ≈{logZ_mean:.3f} ± {logZ_std:.3f} (bootstrap), "
        f"ESS≈{ess_val:.1f}, evals={evals_val}, wall_time={runtime_val:.2f}s"
    )

    # Draw posterior samples by importance-resampling the stored NS particles.
    n_draws_total = int(cfg.num_chains) * int(cfg.num_samples)
    idx = jax.random.choice(
        draw_key,
        n_total_ns_particles,
        shape=(n_draws_total,),
        p=probs,
        replace=True,
    )
    u_draws = ns_run.particles.position[idx]
    theta_draws = jax.vmap(theta_from_u)(u_draws)
    theta_np = np.asarray(theta_draws, dtype=np_float_dtype()).reshape(
        int(cfg.num_chains), int(cfg.num_samples), n_dim
    )

    save_npz(
        samples_path,
        param_names=np.asarray(param_names, dtype="<U64"),
        param_labels=np.asarray(param_labels, dtype="<U64"),
        samples=theta_np,
        inferred_param_names=np.asarray(param_names, dtype="<U64"),
        inferred_param_labels=np.asarray(param_labels, dtype="<U64"),
        inferred_param_truth=np.asarray(param_truth, dtype=np.float64),
    )
    logger.info(f"Saved posterior samples to: {samples_path}")

    save_npz(
        extra_path,
        inference_method=np.asarray("nss", dtype="<U16"),
        ns_backend=np.asarray("blackjax.nss", dtype="<U32"),
        ns_blackjax_version=np.asarray(str(getattr(blackjax, "__version__", "unknown")), dtype="<U64"),
        ns_inner_kernel_requested=np.asarray(str(cfg.ns_inner_kernel).strip().lower(), dtype="<U32"),
        ns_inner_kernel_used=np.asarray(str(ns_inner_kernel_used), dtype="<U32"),
        ns_inner_step_size=np.asarray(ns_inner_step_size, dtype=np.float64),
        ns_inner_num_integration_steps=np.asarray(ns_inner_num_integration_steps, dtype=np.int32),
        ns_num_live=np.asarray(n_live, dtype=np.int32),
        ns_num_delete=np.asarray(n_delete, dtype=np.int32),
        ns_num_inner_steps=np.asarray(n_inner, dtype=np.int32),
        ns_steps=np.asarray(n_steps_done, dtype=np.int32),
        ns_dead_points=np.asarray(ns_dead_points, dtype=np.int32),
        ns_logZ_mean=np.asarray(logZ_mean, dtype=np.float64),
        ns_logZ_std=np.asarray(logZ_std, dtype=np.float64),
        ns_ess=np.asarray(ess_val, dtype=np.float64),
        ns_evals=np.asarray(evals_val, dtype=np.int64),
        ns_runtime_seconds=np.asarray(runtime_val, dtype=np.float64),
        inferred_param_names=np.asarray(param_names, dtype="<U64"),
        inferred_param_truth=np.asarray(param_truth, dtype=np.float64),
    )
    logger.info(f"Saved NSS diagnostics to: {extra_path}")

if not cfg.run_inference:
    logger.info("cfg.run_inference=False; skipping inference. (Plot script requires posterior_samples.npz)")
    if not samples_path.exists():
        raise FileNotFoundError(f"posterior_samples.npz not found at {samples_path}")


# =============================================================================
# Posterior predictive (optional)
# =============================================================================


ppc_path = cfg.out_dir / "posterior_predictive.npz"
ppc_quant_path = cfg.out_dir / "posterior_predictive_quantiles.npz"

if cfg.do_ppc:
    logger.info("Computing posterior predictive phase curves (subset of draws)...")

    s = np.load(samples_path)
    theta_all = np.asarray(s["samples"]).reshape(-1, n_dim).astype(np_float_dtype())

    rng = np.random.default_rng(cfg.seed + 1)
    n_available = theta_all.shape[0]
    n_take = min(int(cfg.ppc_draws), n_available)
    take_idx = rng.choice(n_available, size=n_take, replace=False)

    theta_sel = theta_all[take_idx]
    theta_sel_jax = jnp.asarray(theta_sel, dtype=float_dtype())

    @jax.jit
    def _batch_forward(theta_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda th: phase_curve_model_jit(th))(theta_batch)

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

    save_npz(ppc_path, ppc_draws=ppc_draws, theta_sel=theta_sel, times_days=times_days)
    save_npz(ppc_quant_path, **ppc_q, times_days=times_days)

    logger.info(f"Saved PPC draws to: {ppc_path}")
    logger.info(f"Saved PPC quantiles to: {ppc_quant_path}")
else:
    logger.info("cfg.do_ppc=False; skipping posterior predictive computation.")


# =============================================================================
# Save truth + posterior-summary maps (so plotting does NOT rerun SWAMP)
# =============================================================================


maps_path = cfg.out_dir / "maps_truth_and_posterior_summary.npz"


def compute_maps_for_theta(theta: jnp.ndarray) -> Dict[str, np.ndarray]:
    """Compute terminal Phi/T/I and y_dense for a given physical theta vector."""
    p = _theta_vector_to_model_kwargs(theta)

    taurad_s = jnp.asarray(tau_hours_to_seconds(p["tau_rad_hours"]), dtype=float_dtype())
    taudrag_s = jnp.asarray(tau_hours_to_seconds(p["tau_drag_hours"]), dtype=float_dtype())

    k6phi_val = p["K6Phi"]
    if k6phi_val is not None:
        k6phi_val = jnp.asarray(k6phi_val, dtype=float_dtype())

    phi = swamp_terminal_phi(
        taurad_s,
        taudrag_s,
        Phibar=jnp.asarray(p["Phibar"], dtype=float_dtype()),
        DPhieq=jnp.asarray(p["DPhieq"], dtype=float_dtype()),
        K6=jnp.asarray(p["K6"], dtype=float_dtype()),
        K6Phi=k6phi_val,
        omega=jnp.asarray(p["omega_rad_s"], dtype=float_dtype()),
        a=jnp.asarray(p["a_planet_m"], dtype=float_dtype()),
        g=jnp.asarray(p["g_m_s2"], dtype=float_dtype()),
    )

    T = phi_to_temperature(phi)
    I_map = temperature_to_intensity(T)
    y_dense, y0_raw = intensity_map_to_y_dense_and_monopole(I_map)

    return {
        "phi": np.asarray(phi),
        "T": np.asarray(T),
        "I": np.asarray(I_map),
        "y_dense": np.asarray(y_dense),
        "y0_raw": np.asarray(y0_raw),
    }


logger.info("Computing truth and posterior-summary terminal maps...")

truth_maps = compute_maps_for_theta(theta_truth)

s = np.load(samples_path)
theta_flat = np.asarray(s["samples"]).reshape(-1, n_dim)
theta_median = np.median(theta_flat, axis=0).astype(np_float_dtype())
theta_median_jax = jnp.asarray(theta_median, dtype=float_dtype())

post_maps = compute_maps_for_theta(theta_median_jax)

save_npz(
    maps_path,
    lon=np.asarray(lon),
    lat=np.asarray(lat),
    phi_truth=truth_maps["phi"],
    T_truth=truth_maps["T"],
    I_truth=truth_maps["I"],
    y_truth=truth_maps["y_dense"],
    y0_truth=truth_maps["y0_raw"],
    phi_post=post_maps["phi"],
    T_post=post_maps["T"],
    I_post=post_maps["I"],
    y_post=post_maps["y_dense"],
    y0_post=post_maps["y0_raw"],
    inferred_param_names=np.asarray(param_names, dtype="<U64"),
    inferred_param_truth=np.asarray(param_truth, dtype=np.float64),
    inferred_param_post_median=np.asarray(theta_median, dtype=np.float64),
)
logger.info(f"Saved truth + posterior-median maps to: {maps_path}")

logger.info("DONE.")
logger.info(f"Outputs saved to: {cfg.out_dir}")
