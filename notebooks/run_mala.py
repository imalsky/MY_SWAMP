#!/usr/bin/env python3
"""
run.py

End-to-end differentiable SWAMP (my_swamp) + starry (jaxoplanet) retrieval with
BlackJAX adaptive tempered SMC.

What this script does
---------------------
1) Builds a SWAMP static model (grid, spectral basis, etc.) once to define the pixel grid.
2) Precomputes a robust linear projector from the SWAMP pixel grid -> starry spherical harmonics,
   using starry's own Surface.intensity() evaluation to avoid convention mismatches.
3) Defines a JAX-traceable forward model mapping a parameter vector -> SWAMP terminal Phi
   -> temperature proxy -> intensity map -> starry phase curve.
4) Generates synthetic observations (optional) and runs Bayesian inference with adaptive tempered SMC.
5) Writes everything needed for plotting into cfg.out_dir; plot.py never reruns SWAMP.

Important design choices / fixes relative to the previous version
-----------------------------------------------------------------
- Output filenames are consistent with the header docstring and plot.py. The previous version's
  filenames had a "_mala" suffix in code but not in the docstring, and plot.py was reading
  different filenames (bug).
- Priors on tau_rad and tau_drag are Uniform *in linear hours* with independently configurable
  bounds (as requested). Previously the prior was Uniform in log10(tau_hours).
- Inference can be toggled parameter-by-parameter (bools in Config). Parameters not inferred
  are held fixed at the values in Config.
- The SMC mutation kernel step_size is passed with a shape that matches the parameter dimension
  (previously it was shape (1,), which can degrade mixing or error depending on BlackJAX version).
- More conservative default SMC settings are used (more MCMC steps and tempering steps) to reduce
  weight collapse; these are common causes of "nonsense" posteriors in expensive forward models.

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
- mcmc_extra_fields.npz                  (SMC diagnostics; name kept for plot compatibility)
- posterior_predictive.npz               (optional)
- posterior_predictive_quantiles.npz     (optional)
- maps_truth_and_posterior_summary.npz   (truth + posterior-median maps; for plotting without reruns)

Run `plot.py` after this completes.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
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
    # NOTE: my_swamp defaults to enabling x64 at import time (see README). We keep an explicit switch.
    use_x64: bool = False
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
    # Inference: BlackJAX Adaptive Tempered SMC (batched particles)
    # -----------------------
    run_inference: bool = True

    smc_num_particles: int = 64

    # Adaptive tempering: choose beta increments to keep ESS near this fraction of N.
    # Lower => fewer steps, higher => more steps and more robustness against degeneracy.
    smc_target_ess_frac: float = 0.6

    # Mutation: number of MCMC steps per tempering stage (applied to each particle)
    smc_num_mcmc_steps: int = 32

    # Mutation kernel
    smc_mcmc_kernel: str = "mala"  # {"mala", "hmc"}

    # MALA parameters (in unconstrained u-space)
    mala_step_size: float = 0.2

    # HMC parameters (in unconstrained u-space)
    hmc_step_size: float = 0.07
    hmc_num_integration_steps: int = 8

    # -----------------------
    # Optional: automatic tuning of MCMC step size (cheap pilot run)
    # -----------------------
    # If True, run a small pilot MCMC at a tempered target to auto-tune the kernel step size.
    # This is usually the highest-leverage fix when MALA acceptance is ~0.95-0.99 (step too small).
    mcmc_auto_tune: bool = True

    # Tune against a tempered target: logprior(u) + beta * loglikelihood(u)
    mcmc_tune_beta: float = 0.5

    # Work per tuning iteration (kept small; each step is still expensive due to SWAMP)
    mcmc_tune_particles: int = 8
    mcmc_tune_steps: int = 8
    mcmc_tune_iters: int = 8

    # Target acceptance rates (heuristics)
    mcmc_target_accept_mala: float = 0.75
    mcmc_target_accept_hmc: float = 0.80

    # Step size bounds (in u-space)
    mcmc_step_size_min: float = 1.0e-3
    mcmc_step_size_max: float = 5.0

    # Robbins–Monro gain for log(step_size) updates (larger = faster, noisier)
    mcmc_tune_gain: float = 0.7

    # Resampling scheme
    smc_resampling: str = "systematic"  # {"systematic", "stratified", "multinomial"}

    # Safety: max number of adaptive tempering steps before giving up.
    # The previous version used 24, which is often too small if smc_target_ess_frac is high.
    smc_max_steps: int = 32

    # Gradient safety:
    # If True, override gradients of the likelihood with a custom VJP that uses forward-mode JVPs.
    # This is cheap and memory-stable for low parameter dimension but scales ~O(D) per likelihood eval.
    smc_use_custom_gradients: bool = True
    smc_custom_grad_max_dim: int = 8

    # Posterior samples to save (for plotting). These are resampled from the final weighted SMC particle set.
    num_samples: int = smc_num_particles
    num_chains: int = 2

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

_valid_smc_kernels = {"mala", "hmc"}
if str(cfg.smc_mcmc_kernel).strip().lower() not in _valid_smc_kernels:
    raise ValueError(f"cfg.smc_mcmc_kernel must be one of {_valid_smc_kernels}, got {cfg.smc_mcmc_kernel!r}")

_valid_resampling = {"systematic", "stratified", "multinomial"}
if str(cfg.smc_resampling).strip().lower() not in _valid_resampling:
    raise ValueError(f"cfg.smc_resampling must be one of {_valid_resampling}, got {cfg.smc_resampling!r}")

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
    from blackjax.smc import adaptive_tempered as smc_adaptive_tempered  # type: ignore
    from blackjax.smc import resampling as smc_resampling  # type: ignore
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


def intensity_map_to_y_dense(I_map: jnp.ndarray) -> jnp.ndarray:
    """Convert SWAMP-grid intensity map (J,I) -> dense starry coefficients with stability guards.

    Key protections:
    - Compute the weighted mean using only finite pixels.
    - Guard division by mean and by y[0].
    - If any non-finite values are present (or the normalization fails), return all-NaN so the
      likelihood can reject this particle deterministically.
    """
    dtype = float_dtype()
    I_flat = jnp.asarray(I_map, dtype=dtype).reshape(-1)

    finite_mask = jnp.isfinite(I_flat)
    all_finite_in = jnp.all(finite_mask)

    # Use only finite pixels to compute mean; if none are finite, mean is invalid.
    w_eff = jnp.asarray(w_pix, dtype=dtype) * finite_mask
    w_sum = jnp.sum(w_eff)

    # Replace non-finite values with 0 for the RHS; mean uses only finite pixels anyway.
    I_safe = jnp.where(finite_mask, I_flat, jnp.asarray(0.0, dtype=dtype))

    def _mean_ok(_: None) -> jnp.ndarray:
        return jnp.sum(w_eff * I_safe) / w_sum

    def _mean_bad(_: None) -> jnp.ndarray:
        return jnp.asarray(jnp.nan, dtype=dtype)

    I_mean = jax.lax.cond(w_sum > jnp.asarray(0.0, dtype=dtype), _mean_ok, _mean_bad, operand=None)

    eps = jnp.asarray(1.0e-30, dtype=dtype)
    mean_ok = jnp.isfinite(I_mean) & (jnp.abs(I_mean) > eps)

    # Normalize intensity map; if mean is invalid, force invalid output (NaNs).
    I_rel = jnp.where(mean_ok, I_safe / I_mean, jnp.asarray(jnp.nan, dtype=dtype))

    # Solve for y in weighted LSQ sense
    rhs = jnp.asarray(w_sqrt, dtype=dtype) * I_rel
    y = jnp.asarray(projector, dtype=dtype) @ rhs

    y0 = y[0]
    y0_ok = jnp.isfinite(y0) & (jnp.abs(y0) > eps)
    y = jnp.where(y0_ok, y / y0, jnp.full_like(y, jnp.asarray(jnp.nan, dtype=dtype)))

    # Final validity: require finite input + finite normalizations + finite output
    out_ok = all_finite_in & mean_ok & y0_ok & _all_finite(y)
    y = jax.lax.cond(
        out_ok,
        lambda _: y,
        lambda _: jnp.full_like(y, jnp.asarray(jnp.nan, dtype=dtype)),
        operand=None,
    )
    return y

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

        y_dense = intensity_map_to_y_dense(I_map)
        ylm = ylm_from_dense(y_dense, lm_list)

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
            amplitude=jnp.asarray(p["planet_fpfs"], dtype=dtype),
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
# u-space parameterization (SMC lives in unconstrained R^D)
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
# Posterior + inference (BlackJAX Adaptive Tempered SMC)
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


_use_custom_grads = bool(cfg.smc_use_custom_gradients) and (n_dim <= int(cfg.smc_custom_grad_max_dim))
if cfg.smc_use_custom_gradients and not _use_custom_grads:
    logger.warning(
        "cfg.smc_use_custom_gradients=True but n_dim=%d > smc_custom_grad_max_dim=%d. "
        "Disabling custom forward-mode gradients.",
        n_dim,
        int(cfg.smc_custom_grad_max_dim),
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


# --- Automatic MCMC step-size tuning (pilot run) ----------------------------


def _extract_acceptance_rate(info: Any) -> jnp.ndarray:
    """Best-effort extraction of an acceptance statistic from a BlackJAX kernel info object."""
    # Common names across kernels/versions
    for attr in ("acceptance_rate", "acceptance_probability", "accept_prob", "prob_accept"):
        if hasattr(info, attr):
            return jnp.asarray(getattr(info, attr), dtype=float_dtype())

    # Some kernels expose a boolean "is_accepted"
    if hasattr(info, "is_accepted"):
        return jnp.asarray(getattr(info, "is_accepted"), dtype=float_dtype())

    # SMC wrappers often nest the kernel info
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


def tune_mcmc_step_size(rng_key: jax.Array) -> float:
    """Auto-tune MALA/HMC step size in u-space with a small pilot run.

    This targets a mean acceptance rate using a Robbins–Monro update in log(step_size).
    It is intentionally simple and robust (no extra deps, no reliance on optional BlackJAX adapters).

    Returns a Python float step size.
    """
    if not bool(cfg.mcmc_auto_tune):
        if str(cfg.smc_mcmc_kernel).strip().lower() == "mala":
            return float(cfg.mala_step_size)
        return float(cfg.hmc_step_size)

    kernel = str(cfg.smc_mcmc_kernel).strip().lower()
    beta = jnp.asarray(float(cfg.mcmc_tune_beta), dtype=float_dtype())

    n_particles = int(cfg.mcmc_tune_particles)
    n_steps = int(cfg.mcmc_tune_steps)
    n_iters = int(cfg.mcmc_tune_iters)

    step_min = float(cfg.mcmc_step_size_min)
    step_max = float(cfg.mcmc_step_size_max)
    gain = float(cfg.mcmc_tune_gain)

    if n_particles <= 0 or n_steps <= 0 or n_iters <= 0:
        raise ValueError("MCMC tuning counts must be positive.")
    if not (0.0 < float(beta) <= 1.0):
        raise ValueError(f"cfg.mcmc_tune_beta must be in (0,1], got {float(beta)}")
    if not (0.0 < float(gain) <= 5.0):
        raise ValueError(f"cfg.mcmc_tune_gain must be in (0,5], got {gain}")
    if not (0.0 < step_min < step_max):
        raise ValueError(f"Invalid step size bounds: min={step_min}, max={step_max}")

    if kernel == "mala":
        target = float(cfg.mcmc_target_accept_mala)
        step0 = float(cfg.mala_step_size)
    elif kernel == "hmc":
        target = float(cfg.mcmc_target_accept_hmc)
        step0 = float(cfg.hmc_step_size)
    else:  # pragma: no cover
        raise ValueError(f"Unknown cfg.smc_mcmc_kernel={cfg.smc_mcmc_kernel!r}")

    if not (0.0 < target < 1.0):
        raise ValueError(f"Target acceptance must be in (0,1), got {target}")

    def logdensity(u: jnp.ndarray) -> jnp.ndarray:
        return log_prior_u(u) + beta * loglikelihood_for_blackjax(u)

    # Pilot initial positions from the prior
    rng_key, subkey = jax.random.split(rng_key)
    u0 = sample_prior_u(subkey, n_particles)

    if kernel == "mala":
        state = jax.vmap(lambda uu: _mala_init_one(uu, logdensity))(u0)

        @jax.jit
        def _run_block(key_in: jax.Array, state_in, step_size_scalar: jnp.ndarray):
            def one_step(carry, _):
                key, st = carry
                key, sub = jax.random.split(key)
                keys = jax.random.split(sub, n_particles)

                st_new, info = jax.vmap(lambda kk, ss: _mala_step_one(kk, ss, logdensity, step_size_scalar))(keys, st)
                acc = jnp.mean(_extract_acceptance_rate(info))
                return (key, st_new), acc

            (key_out, state_out), accs = jax.lax.scan(one_step, (key_in, state_in), None, length=n_steps)
            return key_out, state_out, jnp.mean(accs)

    else:
        # HMC pilot uses identity inverse mass matrix (diag), fixed num integration steps.
        inv_mass = jnp.ones((n_dim,), dtype=float_dtype())
        n_leapfrog = jnp.asarray(int(cfg.hmc_num_integration_steps), dtype=jnp.int32)

        state = jax.vmap(lambda uu: _hmc_init_one(uu, logdensity))(u0)

        @jax.jit
        def _run_block(key_in: jax.Array, state_in, step_size_scalar: jnp.ndarray):
            def one_step(carry, _):
                key, st = carry
                key, sub = jax.random.split(key)
                keys = jax.random.split(sub, n_particles)

                st_new, info = jax.vmap(
                    lambda kk, ss: _hmc_step_one(
                        kk,
                        ss,
                        logdensity,
                        step_size_scalar,
                        inv_mass,
                        n_leapfrog,
                    )
                )(keys, st)
                acc = jnp.mean(_extract_acceptance_rate(info))
                return (key, st_new), acc

            (key_out, state_out), accs = jax.lax.scan(one_step, (key_in, state_in), None, length=n_steps)
            return key_out, state_out, jnp.mean(accs)

    log_step = math.log(min(max(step0, step_min), step_max))

    tol = 0.05
    for it in range(n_iters):
        step_jax = jnp.asarray(math.exp(log_step), dtype=float_dtype())
        rng_key, state, acc = _run_block(rng_key, state, step_jax)
        acc_f = float(jax.device_get(acc))

        # Robbins–Monro update in log space
        log_step = log_step + gain * (acc_f - target)
        step_now = min(max(math.exp(log_step), step_min), step_max)
        log_step = math.log(step_now)

        # Early exit once we're reasonably close to target (saves expensive pilot work)
        if it >= 2 and abs(acc_f - target) < tol:
            break

    tuned = float(math.exp(log_step))
    logger.info(
        f"Auto-tuned {kernel.upper()} step size (u-space): {tuned:.4g} "
        f"(target_accept={target:.2f}, beta_tune={float(beta):.2f}, "
        f"pilot_particles={n_particles}, pilot_steps={n_steps}, iters={n_iters})"
    )
    return tuned


# --- Build the BlackJAX SMC algorithm ----------------------------------------


def build_smc_algorithm(
    *,
    step_size_override: Optional[float] = None,
    inverse_mass_diag_override: Optional[jnp.ndarray] = None,
):
    kernel = str(cfg.smc_mcmc_kernel).strip().lower()

    if kernel == "mala":
        mcmc_step_fn = blackjax.mala.build_kernel()
        mcmc_init_fn = blackjax.mala.init
        N = int(cfg.smc_num_particles)

        step = float(cfg.mala_step_size) if step_size_override is None else float(step_size_override)
        mcmc_parameters = {
            "step_size": jnp.full((N,), step, dtype=float_dtype())
        }
        target_accept_str = "(typical good range ~0.4-0.8; optimal ~0.57 in high-d)"

    elif kernel == "hmc":
        mcmc_step_fn = blackjax.hmc.build_kernel()
        mcmc_init_fn = blackjax.hmc.init
        N = int(cfg.smc_num_particles)
        dim = int(n_dim)

        step = float(cfg.hmc_step_size) if step_size_override is None else float(step_size_override)

        inv_mass_diag = (
            jnp.ones((dim,), dtype=float_dtype())
            if inverse_mass_diag_override is None
            else jnp.asarray(inverse_mass_diag_override, dtype=float_dtype())
        )
        if tuple(inv_mass_diag.shape) != (dim,):
            raise ValueError(f"inverse_mass_diag_override must have shape ({dim},), got {tuple(inv_mass_diag.shape)}")

        mcmc_parameters = {
            "step_size": jnp.full((N,), step, dtype=float_dtype()),
            # Per-particle diagonal inverse mass matrix (shape: (N, dim))
            "inverse_mass_matrix": jnp.tile(inv_mass_diag[None, :], (N, 1)),
            "num_integration_steps": jnp.full((N,), int(cfg.hmc_num_integration_steps), dtype=jnp.int32),
        }

        target_accept_str = "(typical good range ~0.6-0.9)"

    else:  # pragma: no cover
        raise ValueError(f"Unknown cfg.smc_mcmc_kernel={cfg.smc_mcmc_kernel!r}")

    resampling_name = str(cfg.smc_resampling).strip().lower()
    if resampling_name == "systematic":
        resampling_fn = smc_resampling.systematic
    elif resampling_name == "stratified":
        resampling_fn = smc_resampling.stratified
    elif resampling_name == "multinomial":
        resampling_fn = smc_resampling.multinomial
    else:  # pragma: no cover
        raise ValueError(f"Unknown cfg.smc_resampling={cfg.smc_resampling!r}")

    target_ess_frac = float(cfg.smc_target_ess_frac)
    if (not math.isfinite(target_ess_frac)) or target_ess_frac <= 0.0 or target_ess_frac > 1.0:
        raise ValueError(f"cfg.smc_target_ess_frac must be in (0, 1]. Got {cfg.smc_target_ess_frac!r}.")

    step_for_log = step
    logger.info(
        "Building adaptive tempered SMC: "
        f"kernel={kernel}, N={cfg.smc_num_particles}, "
        f"target_ess_frac={target_ess_frac:.3f}, "
        f"num_mcmc_steps={cfg.smc_num_mcmc_steps}, "
        f"step_size={step_for_log:.4g} {target_accept_str}"
    )

    smc = smc_adaptive_tempered.as_top_level_api(
        logprior_fn=log_prior_u,
        loglikelihood_fn=loglikelihood_for_blackjax,
        mcmc_step_fn=mcmc_step_fn,
        mcmc_init_fn=mcmc_init_fn,
        mcmc_parameters=mcmc_parameters,
        resampling_fn=resampling_fn,
        num_mcmc_steps=int(cfg.smc_num_mcmc_steps),
        target_ess=target_ess_frac,
    )
    return smc



# --- Run inference ------------------------------------------------------------


if cfg.run_inference:
    logger.info("Running inference with BlackJAX adaptive tempered SMC...")

    key = jax.random.PRNGKey(int(cfg.seed))
    key, subkey = jax.random.split(key)

    particles0 = sample_prior_u(subkey, int(cfg.smc_num_particles))

    tuned_step_size: Optional[float] = None
    if bool(cfg.mcmc_auto_tune):
        key, tune_key = jax.random.split(key)
        tuned_step_size = tune_mcmc_step_size(tune_key)

    kernel_name = str(cfg.smc_mcmc_kernel).strip().lower()
    if kernel_name == "mala":
        mcmc_step_size_used = float(cfg.mala_step_size) if tuned_step_size is None else float(tuned_step_size)
        inv_mass_diag_used = None
    else:
        mcmc_step_size_used = float(cfg.hmc_step_size) if tuned_step_size is None else float(tuned_step_size)
        inv_mass_diag_used = None

    smc = build_smc_algorithm(step_size_override=mcmc_step_size_used, inverse_mass_diag_override=inv_mass_diag_used)
    smc_step = jax.jit(smc.step)
    state = smc.init(particles0)

    # Force compilation up front (so progress bar doesn't hang at 0% with no explanation).
    logger.info("Compiling smc_step (first-time XLA compile can be long)...")
    key, subkey = jax.random.split(key)

    t_compile0 = time.perf_counter()
    did_execute_warmup = False
    try:
        smc_step.lower(subkey, state).compile()
    except Exception:
        _state_tmp, _info_tmp = smc_step(subkey, state)
        jax.block_until_ready(_state_tmp)
        did_execute_warmup = True
    logger.info(f"smc_step compilation finished in {time.perf_counter() - t_compile0:.1f} s")

    if did_execute_warmup:
        state = smc.init(particles0)

    betas: List[float] = [0.0]
    ess_hist: List[float] = []
    acc_hist: List[float] = []
    logz_inc_hist: List[float] = []
    step_times: List[float] = []
    beta_increments: List[float] = []

    pbar = tqdm(range(int(cfg.smc_max_steps)), desc="Adaptive tempered SMC", leave=True)

    for i in pbar:
        key, subkey = jax.random.split(key)

        t_step0 = time.perf_counter()
        state, info = smc_step(subkey, state)
        jax.block_until_ready(state)
        dt = time.perf_counter() - t_step0

        beta = float(jax.device_get(state.tempering_param))
        w = state.weights
        ess = float(jax.device_get(1.0 / jnp.sum(w * w)))

        if (not math.isfinite(beta)) or (not math.isfinite(ess)):
            raise RuntimeError(
                f"Non-finite SMC diagnostics at step {i:03d}: beta={beta}, ess={ess}. "
                "This usually indicates NaNs/Infs in the log-likelihood."
            )

        acc = float("nan")
        try:
            acc = float(jax.device_get(jnp.mean(info.update_info.acceptance_rate)))
        except Exception:
            pass

        step_times.append(dt)
        if len(betas) >= 1:
            beta_increments.append(beta - betas[-1])

        betas.append(beta)
        ess_hist.append(ess)
        acc_hist.append(acc)

        # BlackJAX info naming differs slightly across versions; be defensive.
        logz_inc = float("nan")
        for key_name in ("log_likelihood_increment", "logZ_increment", "log_normalizer_increment"):
            if hasattr(info, key_name):
                try:
                    logz_inc = float(jax.device_get(getattr(info, key_name)))
                    break
                except Exception:
                    pass
        if math.isnan(logz_inc):
            # Keep array lengths consistent; plot.py will ignore NaNs.
            logz_inc = float("nan")
        logz_inc_hist.append(logz_inc)

        k = min(5, len(step_times))
        mean_dt = float(np.mean(step_times[-k:]))

        est_steps_left = float(cfg.smc_max_steps - i - 1)
        if len(beta_increments) >= 2:
            kb = min(5, len(beta_increments))
            mean_dbeta = float(np.mean(beta_increments[-kb:]))
            if mean_dbeta > 1e-6:
                est_steps_left = min(est_steps_left, (1.0 - beta) / mean_dbeta)
        eta_s = max(0.0, est_steps_left * mean_dt)

        pbar.set_postfix(beta=f"{beta:.3f}", ess=f"{ess:.1f}", acc=f"{acc:.3f}", step_s=f"{dt:.1f}", eta_min=f"{eta_s/60.0:.1f}")

        logger.info(
            f"SMC step {i:03d}: beta={beta:.6f}, ESS={ess:.1f}/{cfg.smc_num_particles}, "
            f"mean_accept={acc:.3f}, step_time={dt:.2f}s, eta≈{eta_s/60.0:.1f} min"
        )

        if beta >= 1.0 - 1e-8:
            logger.info("Reached beta=1.0 (posterior). Stopping tempering loop.")
            break

    final_beta = float(jax.device_get(state.tempering_param))
    if (not math.isfinite(final_beta)) or final_beta < 1.0 - 1e-6:
        raise RuntimeError(
            f"Adaptive tempering did not reach beta=1 (final_beta={final_beta}). "
            "Increase cfg.smc_max_steps and/or reduce cfg.smc_target_ess_frac."
        )

    # Resample posterior draws for downstream plotting.
    n_draws_total = int(cfg.num_chains) * int(cfg.num_samples)
    key, subkey = jax.random.split(key)

    idx = jax.random.choice(
        subkey,
        int(cfg.smc_num_particles),
        shape=(n_draws_total,),
        p=state.weights,
        replace=True,
    )
    u_draws = state.particles[idx]
    theta_draws = jax.vmap(theta_from_u)(u_draws)  # (n_draws_total, n_dim)

    theta_np = np.asarray(theta_draws, dtype=np.float64).reshape((int(cfg.num_chains), int(cfg.num_samples), n_dim))

    # Save in a generic format (param_names + samples cube).
    save_npz(
        samples_path,
        param_names=np.asarray(param_names, dtype="<U64"),
        param_labels=np.asarray(param_labels, dtype="<U64"),
        samples=theta_np,
    )
    logger.info(f"Saved posterior samples to: {samples_path}")

    # Save SMC diagnostics for plotting.
    betas_np = np.asarray(betas, dtype=np.float64)
    ess_np = np.asarray(ess_hist, dtype=np.float64)
    acc_np = np.asarray(acc_hist, dtype=np.float64)
    logz_inc_np = np.asarray(logz_inc_hist, dtype=np.float64)
    logz_np = np.cumsum(np.nan_to_num(logz_inc_np, nan=0.0))

    save_npz(
        extra_path,
        inference_method=np.asarray(2, dtype=np.int32),  # 2=blackjax_adaptive_tempered_smc (local convention)
        smc_kernel=np.asarray(str(cfg.smc_mcmc_kernel), dtype="<U16"),
        smc_resampling=np.asarray(str(cfg.smc_resampling), dtype="<U16"),
        smc_num_particles=np.asarray(int(cfg.smc_num_particles), dtype=np.int32),
        smc_num_mcmc_steps=np.asarray(int(cfg.smc_num_mcmc_steps), dtype=np.int32),
        smc_mcmc_step_size=np.asarray(float(mcmc_step_size_used), dtype=np.float64),
        smc_mcmc_step_size_auto_tuned=np.asarray(int(bool(cfg.mcmc_auto_tune)), dtype=np.int32),
        smc_target_ess_frac=np.asarray(float(cfg.smc_target_ess_frac), dtype=np.float64),
        smc_betas=betas_np,
        smc_ess=ess_np,
        smc_acceptance_rate=acc_np,
        smc_logZ_increment=logz_inc_np,
        smc_logZ=logz_np,
        smc_final_weights=np.asarray(jax.device_get(state.weights), dtype=np.float64),
        inferred_param_names=np.asarray(param_names, dtype="<U64"),
        inferred_param_truth=np.asarray(param_truth, dtype=np.float64),
    )
    logger.info(f"Saved SMC diagnostics to: {extra_path}")

else:
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
    y_dense = intensity_map_to_y_dense(I_map)

    return {
        "phi": np.asarray(phi),
        "T": np.asarray(T),
        "I": np.asarray(I_map),
        "y_dense": np.asarray(y_dense),
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
    phi_post=post_maps["phi"],
    T_post=post_maps["T"],
    I_post=post_maps["I"],
    y_post=post_maps["y_dense"],
    inferred_param_names=np.asarray(param_names, dtype="<U64"),
    inferred_param_truth=np.asarray(param_truth, dtype=np.float64),
    inferred_param_post_median=np.asarray(theta_median, dtype=np.float64),
)
logger.info(f"Saved truth + posterior-median maps to: {maps_path}")

logger.info("DONE.")
logger.info(f"Outputs saved to: {cfg.out_dir}")
