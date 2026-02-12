#!/usr/bin/env python3
"""
swamp_run.py

Run the full differentiable end-to-end pipeline:

  1) Build a MY_SWAMP (my_swamp) static model once
  2) Precompute a robust pixel-map -> starry(Ylm) linear projector on the SWAMP grid
     using starry's own Surface.intensity() evaluation (best practice for convention safety)
  3) Define a JAX-traceable forward model:
        (tau_rad, tau_drag) -> SWAMP terminal Phi -> T -> I -> Ylm -> starry phase curve
  4) Generate synthetic phase-curve data (or reuse an existing observations.npz)
  5) Run NumPyro NUTS/HMC retrieving only tau_rad and tau_drag
     **Sampling is done in log-space** (log10 hours) to ensure positivity and
     better exploration across orders of magnitude.
  6) Save everything needed for later plotting (no plotting here)
  7) Optimize in float32 for speed
This script is robust to small API differences across my_swamp and jaxoplanet versions by:
  - filtering kwargs against function signatures (dropping unexpected kwargs with logging)
  - wrapping starry calls whose signatures may change (e.g., Surface.intensity)
  - avoiding internal my_swamp helpers that may not exist (e.g., my_swamp.dtypes)

Outputs written to cfg.out_dir
------------------------------
- config.json
- run.log
- observations.npz
- posterior_samples.npz
- mcmc_extra_fields.npz                 (only if extra fields are available)
- posterior_predictive.npz              (optional)
- posterior_predictive_quantiles.npz    (optional)
- maps_truth_and_posterior_mean.npz

Run `swamp_plot.py` after this completes.

"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict, replace as dc_replace
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
    # SWAMP numerical params
    # -----------------------
    M: int = 42
    dt_seconds: float = 120.0
    model_days: float = 0.5          # <-- run SWAMP ONLY for N days
    starttime_index: int = 2         # leapfrog start index (>=2)

    # -----------------------
    # SWAMP physical params (treated as known)
    # -----------------------
    a_planet_m: float = 8.2e7
    omega_rad_s: float = 3.2e-5
    g_m_s2: float = 9.8
    Phibar: float = 3.0e5
    DPhieq: float = 1.0e6
    K6: float = 1.24e33
    K6Phi: Optional[float] = None    # additional Phi diffusion; None disables

    # RunFlags (treated as fixed)
    forcflag: bool = True
    diffflag: bool = True
    expflag: bool = False
    modalflag: bool = True
    alpha: float = 0.01
    diagnostics: bool = False        # must be False for clean JAX/HMC
    blowup_rms: float = 1.0e30

    # -----------------------
    # Phi -> temperature -> intensity (toy emission layer)
    # -----------------------
    T_ref: float = 300.0
    phi_to_T_scale: float = 4.0e4
    Tmin_K: float = 1.0

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
    # Orbit/system geometry (known)
    # -----------------------
    star_mass_msun: float = 1.0
    star_radius_rsun: float = 1.0
    planet_radius_rjup: float = 1.0
    impact_param: float = 0.0
    time_transit_days: float = 0.0
    orbital_period_override_days: Optional[float] = None  # if None, derived from omega

    # Flux scaling: planet/star flux ratio (fixed in retrieval)
    planet_fpfs: float = 1500e-6

    # -----------------------
    # Synthetic observations
    # -----------------------
    generate_synthetic_data: bool = True
    n_times: int = 250
    n_orbits_observed: float = 1.0
    obs_sigma: float = 80e-6

    taurad_true_hours: float = 18.0
    taudrag_true_hours: float = 6.0

    # -----------------------
    # Inference (NumPyro NUTS)
    # -----------------------
    run_mcmc: bool = True
    num_warmup: int = 50
    num_samples: int = 100
    num_chains: int = 1
    chain_method: str = "vectorized"
    target_accept_prob: float = 0.85

    # Priors on log10(tau_hours)  (LOG-SPACE SAMPLING)
    prior_log10_tau_hours_min: float = -1.0   # 0.1 hours
    prior_log10_tau_hours_max: float = 3.0    # 1000 hours

    # Autodiff option (only if supported by your NumPyro version)
    use_forward_mode_ad: bool = True

    # -----------------------
    # Posterior predictive (optional)
    # -----------------------
    do_ppc: bool = True
    ppc_draws: int = 64
    ppc_chunk_size: int = 16

    # -----------------------
    # Plot-related config saved for the plot script
    # -----------------------
    fig_dpi: int = 160
    render_res: int = 250
    render_phases: Tuple[float, ...] = (0.0, 0.25, 0.49, 0.51, 0.75)


cfg = Config()


# =============================================================================
# Environment + logging
# =============================================================================

# my_swamp may read this at import time in some builds
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
# Reduce backend probing noise (especially on macOS)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

(cfg.out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
logger.info(f"Wrote config to: {cfg.out_dir / 'config.json'}")


# =============================================================================
# Imports (after env vars)
# =============================================================================

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", cfg.use_x64)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import my_swamp.model as swamp_model
from my_swamp.model import RunFlags, build_static

from jaxoplanet.orbits.keplerian import Body, Central
from jaxoplanet.starry.light_curves import light_curve as starry_light_curve
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

logger.info(f"JAX backend: {jax.default_backend()}")
logger.info(f"JAX devices: {jax.devices()}")


# =============================================================================
# Utility helpers
# =============================================================================

def float_dtype() -> Any:
    """Local dtype helper (do NOT depend on my_swamp.dtypes, which may not exist)."""
    return jnp.float64 if cfg.use_x64 else jnp.float32

def tau_hours_to_seconds(x: Any) -> Any:
    return 3600.0 * x

def orbital_period_days_from_omega(omega_rad_s: float) -> float:
    return float((2.0 * math.pi / omega_rad_s) / 86400.0)

def compute_n_steps(model_days: float, dt_seconds: float) -> int:
    n = int(np.round(model_days * 86400.0 / dt_seconds))
    return max(n, 1)

def flatten_chain_draw(x: np.ndarray) -> np.ndarray:
    """(chains, draws, ...) -> (chains*draws, ...)"""
    return x.reshape((-1,) + x.shape[2:])

def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def call_with_filtered_kwargs(func, kwargs: Dict[str, Any], *, name: Optional[str] = None):
    """
    Call func(**kwargs), filtering out unexpected kwargs using inspect.signature.

    This prevents version-to-version errors like:
      TypeError: build_static() got an unexpected keyword argument '...'
    """
    fn_name = name or getattr(func, "__name__", repr(func))
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(**kwargs)

    filtered = {}
    dropped = []
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered[k] = v
        else:
            dropped.append(k)

    if dropped:
        logger.warning(f"{fn_name}: dropped unexpected kwargs: {dropped}")

    return func(**filtered)

def safe_replace_static(static_base: Any, taurad_s: Any, taudrag_s: Any, static_kwargs_for_rebuild: Dict[str, Any]) -> Any:
    """
    Replace taurad/taudrag on the Static object.

    Preferred: dataclasses.replace (fast; keeps all precomputed arrays).
    Fallback: rebuild static via build_static (slow; should not happen in HMC).
    """
    try:
        return dc_replace(static_base, taurad=taurad_s, taudrag=taudrag_s)
    except Exception as e:
        logger.error(
            "dataclasses.replace(static, taurad=..., taudrag=...) failed. "
            "Falling back to rebuilding static each call (VERY SLOW, not recommended). "
            f"Error: {e}"
        )
        kw = dict(static_kwargs_for_rebuild)
        kw["taurad"] = taurad_s
        kw["taudrag"] = taudrag_s
        return call_with_filtered_kwargs(build_static, kw, name="build_static (fallback)")

def phi_to_temperature(phi: jnp.ndarray) -> jnp.ndarray:
    """Toy Phi -> temperature mapping consistent with your notebook."""
    T = jnp.asarray(cfg.T_ref, dtype=float_dtype()) + phi / jnp.asarray(cfg.phi_to_T_scale, dtype=float_dtype())
    return jnp.maximum(T, jnp.asarray(cfg.Tmin_K, dtype=float_dtype()))

def temperature_to_intensity(T: jnp.ndarray) -> jnp.ndarray:
    """Bolometric Lambertian intensity proxy: I ∝ T^4."""
    return T**4

def build_lm_list(ydeg: int) -> List[Tuple[int, int]]:
    return [(ell, m) for ell in range(ydeg + 1) for m in range(-ell, ell + 1)]

def ylm_from_dense(y_dense: jnp.ndarray, lm_list: Sequence[Tuple[int, int]]) -> Ylm:
    """
    Create a starry Ylm object from a dense coefficient vector using a static (l,m) list.
    """
    data = {lm: y_dense[i] for i, lm in enumerate(lm_list)}
    return Ylm(data)

def surface_intensity(surf: Surface, latv: jnp.ndarray, lonv: jnp.ndarray) -> jnp.ndarray:
    """
    Call Surface.intensity with signature safety.

    Some jaxoplanet versions support intensity(lat, lon, theta=...); others do not.
    For map fitting, theta is irrelevant as long as we remain consistent.
    """
    try:
        sig = inspect.signature(surf.intensity)
        if "theta" in sig.parameters:
            return surf.intensity(latv, lonv, theta=jnp.asarray(0.0, dtype=float_dtype()))
        return surf.intensity(latv, lonv)
    except (TypeError, ValueError):
        return surf.intensity(latv, lonv)


# =============================================================================
# Build SWAMP static + RunFlags (once)
# =============================================================================

if cfg.model_days <= 0:
    raise ValueError("cfg.model_days must be > 0")
if cfg.dt_seconds <= 0:
    raise ValueError("cfg.dt_seconds must be > 0")
if cfg.starttime_index < 2:
    raise ValueError("cfg.starttime_index must be >= 2 for leapfrog startup")

n_steps = compute_n_steps(cfg.model_days, cfg.dt_seconds)
logger.info(f"SWAMP integration length: model_days={cfg.model_days} -> n_steps={n_steps} (dt={cfg.dt_seconds}s)")

# Use plain Python floats for compatibility across my_swamp versions
static_kwargs = dict(
    M=cfg.M,
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
    forcflag=cfg.forcflag,
    diffflag=cfg.diffflag,
    expflag=cfg.expflag,
    modalflag=cfg.modalflag,
    diagnostics=cfg.diagnostics,
    alpha=float(cfg.alpha),
    blowup_rms=float(cfg.blowup_rms),
)
flags = call_with_filtered_kwargs(RunFlags, flags_kwargs, name="RunFlags")

I = int(getattr(static_base, "I", -1))
J = int(getattr(static_base, "J", -1))
logger.info(f"SWAMP grid: I={I}, J={J}, M={getattr(static_base,'M','?')}, N={getattr(static_base,'N','?')}")

# Time indices used by SWAMP stepper
t_seq = jnp.arange(cfg.starttime_index, cfg.starttime_index + n_steps, dtype=jnp.int32)


# =============================================================================
# Initial conditions + state init (once)
# =============================================================================

def init_rest_state(static: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Rest initial conditions (matching your toy notebook):
      eta0 = omega * mu (planetary vorticity background)
      delta0 = 0
      U0=V0=0
      Phi0 = Phieq
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
        eta1d = jnp.asarray(omega, dtype=dtype) * jnp.asarray(mus, dtype=dtype)
        eta0 = eta1d[:, None] * jnp.ones((Jloc, Iloc), dtype=dtype)

    delta0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
    U0 = jnp.zeros((Jloc, Iloc), dtype=dtype)
    V0 = jnp.zeros((Jloc, Iloc), dtype=dtype)

    Phieq = getattr(static, "Phieq", 0.0)
    Phi0 = jnp.asarray(Phieq, dtype=dtype)
    return eta0, delta0, U0, V0, Phi0

eta0, delta0, U0, V0, Phi0 = init_rest_state(static_base)

# State initialization: requires forward-optimized my_swamp
init_fn = getattr(swamp_model, "_init_state_from_fields", None) or getattr(swamp_model, "init_state_from_fields", None)
if init_fn is None:
    raise RuntimeError(
        "Could not find my_swamp.model._init_state_from_fields or init_state_from_fields. "
        "This retrieval pipeline requires initializing a State from fields without allocating history."
    )

state0 = call_with_filtered_kwargs(
    init_fn,
    dict(
        static=static_base,
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


# =============================================================================
# Precompute SWAMP pixel grid + weights (once)
# =============================================================================

lambdas = getattr(static_base, "lambdas", None)
mus = getattr(static_base, "mus", None)
w_lat = getattr(static_base, "w", None)

if lambdas is None or mus is None:
    raise RuntimeError("static_base.lambdas and static_base.mus are required to build the starry projector.")

lon = jnp.asarray(lambdas, dtype=float_dtype())               # (I,)
lat = jnp.arcsin(jnp.asarray(mus, dtype=float_dtype()))       # (J,)

lon2d = jnp.broadcast_to(lon[None, :], (lat.shape[0], lon.shape[0]))
lat2d = jnp.broadcast_to(lat[:, None], (lat.shape[0], lon.shape[0]))
lon_flat = lon2d.reshape(-1)
lat_flat = lat2d.reshape(-1)

if w_lat is None:
    logger.warning("static_base.w not found; using uniform weights for LSQ projector.")
    w_pix = jnp.ones_like(lat_flat)
else:
    w_lat = jnp.asarray(w_lat, dtype=float_dtype())           # (J,)
    w_pix = jnp.repeat(w_lat, lon.shape[0])                   # (J*I,)
w_sqrt = jnp.sqrt(w_pix)


# =============================================================================
# Build starry design matrix + LSQ projector (once)
# =============================================================================

lm_list = build_lm_list(cfg.ydeg)
n_coeff = (cfg.ydeg + 1) ** 2
n_pix = int(lat_flat.shape[0])

logger.info(f"Building starry design matrix: n_pix={n_pix}, n_coeff={n_coeff} (ydeg={cfg.ydeg})")

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
B = jax.vmap(_intensity_from_yvec_jit)(eye).T   # (n_pix, n_coeff)
_ = B.block_until_ready()
logger.info(f"Design matrix built in {time.time() - t0_B:.2f} s; shape={tuple(B.shape)}")

# Weighted ridge LSQ projector: y = (Bᵀ W B + λI)^(-1) Bᵀ W
Bw = w_sqrt[:, None] * B
ridge = jnp.asarray(cfg.projector_ridge, dtype=float_dtype())
gram = Bw.T @ Bw + ridge * jnp.eye(n_coeff, dtype=float_dtype())

t0_proj = time.time()
projector = jnp.linalg.solve(gram, Bw.T)  # (n_coeff, n_pix)
_ = projector.block_until_ready()
logger.info(f"Projector built in {time.time() - t0_proj:.2f} s")

def intensity_map_to_y_dense(I_map: jnp.ndarray) -> jnp.ndarray:
    """
    Convert SWAMP-grid intensity map (J,I) to dense starry coefficients.

    Best practices:
      - Normalize by area-weighted mean intensity so the LSQ solve is well-scaled.
      - Enforce y00=1 so cfg.planet_fpfs controls overall amplitude.
    """
    I_flat = I_map.reshape(-1)
    I_mean = (w_pix * I_flat).sum() / w_pix.sum()
    I_rel = I_flat / I_mean

    y = projector @ (w_sqrt * I_rel)
    y = y / y[0]
    return y


# =============================================================================
# Orbit/system setup (once)
# =============================================================================

orbital_period_days = (
    float(cfg.orbital_period_override_days)
    if cfg.orbital_period_override_days is not None
    else orbital_period_days_from_omega(cfg.omega_rad_s)
)
logger.info(f"Orbital/rotation period (days): {orbital_period_days:.6f}")

central = Central(radius=cfg.star_radius_rsun, mass=cfg.star_mass_msun)

# Approx conversion Rjup -> Rsun (fine for a toy demo)
RJUP_TO_RSUN = 0.10045

planet = Body(
    radius=cfg.planet_radius_rjup * RJUP_TO_RSUN,
    period=orbital_period_days,
    time_transit=cfg.time_transit_days,
    impact_param=cfg.impact_param,
)

# Planet-only emission: star amplitude 0, but star still exists for occultation.
star_surface = Surface(amplitude=jnp.asarray(0.0, dtype=float_dtype()), normalize=False)

times_days = np.linspace(
    cfg.time_transit_days,
    cfg.time_transit_days + cfg.n_orbits_observed * orbital_period_days,
    cfg.n_times,
    endpoint=False,
)
times_days_jax = jnp.asarray(times_days, dtype=float_dtype())


# =============================================================================
# SWAMP forward model (terminal Phi)
# =============================================================================

def swamp_terminal_phi(taurad_s: jnp.ndarray, taudrag_s: jnp.ndarray) -> jnp.ndarray:
    """
    Run SWAMP for cfg.model_days and return terminal Phi map (J,I).
    """
    static = safe_replace_static(static_base, taurad_s, taudrag_s, static_kwargs)

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

    # Fallback: fori_loop stepping
    step_fn = getattr(swamp_model, "_step_once_state_only", None)
    if step_fn is None:
        raise RuntimeError("Neither simulate_scan_last nor _step_once_state_only found; cannot run SWAMP forward.")

    def body(i: int, st: Any) -> Any:
        t = t_seq[i]
        return step_fn(st, t, static, flags, None, U0, V0)

    state_f = jax.lax.fori_loop(0, int(t_seq.shape[0]), body, state0)
    return getattr(state_f, "Phi_curr")


# =============================================================================
# Full forward model: (tau_rad, tau_drag) -> phase curve
# =============================================================================

def phase_curve_model(taurad_s: jnp.ndarray, taudrag_s: jnp.ndarray) -> jnp.ndarray:
    """
    Complete forward model returning planet flux vs time (shape: n_times).
    """
    phi = swamp_terminal_phi(taurad_s, taudrag_s)

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

phase_curve_model_jit = jax.jit(phase_curve_model)

logger.info("JIT compiling forward model (first call can be slow)...")
t0_compile = time.time()
_ = phase_curve_model_jit(
    jnp.asarray(tau_hours_to_seconds(cfg.taurad_true_hours), dtype=float_dtype()),
    jnp.asarray(tau_hours_to_seconds(cfg.taudrag_true_hours), dtype=float_dtype()),
).block_until_ready()
logger.info(f"Forward model compiled in {time.time() - t0_compile:.2f} s")


# =============================================================================
# Observations (synthetic by default)
# =============================================================================

obs_path = cfg.out_dir / "observations.npz"

if cfg.generate_synthetic_data or not obs_path.exists():
    taurad_true_s = float(tau_hours_to_seconds(cfg.taurad_true_hours))
    taudrag_true_s = float(tau_hours_to_seconds(cfg.taudrag_true_hours))

    logger.info("Generating synthetic observations from SWAMP+starry truth...")
    flux_true = np.asarray(
        phase_curve_model_jit(
            jnp.asarray(taurad_true_s, dtype=float_dtype()),
            jnp.asarray(taudrag_true_s, dtype=float_dtype()),
        )
    )

    rng = np.random.default_rng(cfg.seed)
    flux_obs = flux_true + rng.normal(0.0, cfg.obs_sigma, size=flux_true.shape)

    save_npz(
        obs_path,
        times_days=times_days,
        flux_true=flux_true,
        flux_obs=flux_obs,
        obs_sigma=float(cfg.obs_sigma),
        taurad_true_hours=float(cfg.taurad_true_hours),
        taudrag_true_hours=float(cfg.taudrag_true_hours),
        orbital_period_days=float(orbital_period_days),
    )
    logger.info(f"Saved observations to: {obs_path}")
else:
    d = np.load(obs_path)
    times_days = d["times_days"]
    times_days_jax = jnp.asarray(times_days, dtype=float_dtype())
    flux_true = d["flux_true"]
    flux_obs = d["flux_obs"]
    logger.info(f"Loaded observations from: {obs_path}")

imin = int(np.argmin(flux_true))
imax = int(np.argmax(flux_true))
logger.info(
    f"Truth flux: min={flux_true[imin]:.3e} at t={times_days[imin]:.5f} d, "
    f"max={flux_true[imax]:.3e} at t={times_days[imax]:.5f} d"
)

flux_obs_jax = jnp.asarray(flux_obs, dtype=float_dtype())
obs_sigma_jax = jnp.asarray(cfg.obs_sigma, dtype=float_dtype())


# =============================================================================
# NumPyro model + NUTS
# =============================================================================

def numpyro_model(_times: jnp.ndarray, y_obs: jnp.ndarray) -> None:
    """
    Model definition for NUTS.

    We sample ONLY the parameters of interest:
      - log10_taurad_hours
      - log10_taudrag_hours

    Sampling in log-space is essential because tau must be positive and spans orders of magnitude.
    """
    log10_taurad_h = numpyro.sample(
        "log10_taurad_hours",
        dist.Uniform(cfg.prior_log10_tau_hours_min, cfg.prior_log10_tau_hours_max),
    )
    log10_taudrag_h = numpyro.sample(
        "log10_taudrag_hours",
        dist.Uniform(cfg.prior_log10_tau_hours_min, cfg.prior_log10_tau_hours_max),
    )

    taurad_h = 10.0**log10_taurad_h
    taudrag_h = 10.0**log10_taudrag_h

    taurad_s = tau_hours_to_seconds(taurad_h)
    taudrag_s = tau_hours_to_seconds(taudrag_h)

    mu = phase_curve_model_jit(taurad_s, taudrag_s)

    numpyro.sample("obs", dist.Normal(mu, obs_sigma_jax), obs=y_obs)

samples_path = cfg.out_dir / "posterior_samples.npz"
extra_path = cfg.out_dir / "mcmc_extra_fields.npz"

if cfg.run_mcmc:
    logger.info("Running NUTS/HMC...")

    # forward_mode_differentiation exists in newer NumPyro only
    try:
        nuts = NUTS(
            numpyro_model,
            target_accept_prob=cfg.target_accept_prob,
            forward_mode_differentiation=cfg.use_forward_mode_ad,
        )
        logger.info(f"NUTS forward_mode_differentiation={cfg.use_forward_mode_ad}")
    except TypeError:
        nuts = NUTS(
            numpyro_model,
            target_accept_prob=cfg.target_accept_prob,
        )
        logger.warning(
            "This NumPyro version does not support forward_mode_differentiation in NUTS. "
            "Falling back to default autodiff mode."
        )

    mcmc = MCMC(
        nuts,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        chain_method=cfg.chain_method,
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(cfg.seed)
    t0 = time.time()
    mcmc.run(rng_key, times_days_jax, flux_obs_jax)
    logger.info(f"MCMC completed in {time.time() - t0:.2f} s")
    mcmc.print_summary()

    # Posterior samples (grouped by chain if available)
    samples = mcmc.get_samples(group_by_chain=True)

    # Always present (since they are sample sites)
    log10_taurad = np.asarray(samples["log10_taurad_hours"])
    log10_taudrag = np.asarray(samples["log10_taudrag_hours"])

    # Derived arrays computed OUTSIDE the model for robustness across NumPyro versions
    taurad_hours = np.power(10.0, log10_taurad)
    taudrag_hours = np.power(10.0, log10_taudrag)

    save_npz(
        samples_path,
        log10_taurad_hours=log10_taurad,
        log10_taudrag_hours=log10_taudrag,
        taurad_hours=taurad_hours,
        taudrag_hours=taudrag_hours,
    )
    logger.info(f"Saved posterior samples to: {samples_path}")

    # Extra fields (may differ by NumPyro version). Save only what exists.
    extra_to_save: Dict[str, Any] = {}
    try:
        extra = mcmc.get_extra_fields(group_by_chain=True)
    except TypeError:
        extra = mcmc.get_extra_fields()

    for k in ("potential_energy", "num_steps", "accept_prob", "diverging", "step_size"):
        if isinstance(extra, dict) and (k in extra):
            extra_to_save[k] = np.asarray(extra[k])

    if extra_to_save:
        save_npz(extra_path, **extra_to_save)
        logger.info(f"Saved MCMC diagnostics to: {extra_path}")
    else:
        logger.warning("No MCMC extra fields available to save; skipping mcmc_extra_fields.npz")

else:
    logger.info("cfg.run_mcmc=False; skipping MCMC. (Plot script requires posterior_samples.npz)")
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
    taurad_samps_s = np.asarray(s["taurad_hours"]).reshape(-1) * 3600.0
    taudrag_samps_s = np.asarray(s["taudrag_hours"]).reshape(-1) * 3600.0

    rng = np.random.default_rng(cfg.seed + 1)
    n_available = taurad_samps_s.shape[0]
    n_take = min(cfg.ppc_draws, n_available)
    take_idx = rng.choice(n_available, size=n_take, replace=False)

    theta_sel = np.stack([taurad_samps_s[take_idx], taudrag_samps_s[take_idx]], axis=-1).astype(np.float64)
    theta_sel_jax = jnp.asarray(theta_sel, dtype=float_dtype())

    @jax.jit
    def _batch_forward(theta_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda th: phase_curve_model_jit(th[0], th[1]))(theta_batch)

    preds: List[np.ndarray] = []
    for i0 in tqdm(range(0, n_take, cfg.ppc_chunk_size), desc="PPC batches"):
        i1 = min(i0 + cfg.ppc_chunk_size, n_take)
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

maps_path = cfg.out_dir / "maps_truth_and_posterior_mean.npz"

def compute_maps_for_tau_hours(taurad_h: float, taudrag_h: float) -> Dict[str, np.ndarray]:
    taurad_s = jnp.asarray(tau_hours_to_seconds(taurad_h), dtype=float_dtype())
    taudrag_s = jnp.asarray(tau_hours_to_seconds(taudrag_h), dtype=float_dtype())

    phi = swamp_terminal_phi(taurad_s, taudrag_s)
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

# Truth
truth_maps = compute_maps_for_tau_hours(cfg.taurad_true_hours, cfg.taudrag_true_hours)

# Posterior summary in LOG SPACE (median in log10 => geometric median in linear space)
s = np.load(samples_path)
log10_taurad_flat = np.asarray(s["log10_taurad_hours"]).reshape(-1)
log10_taudrag_flat = np.asarray(s["log10_taudrag_hours"]).reshape(-1)

taurad_med_h = float(np.power(10.0, np.median(log10_taurad_flat)))
taudrag_med_h = float(np.power(10.0, np.median(log10_taudrag_flat)))

post_maps = compute_maps_for_tau_hours(taurad_med_h, taudrag_med_h)

save_npz(
    maps_path,
    lon=np.asarray(lon),
    lat=np.asarray(lat),
    phi_truth=truth_maps["phi"],
    T_truth=truth_maps["T"],
    I_truth=truth_maps["I"],
    y_truth=truth_maps["y_dense"],
    taurad_true_hours=float(cfg.taurad_true_hours),
    taudrag_true_hours=float(cfg.taudrag_true_hours),
    phi_post=post_maps["phi"],
    T_post=post_maps["T"],
    I_post=post_maps["I"],
    y_post=post_maps["y_dense"],
    taurad_post_median_hours=float(taurad_med_h),
    taudrag_post_median_hours=float(taudrag_med_h),
)
logger.info(f"Saved truth + posterior-median maps to: {maps_path}")

logger.info("DONE.")
logger.info(f"Outputs saved to: {cfg.out_dir}")
