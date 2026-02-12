#!/usr/bin/env python3
"""
swamp_plot.py

Plot all results from a completed `swamp_run.py` run.

This script NEVER runs SWAMP and NEVER runs MCMC. It only reads saved outputs from
OUT_DIR and creates plots under OUT_DIR/plots.

Key robustness features
-----------------------
- Handles missing optional files (PPC quantiles, MCMC diagnostics, maps) gracefully.
- Handles NaNs in diagnostics arrays gracefully (e.g., accept_prob missing in older NumPyro).
- Plots both linear and log-space posteriors (since sampling is in log-space).

No CLI args by design: edit OUT_DIR below if needed.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional style: safe fallback if SciencePlots isn't installed
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

plt.style.use('science.mplstyle')


# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = Path("swamp_jaxoplanet_retrieval_outputs")  # <-- edit if your run used a different directory
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOGGING
# =============================================================================

log_path = OUT_DIR / "plot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    force=True,
)
logger = logging.getLogger("swamp_plot")
logger.info(f"Plotting from OUT_DIR={OUT_DIR.resolve()}")


# =============================================================================
# LOAD FILES
# =============================================================================

cfg_path = OUT_DIR / "config.json"
if not cfg_path.exists():
    raise FileNotFoundError(f"Missing config.json at: {cfg_path}")
cfg: Dict[str, Any] = json.loads(cfg_path.read_text())
logger.info("Loaded config.json")

# Apply DPI if present
if "fig_dpi" in cfg:
    plt.rcParams["figure.dpi"] = int(cfg["fig_dpi"])

obs_path = OUT_DIR / "observations.npz"
if not obs_path.exists():
    raise FileNotFoundError(f"Missing observations.npz at: {obs_path}")
obs = np.load(obs_path)
times_days = obs["times_days"]
flux_true = obs["flux_true"]
flux_obs = obs["flux_obs"]
obs_sigma = float(obs["obs_sigma"])
orbital_period_days = float(obs["orbital_period_days"])
logger.info("Loaded observations.npz")

samples_path = OUT_DIR / "posterior_samples.npz"
if not samples_path.exists():
    raise FileNotFoundError(f"Missing posterior_samples.npz at: {samples_path}")
samples = np.load(samples_path)
log10_taurad_hours = samples["log10_taurad_hours"]  # shape: (chains, draws)
log10_taudrag_hours = samples["log10_taudrag_hours"]
taurad_hours = samples["taurad_hours"]
taudrag_hours = samples["taudrag_hours"]
logger.info(
    f"Loaded posterior_samples.npz: "
    f"log10_taurad={log10_taurad_hours.shape}, log10_taudrag={log10_taudrag_hours.shape}"
)

extra_path = OUT_DIR / "mcmc_extra_fields.npz"
extra = None
if extra_path.exists():
    extra = np.load(extra_path)
    logger.info("Loaded mcmc_extra_fields.npz (optional)")

ppc_quant_path = OUT_DIR / "posterior_predictive_quantiles.npz"
ppc_q = None
if ppc_quant_path.exists():
    q = np.load(ppc_quant_path)
    ppc_q = {"p05": q["p05"], "p50": q["p50"], "p95": q["p95"]}
    logger.info("Loaded posterior_predictive_quantiles.npz (optional)")

maps_path = OUT_DIR / "maps_truth_and_posterior_mean.npz"
maps = None
if maps_path.exists():
    maps = np.load(maps_path)
    logger.info("Loaded maps_truth_and_posterior_mean.npz (optional)")


# =============================================================================
# HELPERS
# =============================================================================

def flatten_chain_draw(x: np.ndarray) -> np.ndarray:
    """(chains, draws) -> (chains*draws,)"""
    return x.reshape(-1)

def save_fig(fig, filename: str):
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")

def _finite(x: np.ndarray) -> np.ndarray:
    """Return finite entries of an array (flattened)."""
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]

def _pcolormesh_map(ax, lon_rad: np.ndarray, lat_rad: np.ndarray, Z: np.ndarray, title: str):
    """
    Plot a SWAMP-grid (lat,lon) map using pcolormesh, converting to degrees.
    """
    lon = lon_rad
    lat = lat_rad

    lon_edges = np.zeros(lon.size + 1)
    lon_edges[1:-1] = 0.5 * (lon[:-1] + lon[1:])
    lon_edges[0] = lon[0] - 0.5 * (lon[1] - lon[0])
    lon_edges[-1] = lon[-1] + 0.5 * (lon[-1] - lon[-2])

    lat_edges = np.zeros(lat.size + 1)
    lat_edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    lat_edges[0] = -0.5 * math.pi
    lat_edges[-1] = 0.5 * math.pi

    LonE, LatE = np.meshgrid(lon_edges, lat_edges)
    pcm = ax.pcolormesh(np.degrees(LonE), np.degrees(LatE), Z, shading="auto")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    fig = ax.get_figure()
    fig.colorbar(pcm, ax=ax, shrink=0.85)


# =============================================================================
# PLOTS
# =============================================================================

def plot_phase_curve():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times_days, flux_obs, ".", ms=3, label="observed", alpha=0.65)
    ax.plot(times_days, flux_true, "-", lw=2, label="truth (noise-free)")

    if ppc_q is not None:
        ax.plot(times_days, ppc_q["p50"], "-", lw=2, label="posterior median")
        ax.fill_between(times_days, ppc_q["p05"], ppc_q["p95"], alpha=0.25, label="90% PPC band")

    # Mark transit and secondary eclipse
    t0 = float(cfg.get("time_transit_days", 0.0))
    ax.axvline(t0, ls="--", lw=1, alpha=0.6)
    ax.axvline(t0 + 0.5 * orbital_period_days, ls="--", lw=1, alpha=0.6)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Planet flux (relative; scaled by planet_fpfs)")
    ax.set_title("Thermal phase curve (SWAMP + starry)")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "phase_curve.png")

def plot_phase_curve_residuals():
    model = ppc_q["p50"] if ppc_q is not None else flux_true
    resid = flux_obs - model

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(times_days, resid, ".", ms=3, alpha=0.7)
    ax.axhline(0.0, lw=1)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals (obs - model)")
    save_fig(fig, "phase_curve_residuals.png")

def plot_posteriors_linear_and_log():
    # Flatten
    taurad = flatten_chain_draw(taurad_hours)
    taudrag = flatten_chain_draw(taudrag_hours)
    log10_taurad = flatten_chain_draw(log10_taurad_hours)
    log10_taudrag = flatten_chain_draw(log10_taudrag_hours)

    # Linear-space histograms (hours)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(taurad, bins=40, alpha=0.75)
    if "taurad_true_hours" in cfg:
        ax.axvline(float(cfg["taurad_true_hours"]), lw=2, alpha=0.9)
    ax.set_xlabel("tau_rad [hours]")
    ax.set_ylabel("count")
    ax.set_title("Posterior: tau_rad (linear space)")
    save_fig(fig, "posterior_tau_rad_linear.png")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(taudrag, bins=40, alpha=0.75)
    if "taudrag_true_hours" in cfg:
        ax2.axvline(float(cfg["taudrag_true_hours"]), lw=2, alpha=0.9)
    ax2.set_xlabel("tau_drag [hours]")
    ax2.set_ylabel("count")
    ax2.set_title("Posterior: tau_drag (linear space)")
    save_fig(fig2, "posterior_tau_drag_linear.png")

    # Log-space histograms (the actual sampled parameters)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(log10_taurad, bins=40, alpha=0.75)
    if "taurad_true_hours" in cfg:
        ax3.axvline(math.log10(float(cfg["taurad_true_hours"])), lw=2, alpha=0.9)
    ax3.set_xlabel("log10(tau_rad / hours)")
    ax3.set_ylabel("count")
    ax3.set_title("Posterior: log10(tau_rad)  (sampled)")
    save_fig(fig3, "posterior_log10_tau_rad.png")

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.hist(log10_taudrag, bins=40, alpha=0.75)
    if "taudrag_true_hours" in cfg:
        ax4.axvline(math.log10(float(cfg["taudrag_true_hours"])), lw=2, alpha=0.9)
    ax4.set_xlabel("log10(tau_drag / hours)")
    ax4.set_ylabel("count")
    ax4.set_title("Posterior: log10(tau_drag)  (sampled)")
    save_fig(fig4, "posterior_log10_tau_drag.png")

def plot_traces():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for c in range(log10_taurad_hours.shape[0]):
        ax.plot(log10_taurad_hours[c], alpha=0.75)
    ax.set_xlabel("draw")
    ax.set_ylabel("log10(tau_rad / hours)")
    ax.set_title("Trace: log10(tau_rad)")
    save_fig(fig, "trace_log10_tau_rad.png")

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    for c in range(log10_taudrag_hours.shape[0]):
        ax2.plot(log10_taudrag_hours[c], alpha=0.75)
    ax2.set_xlabel("draw")
    ax2.set_ylabel("log10(tau_drag / hours)")
    ax2.set_title("Trace: log10(tau_drag)")
    save_fig(fig2, "trace_log10_tau_drag.png")

def plot_pair():
    taurad = flatten_chain_draw(log10_taurad_hours)
    taudrag = flatten_chain_draw(log10_taudrag_hours)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(taurad, taudrag, ".", ms=2, alpha=0.3)
    ax.set_xlabel("log10(tau_rad / hours)")
    ax.set_ylabel("log10(tau_drag / hours)")
    ax.set_title("Posterior samples (log space)")
    save_fig(fig, "pair_logspace.png")

def plot_accept_prob():
    """
    Plot accept_prob histogram if available and finite.

    This fixes the crash you saw:
      ValueError: autodetected range of [nan, nan] is not finite

    That error happens when accept_prob is missing or saved as all-NaN.
    """
    if extra is None:
        logger.info("No mcmc_extra_fields.npz; skipping accept_prob plot.")
        return
    if "accept_prob" not in extra.files:
        logger.info("accept_prob not present in mcmc_extra_fields.npz; skipping.")
        return

    acc = _finite(extra["accept_prob"])
    if acc.size == 0:
        logger.info("accept_prob is all-NaN or empty; skipping accept_prob plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(acc, bins=40, alpha=0.8)
    ax.set_xlabel("accept_prob")
    ax.set_ylabel("count")
    ax.set_title("MCMC accept_prob")
    save_fig(fig, "mcmc_accept_prob.png")

def plot_maps():
    if maps is None:
        logger.info("No maps file; skipping maps.png")
        return

    lon = maps["lon"]
    lat = maps["lat"]

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    _pcolormesh_map(axs[0, 0], lon, lat, maps["phi_truth"], "Phi truth")
    _pcolormesh_map(axs[0, 1], lon, lat, maps["T_truth"], "T truth [K]")
    _pcolormesh_map(axs[0, 2], lon, lat, maps["I_truth"], "I truth ∝ T^4")
    _pcolormesh_map(axs[1, 0], lon, lat, maps["phi_post"], "Phi posterior summary")
    _pcolormesh_map(axs[1, 1], lon, lat, maps["T_post"], "T posterior summary [K]")
    _pcolormesh_map(axs[1, 2], lon, lat, maps["I_post"], "I posterior summary ∝ T^4")
    fig.suptitle("Terminal SWAMP maps and intensity proxy")
    path = PLOTS_DIR / "maps.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")

def plot_disk_renders():
    """
    Render visible disk images from saved Ylm coefficients (truth and posterior summary).
    Requires jax + jaxoplanet/starry; otherwise skipped.
    """
    if maps is None:
        logger.info("No maps file; skipping disk renders.")
        return

    try:
        import jax
        import jax.numpy as jnp
        from jaxoplanet.starry.surface import Surface
        from jaxoplanet.starry.ylm import Ylm
    except Exception as e:
        logger.info(f"jax/jaxoplanet not importable; skipping disk renders. Error: {e}")
        return

    ydeg = int(cfg.get("ydeg", 10))
    inc = float(cfg.get("map_inc_rad", math.pi / 2))
    obl = float(cfg.get("map_obl_rad", 0.0))
    phase0 = float(cfg.get("phase_at_transit_rad", math.pi))
    time_transit = float(cfg.get("time_transit_days", 0.0))
    render_res = int(cfg.get("render_res", 250))
    render_phases = cfg.get("render_phases", [0.0, 0.25, 0.49, 0.51, 0.75])
    render_phases = [float(x) for x in render_phases]

    # Dense ordering: l=0..ydeg, m=-l..l
    lm_list: List[Tuple[int, int]] = [(ell, m) for ell in range(ydeg + 1) for m in range(-ell, ell + 1)]

    def ylm_from_dense(y_dense: np.ndarray) -> Ylm:
        y = jnp.asarray(y_dense)
        data = {lm: y[i] for i, lm in enumerate(lm_list)}
        return Ylm(data)

    def make_surface(y_dense: np.ndarray) -> Surface:
        return Surface(
            y=ylm_from_dense(y_dense),
            u=(),
            inc=jnp.asarray(inc),
            obl=jnp.asarray(obl),
            period=jnp.asarray(orbital_period_days),
            phase=jnp.asarray(phase0),
            amplitude=jnp.asarray(1.0),
            normalize=False,
        )

    def rotational_phase(surface: Surface, t: float) -> Any:
        if hasattr(surface, "rotational_phase"):
            return surface.rotational_phase(jnp.asarray(t))
        return jnp.asarray(2.0 * math.pi * (t - time_transit) / orbital_period_days + phase0)

    def safe_render(surface: Surface, theta: Any, res: int) -> np.ndarray:
        try:
            sig = inspect.signature(surface.render)
            if "theta" in sig.parameters:
                img = surface.render(theta=theta, res=res)
            elif "phase" in sig.parameters:
                img = surface.render(phase=theta, res=res)
            else:
                img = surface.render(res=res)
        except (TypeError, ValueError):
            img = surface.render(theta=theta, res=res)
        return np.asarray(img)

    def render_grid(y_dense: np.ndarray, label: str, filename: str):
        surface = make_surface(y_dense)
        fig, axs = plt.subplots(1, len(render_phases), figsize=(3.2 * len(render_phases), 3.0), constrained_layout=True)
        if len(render_phases) == 1:
            axs = [axs]
        for ax, ph in zip(axs, render_phases):
            t = time_transit + ph * orbital_period_days
            theta = rotational_phase(surface, t)
            img = safe_render(surface, theta, render_res)
            ax.imshow(img, origin="lower")
            ax.set_title(f"{label}\nphase={ph:.2f}")
            ax.axis("off")
        path = PLOTS_DIR / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved {path}")

    render_grid(maps["y_truth"], "Truth", "disk_renders_truth.png")
    render_grid(maps["y_post"], "Posterior summary", "disk_renders_posterior.png")


# =============================================================================
# RUN
# =============================================================================

logger.info("Generating plots...")
plot_phase_curve()
plot_phase_curve_residuals()
plot_posteriors_linear_and_log()
plot_traces()
plot_pair()
plot_accept_prob()
plot_maps()
plot_disk_renders()
logger.info(f"DONE. Plots saved to {PLOTS_DIR.resolve()}")
