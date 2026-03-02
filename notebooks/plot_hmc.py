#!/usr/bin/env python3
"""
plot_exposed.py

Plotting utilities for the SWAMP + jaxoplanet/starry NUTS retrieval.

Reads outputs from `run_exposed.py` (or the fixed/older run script) in OUT_DIR and writes:
  - plots/phase_curve_fit.png
  - plots/phase_curve_residuals.png
  - plots/posterior_corner.png         (all inferred parameters, if available)
  - plots/posterior_tau_corner.png     (tau-only, always)
  - plots/nuts_diagnostics.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# =============================================================================
# Config
# =============================================================================

OUT_DIR = Path("swamp_jaxoplanet_retrieval_outputs")
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger("swamp_plot")


# =============================================================================
# Helpers
# =============================================================================


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_npz_optional(path: Path) -> Optional[np.lib.npyio.NpzFile]:
    if path.exists():
        return np.load(path)
    return None


def _save_fig(fig: plt.Figure, name: str) -> None:
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out.resolve()}")


CHAIN_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def _flatten_chains(samples: np.ndarray) -> np.ndarray:
    """Flatten (chains, draws, ...) -> (chains*draws, ...)."""
    if samples.ndim < 2:
        raise ValueError(f"Expected at least 2D samples (chains, draws, ...), got shape {samples.shape}")
    return samples.reshape(-1, *samples.shape[2:])


def _truth_value_for_param(name: str, cfg: Dict[str, Any]) -> Optional[float]:
    # Mapping from posterior_samples param name -> config key
    key_map = {
        "taurad_hours": "taurad_true_hours",
        "taudrag_hours": "taudrag_true_hours",
        "a_planet_m": "a_planet_m",
        "omega_rad_s": "omega_rad_s",
        "g_m_s2": "g_m_s2",
        "Phibar": "Phibar",
        "DPhieq": "DPhieq",
        "K6": "K6",
        "K6Phi": "K6Phi",
        "alpha": "alpha",
    }
    key = key_map.get(name, name)
    if key not in cfg:
        return None
    val = cfg[key]
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


# =============================================================================
# Phase curve plots
# =============================================================================


def plot_phase_curve_fit(
    *,
    times_days: np.ndarray,
    flux_obs: np.ndarray,
    flux_true: np.ndarray,
    obs_sigma: float,
    orbital_period_days: float,
    cfg: Dict[str, Any],
    ppc_q: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)

    phase = ((times_days - float(cfg.get("time_transit_days", 0.0))) / orbital_period_days) % 1.0

    ax.errorbar(
        phase,
        flux_obs,
        yerr=obs_sigma,
        fmt=".",
        ms=4,
        lw=0.7,
        alpha=0.7,
        label="observed",
    )
    ax.plot(phase, flux_true, lw=1.5, label="truth")

    if ppc_q is not None:
        ax.plot(phase, ppc_q["p50"], lw=1.5, label="posterior median")
        ax.fill_between(phase, ppc_q["p05"], ppc_q["p95"], alpha=0.25, label="90% PPC band")

    ax.set_xlabel("orbital phase")
    ax.set_ylabel("planet flux (relative)")
    ax.set_title("Phase curve fit")
    ax.legend(loc="best", frameon=False)

    _save_fig(fig, "phase_curve_fit.png")


def plot_phase_curve_residuals(
    *,
    times_days: np.ndarray,
    flux_obs: np.ndarray,
    flux_true: np.ndarray,
    obs_sigma: float,
    orbital_period_days: float,
    cfg: Dict[str, Any],
    ppc_q: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)

    phase = ((times_days - float(cfg.get("time_transit_days", 0.0))) / orbital_period_days) % 1.0

    resid_true = (flux_obs - flux_true) / obs_sigma
    ax.plot(phase, resid_true, ".", ms=4, alpha=0.7, label="(obs - truth)/sigma")

    if ppc_q is not None:
        resid_med = (flux_obs - ppc_q["p50"]) / obs_sigma
        ax.plot(phase, resid_med, ".", ms=4, alpha=0.7, label="(obs - post median)/sigma")
        ax.fill_between(
            phase,
            (flux_obs - ppc_q["p95"]) / obs_sigma,
            (flux_obs - ppc_q["p05"]) / obs_sigma,
            alpha=0.25,
            label="90% PPC band (resid)",
        )

    ax.axhline(0.0, lw=1.0, alpha=0.8)
    ax.set_xlabel("orbital phase")
    ax.set_ylabel("residual / sigma")
    ax.set_title("Residuals")
    ax.legend(loc="best", frameon=False)

    _save_fig(fig, "phase_curve_residuals.png")


# =============================================================================
# Posterior corner plots
# =============================================================================


def plot_posterior_tau_corner(
    *,
    taurad_hours: np.ndarray,
    taudrag_hours: np.ndarray,
    cfg: Dict[str, Any],
) -> None:
    taurad = _flatten_chains(taurad_hours)
    taudrag = _flatten_chains(taudrag_hours)

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    # 1D hists
    axs[0, 0].hist(taurad, bins=40, density=True)
    axs[1, 1].hist(taudrag, bins=40, density=True)

    # Truth lines
    tr = float(cfg.get("taurad_true_hours", np.median(taurad)))
    td = float(cfg.get("taudrag_true_hours", np.median(taudrag)))
    axs[0, 0].axvline(tr, lw=1.5)
    axs[1, 1].axvline(td, lw=1.5)

    # 2D
    axs[1, 0].hist2d(taurad, taudrag, bins=40)
    axs[1, 0].plot([tr], [td], "x", ms=8)

    axs[0, 1].axis("off")

    axs[0, 0].set_xlabel(r"$\tau_{\rm rad}$ [hours]")
    axs[1, 1].set_xlabel(r"$\tau_{\rm drag}$ [hours]")
    axs[1, 0].set_xlabel(r"$\tau_{\rm rad}$ [hours]")
    axs[1, 0].set_ylabel(r"$\tau_{\rm drag}$ [hours]")

    title = "\n".join(
        [
            "Posterior (tau-only)",
            f"truth: taurad={tr:.3g} h, taudrag={td:.3g} h",
            f"posterior medians: taurad={np.median(taurad):.3g} h, taudrag={np.median(taudrag):.3g} h",
        ]
    )
    fig.suptitle(title)
    fig.tight_layout()

    _save_fig(fig, "posterior_tau_corner.png")


def plot_posterior_param_corner(
    *,
    param_samples: np.ndarray,      # (chains, draws, dim)
    param_names: Sequence[str],
    param_labels: Optional[Sequence[str]],
    cfg: Dict[str, Any],
) -> None:
    if param_samples.ndim != 3:
        raise ValueError(f"param_samples must be (chains, draws, dim), got shape {param_samples.shape}")
    dim = int(param_samples.shape[-1])
    if dim < 1:
        raise ValueError("dim must be >= 1")

    flat = param_samples.reshape(-1, dim)

    labels = list(param_labels) if param_labels is not None else list(param_names)
    if len(labels) != dim:
        labels = list(param_names)

    # Decide per-parameter axis scaling
    is_pos = np.all(flat > 0.0, axis=0)

    fig, axs = plt.subplots(dim, dim, figsize=(2.4 * dim, 2.4 * dim))

    # Handle dim==1 case: axs is a single Axes
    if dim == 1:
        axs = np.asarray([[axs]])

    for i in range(dim):
        xi = flat[:, i]
        truth_i = _truth_value_for_param(param_names[i], cfg)

        # Choose bounds from robust quantiles
        qlo, qhi = np.quantile(xi, [0.001, 0.999])
        if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
            qlo, qhi = np.min(xi), np.max(xi)
        if is_pos[i]:
            qlo = max(qlo, np.min(xi[xi > 0]))
            qhi = max(qhi, qlo * 1.01)
            bins_1d = np.logspace(np.log10(qlo), np.log10(qhi), 45)
        else:
            bins_1d = np.linspace(qlo, qhi, 45)

        for j in range(dim):
            ax = axs[i, j]
            if i < j:
                ax.axis("off")
                continue

            xj = flat[:, j] if j != i else None
            truth_j = _truth_value_for_param(param_names[j], cfg) if j != i else None

            if i == j:
                ax.hist(xi, bins=bins_1d, density=True)
                if truth_i is not None and np.isfinite(truth_i):
                    ax.axvline(truth_i, lw=1.3)
                if is_pos[i]:
                    ax.set_xscale("log")
                ax.set_yticks([])
            else:
                # 2D hist
                xjv = flat[:, j]
                # Robust bounds
                qlo_j, qhi_j = np.quantile(xjv, [0.001, 0.999])
                if is_pos[j]:
                    qlo_j = max(qlo_j, np.min(xjv[xjv > 0]))
                    qhi_j = max(qhi_j, qlo_j * 1.01)
                    bins_j = np.logspace(np.log10(qlo_j), np.log10(qhi_j), 45)
                    ax.set_xscale("log")
                else:
                    bins_j = np.linspace(qlo_j, qhi_j, 45)

                if is_pos[i]:
                    ax.set_yscale("log")
                    bins_i = bins_1d
                else:
                    bins_i = bins_1d

                ax.hist2d(xjv, xi, bins=[bins_j, bins_i])

                if truth_i is not None and truth_j is not None and np.isfinite(truth_i) and np.isfinite(truth_j):
                    ax.plot([truth_j], [truth_i], "x", ms=6)

            if i == dim - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j != 0:
                ax.set_yticklabels([])

    fig.suptitle("Posterior (all inferred parameters)")
    fig.tight_layout()
    _save_fig(fig, "posterior_corner.png")


# =============================================================================
# NUTS diagnostics plot
# =============================================================================


def plot_nuts_diagnostics(
    *,
    diag: Optional[np.lib.npyio.NpzFile],
    samples: np.lib.npyio.NpzFile,
    cfg: Dict[str, Any],
) -> None:
    taurad = np.asarray(samples["taurad_hours"])
    taudrag = np.asarray(samples["taudrag_hours"])
    num_chains = taurad.shape[0]
    num_samples = taurad.shape[1]

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    for c in range(num_chains):
        ax1.plot(taurad[c], lw=1.0, alpha=0.8, color=CHAIN_COLORS[c % len(CHAIN_COLORS)])
    ax1.set_ylabel(r"$\tau_{\rm rad}$ [hours]")
    ax1.set_title("Trace plots (tau only)")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for c in range(num_chains):
        ax2.plot(taudrag[c], lw=1.0, alpha=0.8, color=CHAIN_COLORS[c % len(CHAIN_COLORS)])
    ax2.set_ylabel(r"$\tau_{\rm drag}$ [hours]")

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    if diag is not None:
        # Prefer num_integration_steps
        depth_key = None
        if "num_integration_steps" in diag.files:
            depth_key = "num_integration_steps"
        elif "tree_depth" in diag.files:
            depth_key = "tree_depth"

        if depth_key is not None:
            depth = np.asarray(diag[depth_key])
            if depth.ndim == 1:
                depth = depth[np.newaxis, :]
            for c in range(min(depth.shape[0], num_chains)):
                window = max(1, num_samples // 50)
                depth_smooth = np.convolve(depth[c].astype(float), np.ones(window) / window, mode="valid")
                ax3.plot(
                    np.arange(len(depth_smooth)),
                    depth_smooth,
                    lw=1.0,
                    alpha=0.8,
                    color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                )
            ax3.set_ylabel(depth_key)
            ax3.set_title("Trajectory length / tree depth")

            div_key = "is_divergent" if "is_divergent" in diag.files else ("divergences" if "divergences" in diag.files else None)
            if div_key is not None:
                divs = np.asarray(diag[div_key])
                if divs.ndim == 1:
                    divs = divs[np.newaxis, :]
                n_div = int(np.sum(divs))
                if n_div > 0:
                    for c in range(min(divs.shape[0], num_chains)):
                        div_idx = np.where(divs[c])[0]
                        if len(div_idx) > 0:
                            ax3.scatter(
                                div_idx,
                                depth[c][div_idx],
                                marker="|",
                                s=20,
                                color="red",
                                alpha=0.6,
                                zorder=5,
                            )
                    ax3.set_title(f"Trajectory length — {n_div} divergences (red)")
        else:
            ax3.text(0.5, 0.5, "Tree depth / integration steps not available.", ha="center", va="center", transform=ax3.transAxes)
            ax3.axis("off")
    else:
        ax3.text(0.5, 0.5, "No diagnostics file found.", ha="center", va="center", transform=ax3.transAxes)
        ax3.axis("off")

    ax3.set_xlabel("sample index")
    _save_fig(fig, "nuts_diagnostics.png")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logger.info(f"OUT_DIR={OUT_DIR.resolve()}")
    logger.info("Generating plots: phase curve fit, residuals, posterior corner, tau corner, NUTS diagnostics.")

    cfg = _load_json(OUT_DIR / "config_hmc.json")

    obs = np.load(OUT_DIR / "observations_hmc.npz")
    times_days = np.asarray(obs["times_days"])
    flux_true = np.asarray(obs["flux_true"])
    flux_obs = np.asarray(obs["flux_obs"])
    obs_sigma = float(obs["obs_sigma"])
    orbital_period_days = float(obs["orbital_period_days"])

    # Posterior samples
    s = np.load(OUT_DIR / "posterior_samples_hmc.npz")
    taurad_hours = np.asarray(s["taurad_hours"])
    taudrag_hours = np.asarray(s["taudrag_hours"])

    # Parameter corner (preferred)
    if "param_samples" in s.files and "param_names" in s.files:
        param_samples = np.asarray(s["param_samples"])
        param_names = [str(x) for x in np.asarray(s["param_names"]).tolist()]
        param_labels = None
        if "param_labels" in s.files:
            param_labels = [str(x) for x in np.asarray(s["param_labels"]).tolist()]
        plot_posterior_param_corner(
            param_samples=param_samples,
            param_names=param_names,
            param_labels=param_labels,
            cfg=cfg,
        )
    else:
        logger.warning("posterior_samples_hmc.npz does not contain param_samples/param_names; skipping full corner plot.")

    # PPC quantiles (optional)
    ppc_q = None
    q = _load_npz_optional(OUT_DIR / "posterior_predictive_quantiles_hmc.npz")
    if q is not None and all(k in q.files for k in ("p05", "p50", "p95")):
        ppc_q = {"p05": np.asarray(q["p05"]), "p50": np.asarray(q["p50"]), "p95": np.asarray(q["p95"])}

    # HMC diagnostics (optional)
    diag = _load_npz_optional(OUT_DIR / "diagnostics_hmc.npz")

    plot_phase_curve_fit(
        times_days=times_days,
        flux_obs=flux_obs,
        flux_true=flux_true,
        obs_sigma=obs_sigma,
        orbital_period_days=orbital_period_days,
        cfg=cfg,
        ppc_q=ppc_q,
    )
    plot_phase_curve_residuals(
        times_days=times_days,
        flux_obs=flux_obs,
        flux_true=flux_true,
        obs_sigma=obs_sigma,
        orbital_period_days=orbital_period_days,
        cfg=cfg,
        ppc_q=ppc_q,
    )
    plot_posterior_tau_corner(
        taurad_hours=taurad_hours,
        taudrag_hours=taudrag_hours,
        cfg=cfg,
    )
    plot_nuts_diagnostics(
        diag=diag,
        samples=s,
        cfg=cfg,
    )

    logger.info(f"DONE. Plots saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
