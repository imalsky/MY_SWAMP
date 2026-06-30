#!/usr/bin/env python3
"""make_dashboard.py — one consolidated results figure for a finished retrieval.

Reads the .npz outputs (no JAX, no re-run) and assembles a single
``results_dashboard.png`` with the panels that tell the whole story:
  (a) phase-curve fit (data + truth + posterior median + PPC band)
  (b) joint posterior with truth crosshair + correlation (the degeneracy)
  (c) SMC convergence (tempering schedule + ESS)
  (d,e) 1-D marginals with truth + prior range
  (f) terminal brightness-temperature map (truth)

    python make_dashboard.py [OUT_DIR]   # default ./swamp_jaxoplanet_retrieval_outputs
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RETRIEVAL_ROOT = _SCRIPTS_DIR.parent
_STYLE_FILE = _SCRIPTS_DIR / "science.mplstyle"
if _STYLE_FILE.exists():
    plt.style.use(str(_STYLE_FILE))

# data read from retrieval/data/, figure written to retrieval/plots/
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (_RETRIEVAL_ROOT / "data")
PLOTS_DIR = _RETRIEVAL_ROOT / "plots"


def load(name):
    p = OUT / name
    return np.load(p, allow_pickle=True) if p.exists() else None


def main():
    cfg = json.loads((OUT / "config.json").read_text())
    obs = load("observations.npz")
    samps = load("posterior_samples.npz")
    extra = load("mcmc_extra_fields.npz")
    ppc = load("posterior_predictive_quantiles.npz")
    maps = load("maps_truth_and_posterior_summary.npz")

    names = [str(x) for x in samps["param_names"].tolist()]
    labels = [str(x) for x in samps["param_labels"].tolist()] if "param_labels" in samps.files else names
    S = np.asarray(samps["samples"]).reshape(-1, len(names))
    truth = np.asarray(cfg.get("inferred_param_truth"), float)
    plo = np.asarray(cfg.get("inferred_param_prior_lo"), float)
    phi_ = np.asarray(cfg.get("inferred_param_prior_hi"), float)

    t = np.asarray(obs["times_days"]); ftrue = np.asarray(obs["flux_true"])
    fobs = np.asarray(obs["flux_obs"]); sigma = float(obs["obs_sigma"])

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    fig.suptitle("Differentiable SWAMP -> phase-curve retrieval: injection-recovery "
                 f"(N={int(extra['smc_num_particles']) if extra is not None and 'smc_num_particles' in extra.files else '?'} "
                 f"particles, {cfg.get('model_days')}-day spin-up, "
                 f"{sigma*1e6:.0f} ppm noise, float{'64' if cfg.get('use_x64') else '32'})",
                 fontsize=14, fontweight="bold")

    # (a) phase-curve fit
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, fobs * 1e6, ".", ms=3, color="0.5", label="observed")
    ax.plot(t, ftrue * 1e6, "-", lw=2, color="C1", label="truth")
    if ppc is not None:
        ax.plot(t, np.asarray(ppc["p50"]) * 1e6, "-", lw=1.5, color="C2", label="posterior median")
        ax.fill_between(t, np.asarray(ppc["p05"]) * 1e6, np.asarray(ppc["p95"]) * 1e6, alpha=0.3, color="C2", label="90% PPC")
    ax.set_xlabel("time [days]"); ax.set_ylabel("planet flux [ppm]")
    ax.set_title("(a) thermal phase-curve fit"); ax.legend(fontsize=8)

    # (b) joint posterior: KDE density contours + scatter + truth + correlation
    ax = fig.add_subplot(gs[0, 1])
    if len(names) >= 2:
        x, y = S[:, 0], S[:, 1]
        if gaussian_kde is not None and len(x) > 5 and np.std(x) > 0 and np.std(y) > 0:
            try:
                xg = np.linspace(x.min(), x.max(), 80); yg = np.linspace(y.min(), y.max(), 80)
                XX, YY = np.meshgrid(xg, yg)
                ZZ = gaussian_kde(np.vstack([x, y]))(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                ax.contourf(XX, YY, ZZ, levels=8, cmap="Blues", alpha=0.85)
            except Exception:
                pass
        ax.scatter(x, y, s=6, alpha=0.35, color="0.25")
        ax.axvline(truth[0], color="k", ls="--", lw=1); ax.axhline(truth[1], color="k", ls="--", lw=1)
        ax.plot(truth[0], truth[1], "*", ms=16, color="C3", label="truth")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_title(f"(b) joint posterior  (corr = {r:+.2f})")
        ax.legend(fontsize=8)
    else:
        ax.hist(S[:, 0], bins=30); ax.set_title("(b) posterior")

    # (c) SMC convergence
    ax = fig.add_subplot(gs[0, 2])
    if extra is not None and "smc_betas" in extra.files:
        betas = np.asarray(extra["smc_betas"]).reshape(-1)
        steps = np.arange(len(betas))
        ax.plot(steps, betas, "o-", color="C0", label="beta (temperature)")
        ax.set_xlabel("SMC step"); ax.set_ylabel("beta", color="C0"); ax.set_ylim(-0.02, 1.05)
        ax.set_title("(c) SMC convergence")
        if "smc_ess" in extra.files:
            ess = np.asarray(extra["smc_ess"]).reshape(-1)
            N = int(extra["smc_num_particles"]) if "smc_num_particles" in extra.files else ess.max()
            ax2 = ax.twinx()
            ax2.plot(steps[1:], ess / N, "s-", color="C4", label="ESS / N")
            ax2.set_ylabel("ESS fraction", color="C4"); ax2.set_ylim(0, 1.05)

    # (d,e) 1-D marginals: KDE curve (smooth even for a small swarm) + light hist
    for i in range(min(2, len(names))):
        ax = fig.add_subplot(gs[1, i])
        v = S[:, i]
        ax.hist(v, bins=20, density=True, alpha=0.30, color="C0")
        if gaussian_kde is not None and len(v) > 5 and np.std(v) > 0:
            try:
                xs = np.linspace(v.min(), v.max(), 300)
                ax.plot(xs, gaussian_kde(v)(xs), color="C0", lw=2)
            except Exception:
                pass
        ax.axvline(truth[i], color="C3", lw=2, label=f"truth = {truth[i]:.1f}")
        q = np.percentile(v, [5, 50, 95])
        ax.axvspan(q[0], q[2], alpha=0.12, color="C0")
        ax.axvline(q[1], color="C0", ls="--", lw=1.5, label=f"median = {q[1]:.2f}")
        ax.axvline(plo[i], color="0.6", ls=":", lw=1); ax.axvline(phi_[i], color="0.6", ls=":", lw=1, label="prior range")
        ax.set_xlabel(labels[i]); ax.set_title(f"({'d' if i==0 else 'e'}) {labels[i]} posterior")
        ax.legend(fontsize=8); ax.set_yticks([])

    # (f) terminal brightness-temperature map (truth)
    ax = fig.add_subplot(gs[1, 2])
    if maps is not None and "T_truth" in maps.files:
        lon = np.degrees(np.asarray(maps["lon"])); lat = np.degrees(np.asarray(maps["lat"]))
        T = np.asarray(maps["T_truth"])
        im = ax.pcolormesh(lon, lat, T, shading="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="T [K]")
        ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
        ax.set_title("(f) terminal brightness-T map (truth)")
    else:
        ax.set_title("(f) maps unavailable")

    path = PLOTS_DIR / "results_dashboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[wrote {path}]")


if __name__ == "__main__":
    main()
