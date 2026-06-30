#!/usr/bin/env python3
"""summarize_run.py — concise, human-readable summary of a finished retrieval.

Reads the .npz bundles in an output dir and prints (and writes RETRIEVAL_SUMMARY.md):
injected truth vs recovered posterior (median + 68/95% CI + truth-in-CI),
parameter correlations (the tau_rad-tau_drag degeneracy), SMC convergence
(tempering steps, final ESS, acceptance), and the data SNR. No JAX, no re-run.

    python summarize_run.py [OUT_DIR]   # default ./swamp_jaxoplanet_retrieval_outputs
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path(__file__).resolve().parent.parent / "data")


def fmt(x):
    return f"{x:.3g}"


def main():
    cfg = json.loads((OUT / "config.json").read_text())
    obs = np.load(OUT / "observations.npz", allow_pickle=True)
    samps = np.load(OUT / "posterior_samples.npz", allow_pickle=True)
    extra = np.load(OUT / "mcmc_extra_fields.npz", allow_pickle=True) if (OUT / "mcmc_extra_fields.npz").exists() else None

    names = [str(x) for x in samps["param_names"].tolist()]
    labels = [str(x) for x in samps["param_labels"].tolist()] if "param_labels" in samps.files else names
    samples = np.asarray(samps["samples"]).reshape(-1, len(names))
    truth = np.asarray(cfg.get("inferred_param_truth", obs["inferred_param_truth"]), float)
    prior_types = cfg.get("inferred_param_prior_types", ["?"] * len(names))
    prior_lo = np.asarray(cfg.get("inferred_param_prior_lo", [np.nan] * len(names)), float)
    prior_hi = np.asarray(cfg.get("inferred_param_prior_hi", [np.nan] * len(names)), float)

    flux_true = np.asarray(obs["flux_true"]); sigma = float(obs["obs_sigma"])
    amp = float(np.ptp(flux_true))

    L = []
    L.append(f"# Retrieval summary — {OUT.name}\n")
    L.append(f"- preset/model: M={cfg.get('M')} dt={cfg.get('dt_seconds')}s model_days={cfg.get('model_days')} "
             f"(n_steps≈{int(round(cfg.get('model_days',0)*86400/cfg.get('dt_seconds',1)))}), "
             f"use_x64={cfg.get('use_x64')}, emission={cfg.get('emission_model')}")
    L.append(f"- data: n_times={cfg.get('n_times')}, phase amplitude={amp*1e6:.0f} ppm, "
             f"obs_sigma={sigma*1e6:.0f} ppm, amplitude/noise={amp/sigma:.1f}")
    if extra is not None and "smc_betas" in extra.files:
        betas = np.asarray(extra["smc_betas"]).reshape(-1)
        ess = np.asarray(extra["smc_ess"]).reshape(-1) if "smc_ess" in extra.files else None
        acc = np.asarray(extra["smc_acceptance_rate"]).reshape(-1) if "smc_acceptance_rate" in extra.files else None
        N = int(extra["smc_num_particles"]) if "smc_num_particles" in extra.files else 0
        L.append(f"- SMC: kernel={cfg.get('smc_mcmc_kernel')}, N={N}, tempering_steps={len(betas)-1}, "
                 f"final_beta={betas[-1]:.4f}"
                 + (f", final_ESS={ess[-1]:.1f}/{N}" if ess is not None and len(ess) else "")
                 + (f", mean_accept={np.nanmean(acc):.2f}" if acc is not None and len(acc) else ""))

    L.append("\n## Recovery (injected truth vs posterior)\n")
    L.append("| param | truth | median | 68% CI | 95% CI | truth in 95%? |")
    L.append("|---|---|---|---|---|---|")
    for i, nm in enumerate(labels):
        s = samples[:, i]
        q = np.percentile(s, [2.5, 16, 50, 84, 97.5])
        in95 = q[0] <= truth[i] <= q[4]
        L.append(f"| {nm} | {fmt(truth[i])} | {fmt(q[2])} | [{fmt(q[1])},{fmt(q[3])}] | "
                 f"[{fmt(q[0])},{fmt(q[4])}] | {'YES' if in95 else 'NO'} |")

    if len(names) >= 2:
        C = np.corrcoef(samples.T)
        L.append("\n## Posterior correlations (degeneracy structure)\n")
        L.append("| | " + " | ".join(labels) + " |")
        L.append("|" + "---|" * (len(labels) + 1))
        for i, nm in enumerate(labels):
            L.append(f"| {nm} | " + " | ".join(f"{C[i,j]:+.2f}" for j in range(len(labels))) + " |")

    L.append("\n## Notes")
    L.append("- `tau_rad` is set by the day-night amplitude (strong); `tau_drag` by the hot-spot "
             "offset (weak) and is expected to be broader / correlated with `tau_rad`.")
    L.append(f"- priors: " + ", ".join(f"{labels[i]}~{prior_types[i]}[{fmt(prior_lo[i])},{fmt(prior_hi[i])}]"
                                        for i in range(len(names))))

    text = "\n".join(L) + "\n"
    (OUT / "RETRIEVAL_SUMMARY.md").write_text(text)
    print(text)
    print(f"[wrote {OUT/'RETRIEVAL_SUMMARY.md'}]")


if __name__ == "__main__":
    main()
