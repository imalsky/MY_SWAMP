#!/usr/bin/env python3
"""coverage_study.py — injection-recovery / SBC calibration for the retrieval.

Runs many independent retrievals at different injected truths (drawn from the
prior) and checks that the posteriors are *calibrated*: the truth should fall
inside the X% credible interval X% of the time, and the rank of the truth among
the posterior samples should be uniform (Simulation-Based Calibration; Talts
et al. 2018, arXiv:1804.06788). This is the gold-standard correctness check for
a Bayesian pipeline beyond single-case injection-recovery.

Each retrieval is one SMC run (minutes on CPU), so the full study is a GPU /
cluster job — this is exactly the "run 32+ at once" workload:

  * Embarrassingly parallel: launch as a SLURM array, one task per simulation
    (``--n_sim 1 --seed $SLURM_ARRAY_TASK_ID``), then aggregate the per-task
    ``coverage_sim_*.npz`` files with ``--aggregate``.
  * Or run a batch in one process with ``--n_sim N`` (loops sequentially; the
    SMC swarm itself is vmapped over particles on the device).

The pipeline is built ONCE: the injected truth only changes the tau arguments,
which the fast-path forward model takes per-evaluation (static/IC do not depend
on the timescales), so we just reset ``theta_truth`` + regenerate observations
per simulation.

Usage
-----
  # one simulation (array-task friendly), writes coverage_sim_<seed>.npz
  python coverage_study.py --n_sim 1 --seed 0 --out_dir cov_out
  # a sequential batch
  python coverage_study.py --n_sim 20 --seed 0 --out_dir cov_out
  # aggregate + report
  python coverage_study.py --aggregate --out_dir cov_out
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _build_pipeline(use_x64: bool):
    os.environ["SWAMPE_JAX_ENABLE_X64"] = "1" if use_x64 else "0"
    os.environ.setdefault("JAX_ENABLE_X64", "1" if use_x64 else "0")
    import pipeline as P
    cfg = P.fast_cpu_config(use_x64=use_x64)
    return P, cfg, P.build_pipeline(cfg)


def run_batch(args) -> None:
    import jax
    P, cfg, pipe = _build_pipeline(bool(args.x64))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lo = np.asarray(pipe.param_prior_lo, float)
    hi = np.asarray(pipe.param_prior_hi, float)
    prior_types = [s.prior_type for s in pipe.specs]

    def draw_truth(rng) -> np.ndarray:
        """Draw a truth from the (log-)uniform prior for each parameter."""
        out = np.empty(pipe.n_dim)
        for i, pt in enumerate(prior_types):
            if pt == "log10_uniform":
                out[i] = 10 ** rng.uniform(np.log10(lo[i]), np.log10(hi[i]))
            else:
                out[i] = rng.uniform(lo[i], hi[i])
        return out

    import jax.numpy as jnp
    for k in range(int(args.n_sim)):
        sim_seed = int(args.seed) + k
        rng = np.random.default_rng(1000 + sim_seed)
        truth = draw_truth(rng)
        # reset truth + regenerate observations with this sim's noise
        pipe.theta_truth = jnp.asarray(truth, pipe.dtype)
        P.generate_observations(pipe, seed=sim_seed)
        res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(sim_seed), progress=False)
        samp = res["theta_draws"].reshape(-1, pipe.n_dim)

        rec = {}
        for i, name in enumerate(pipe.param_names):
            s = samp[:, i]
            q = np.percentile(s, [2.5, 25, 50, 75, 97.5])
            rec[f"{name}_truth"] = truth[i]
            rec[f"{name}_q"] = q
            rec[f"{name}_in50"] = float(q[1] <= truth[i] <= q[3])
            rec[f"{name}_in95"] = float(q[0] <= truth[i] <= q[4])
            # SBC rank: number of posterior draws below the truth (normalized)
            rec[f"{name}_rank"] = float(np.mean(s < truth[i]))
        rec["reached_beta1"] = float(res["reached_beta1"])
        path = out_dir / f"coverage_sim_{sim_seed:05d}.npz"
        np.savez(path, param_names=np.asarray(pipe.param_names, "<U64"), **rec)
        print(f"[sim {sim_seed}] truth={np.round(truth,2)} "
              f"in50={[rec[f'{n}_in50'] for n in pipe.param_names]} "
              f"in95={[rec[f'{n}_in95'] for n in pipe.param_names]} beta1={rec['reached_beta1']:.0f}")


def aggregate(args) -> None:
    out_dir = Path(args.out_dir)
    files = sorted(out_dir.glob("coverage_sim_*.npz"))
    if not files:
        raise SystemExit(f"No coverage_sim_*.npz in {out_dir}")
    names = [str(x) for x in np.load(files[0], allow_pickle=True)["param_names"].tolist()]
    agg = {n: {"in50": [], "in95": [], "rank": []} for n in names}
    n_beta1 = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        n_beta1 += int(d["reached_beta1"]) if "reached_beta1" in d.files else 1
        for n in names:
            agg[n]["in50"].append(float(d[f"{n}_in50"]))
            agg[n]["in95"].append(float(d[f"{n}_in95"]))
            agg[n]["rank"].append(float(d[f"{n}_rank"]))
    N = len(files)
    print(f"\n=== Coverage / SBC over {N} simulations ({n_beta1}/{N} reached beta=1) ===")
    print("Well-calibrated: in50 ~ 0.50, in95 ~ 0.95, ranks ~ Uniform(0,1).")
    for n in names:
        in50 = np.mean(agg[n]["in50"]); in95 = np.mean(agg[n]["in95"])
        ranks = np.asarray(agg[n]["rank"])
        # simple uniformity check: mean rank ~0.5, KS-ish max dev
        print(f"  {n:16s} cov50={in50:.2f}  cov95={in95:.2f}  mean_rank={ranks.mean():.2f} (want 0.50)")
    np.savez(out_dir / "coverage_summary.npz",
             param_names=np.asarray(names, "<U64"),
             **{f"{n}_in50": np.asarray(agg[n]["in50"]) for n in names},
             **{f"{n}_in95": np.asarray(agg[n]["in95"]) for n in names},
             **{f"{n}_rank": np.asarray(agg[n]["rank"]) for n in names})
    print(f"Saved {out_dir/'coverage_summary.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sim", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="coverage_outputs")
    ap.add_argument("--x64", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    aggregate(a) if a.aggregate else run_batch(a)
