#!/usr/bin/env python3
"""run_smc.py — thin driver for the differentiable SWAMP -> phase-curve retrieval.

All the science (forward model, starry projector, u-space transform, likelihood,
and the BlackJAX adaptive-tempered-SMC machinery) lives in ``pipeline.py``. This
driver only: chooses a Config, sets up logging + output dir, builds the pipeline,
generates/loads synthetic observations, runs SMC, and writes the ``.npz`` bundles
that ``plot_smc.py`` consumes. The output schema is unchanged from the historical
monolithic script.

Presets / overrides (env vars, read before JAX import)
------------------------------------------------------
- ``SWAMP_RETRIEVAL_PRESET``  : ``fast`` (default; ~2-day CPU smoke, float32),
                                ``gpu`` (large SMC swarm for accelerators), or
                                ``prod`` (50-day, float64, big SMC).
- ``SWAMP_RETRIEVAL_USE_X64`` : ``0``/``1`` to force precision (overrides preset).
- ``SWAMP_RETRIEVAL_OVERRIDES``: JSON object of Config field overrides, e.g.
                                 ``'{"model_days": 3.0, "obs_sigma": 5e-5}'``.
- ``SWAMP_PLOT_OUT_DIR`` / ``cfg.out_dir`` : where outputs are written.

Examples
--------
    # fast local CPU smoke (default):
    SWAMP_RETRIEVAL_PRESET=fast python run_smc.py
    # production float64:
    SWAMP_RETRIEVAL_PRESET=prod python run_smc.py
    # custom:
    SWAMP_RETRIEVAL_OVERRIDES='{"model_days":3.0,"smc_num_particles":48}' python run_smc.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ---- precision must be decided BEFORE importing pipeline (which imports JAX) ----
_PRESET = os.environ.get("SWAMP_RETRIEVAL_PRESET", "fast").strip().lower()
_x64_env = os.environ.get("SWAMP_RETRIEVAL_USE_X64", "").strip()
if _x64_env != "":
    _USE_X64 = _x64_env not in ("0", "false", "no")
else:
    # gpu + prod default to float64 (matches gpu_config()/prod); fast defaults to float32.
    _USE_X64 = _PRESET in ("prod", "gpu")
os.environ["SWAMPE_JAX_ENABLE_X64"] = "1" if _USE_X64 else "0"
os.environ.setdefault("JAX_ENABLE_X64", "1" if _USE_X64 else "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pipeline as P  # noqa: E402  (after env setup)


def make_config() -> P.Config:
    """Build the Config from the preset + env overrides."""
    if _PRESET == "fast":
        cfg = P.fast_cpu_config(use_x64=_USE_X64)
    elif _PRESET == "gpu":
        cfg = P.gpu_config(use_x64=_USE_X64)  # large SMC swarm for accelerators
    elif _PRESET == "prod":
        cfg = P.Config(use_x64=_USE_X64)  # 50-day production defaults
    else:
        raise ValueError(f"Unknown SWAMP_RETRIEVAL_PRESET={_PRESET!r} (use 'fast', 'gpu', or 'prod').")

    # Default outputs land in retrieval/data/ (overridable below).
    cfg = replace(cfg, out_dir=P.DATA_DIR)

    ov = os.environ.get("SWAMP_RETRIEVAL_OVERRIDES", "").strip()
    if ov:
        overrides: Dict[str, Any] = json.loads(ov)
        if "out_dir" in overrides:
            overrides["out_dir"] = Path(overrides["out_dir"])
        cfg = replace(cfg, **overrides)
    return cfg


def write_config_json(cfg: P.Config, pipe: P.Pipeline) -> None:
    """Write config.json with the inferred-parameter metadata plot_smc.py needs."""
    cfg_dict = asdict(cfg)
    cfg_dict.update(dict(
        inferred_param_names=pipe.param_names,
        inferred_param_labels=pipe.param_labels,
        inferred_param_prior_types=[s.prior_type for s in pipe.specs],
        inferred_param_prior_lo=pipe.param_prior_lo.tolist(),
        inferred_param_prior_hi=pipe.param_prior_hi.tolist(),
        inferred_param_truth=pipe.param_truth.tolist(),
    ))
    (cfg.out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))


def main() -> None:
    cfg = make_config()
    P.validate_config(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(cfg.out_dir / "run.log",
                                                               mode="w" if cfg.overwrite else "a")],
        force=True,
    )
    logger = logging.getLogger("swamp_retrieval")

    import jax  # noqa: E402  (already configured via pipeline import)
    logger.info(f"preset={_PRESET} use_x64={_USE_X64} backend={jax.default_backend()} devices={jax.devices()}")

    # ---- build pipeline ----
    t0 = time.perf_counter()
    pipe = P.build_pipeline(cfg)
    logger.info(f"Built pipeline in {time.perf_counter()-t0:.1f}s | grid I={pipe.I} J={pipe.J} "
                f"n_steps={pipe.n_steps} | inferred {pipe.n_dim}: {pipe.param_names}")
    write_config_json(cfg, pipe)

    # ---- observations ----
    obs_path = cfg.out_dir / "observations.npz"
    if cfg.generate_synthetic_data or not obs_path.exists():
        logger.info(f"Generating synthetic observations (noise_model={cfg.noise_model})...")
        obs = P.generate_observations(pipe, seed=cfg.seed)
        sigma_vec = np.asarray(obs["obs_sigma"], dtype=np.float64)  # per-point
        # obs_sigma kept as a SCALAR mean for back-compat with plot_smc.py; the full
        # per-point vector is obs_sigma_vec (used by the likelihood).
        P.save_npz(obs_path, times_days=obs["times_days"], flux_true=obs["flux_true"],
                   flux_obs=obs["flux_obs"],
                   obs_sigma=float(obs["obs_sigma_mean"]), obs_sigma_vec=sigma_vec,
                   noise_model=np.asarray(str(cfg.noise_model), dtype="<U16"),
                   orbital_period_days=obs["orbital_period_days"],
                   inferred_param_names=np.asarray(pipe.param_names, dtype="<U64"),
                   inferred_param_truth=np.asarray(pipe.param_truth, dtype=np.float64))
        logger.info(f"Saved observations to: {obs_path}")
    else:
        d = np.load(obs_path)
        sigma_load = d["obs_sigma_vec"] if "obs_sigma_vec" in d.files else float(d["obs_sigma"])
        pipe.set_observations(d["flux_obs"], obs_sigma=sigma_load)
        pipe.flux_true = np.asarray(d["flux_true"])
        sigma_vec = np.atleast_1d(np.asarray(sigma_load, dtype=np.float64))
        logger.info(f"Loaded observations from: {obs_path}")

    amp = float(np.ptp(pipe.flux_true))
    sig_mean = float(np.mean(sigma_vec))
    logger.info(f"Truth phase-curve amplitude={amp*1e6:.1f} ppm | per-point sigma "
                f"[{sigma_vec.min()*1e6:.1f}-{sigma_vec.max()*1e6:.1f}] ppm (mean {sig_mean*1e6:.1f}) "
                f"| amplitude/noise={amp/sig_mean:.1f}")

    # ---- inference ----
    samples_path = cfg.out_dir / "posterior_samples.npz"
    extra_path = cfg.out_dir / "mcmc_extra_fields.npz"
    if cfg.run_inference:
        logger.info("Running BlackJAX adaptive tempered SMC...")
        t0 = time.perf_counter()
        res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(int(cfg.seed)), progress=True)
        logger.info(f"SMC finished in {(time.perf_counter()-t0)/60:.1f} min | reached_beta1={res['reached_beta1']} "
                    f"| n_temper={len(res['betas'])-1} | step_size={res['step_size_used']:.4g}")
        if not res["reached_beta1"]:
            logger.warning(f"SMC did not reach beta=1 (final {res['final_beta']:.4f}); "
                           "increase smc_max_steps or lower smc_target_ess_frac.")

        P.save_npz(samples_path,
                   param_names=np.asarray(pipe.param_names, dtype="<U64"),
                   param_labels=np.asarray(pipe.param_labels, dtype="<U64"),
                   samples=res["theta_draws"])
        logger.info(f"Saved posterior samples to: {samples_path}")

        logz_inc = res["logZ_increment"]
        logz = np.cumsum(np.nan_to_num(logz_inc, nan=0.0))
        P.save_npz(extra_path,
                   inference_method=np.asarray(2, dtype=np.int32),
                   smc_kernel=np.asarray(str(cfg.smc_mcmc_kernel), dtype="<U16"),
                   smc_resampling=np.asarray(str(cfg.smc_resampling), dtype="<U16"),
                   smc_num_particles=np.asarray(int(cfg.smc_num_particles), dtype=np.int32),
                   smc_num_mcmc_steps=np.asarray(int(cfg.smc_num_mcmc_steps), dtype=np.int32),
                   smc_mcmc_step_size=np.asarray(float(res["step_size_used"]), dtype=np.float64),
                   smc_mcmc_step_size_auto_tuned=np.asarray(int(bool(cfg.mcmc_auto_tune)), dtype=np.int32),
                   smc_target_ess_frac=np.asarray(float(cfg.smc_target_ess_frac), dtype=np.float64),
                   smc_betas=res["betas"], smc_ess=res["ess"],
                   smc_acceptance_rate=res["acceptance_rate"],
                   smc_logZ_increment=logz_inc, smc_logZ=logz,
                   smc_final_weights=res["final_weights"],
                   inferred_param_names=np.asarray(pipe.param_names, dtype="<U64"),
                   inferred_param_truth=np.asarray(pipe.param_truth, dtype=np.float64))
        logger.info(f"Saved SMC diagnostics to: {extra_path}")

        # console recovery summary
        theta = res["theta_draws"].reshape(-1, pipe.n_dim)
        truth = np.asarray(pipe.theta_truth)
        logger.info("Recovery (median [5%,95%], truth):")
        for i, name in enumerate(pipe.param_names):
            q = np.percentile(theta[:, i], [5, 50, 95])
            inside = "in" if q[0] <= truth[i] <= q[2] else "OUT"
            logger.info(f"  {name:16s} {q[1]:9.3f} [{q[0]:8.3f},{q[2]:8.3f}]  truth={truth[i]:9.3f}  ({inside} 90% CI)")
    else:
        if not samples_path.exists():
            raise FileNotFoundError(f"posterior_samples.npz not found at {samples_path}")
        logger.info("run_inference=False; using existing posterior_samples.npz.")

    # ---- posterior predictive ----
    if cfg.do_ppc:
        logger.info("Computing posterior-predictive phase curves...")
        import jax.numpy as jnp
        s = np.load(samples_path)
        theta_all = np.asarray(s["samples"]).reshape(-1, pipe.n_dim).astype(pipe.npdtype)
        rng = np.random.default_rng(cfg.seed + 1)
        n_take = min(int(cfg.ppc_draws), theta_all.shape[0])
        sel = theta_all[rng.choice(theta_all.shape[0], size=n_take, replace=False)]
        preds = []
        for i0 in range(0, n_take, int(cfg.ppc_chunk_size)):
            batch = jnp.asarray(sel[i0:i0 + int(cfg.ppc_chunk_size)], pipe.dtype)
            preds.append(np.asarray(jax.vmap(pipe.phase_curve_model_jit)(batch)))
        ppc = np.concatenate(preds, axis=0)
        P.save_npz(cfg.out_dir / "posterior_predictive.npz", ppc_draws=ppc, theta_sel=sel, times_days=pipe.times_days)
        P.save_npz(cfg.out_dir / "posterior_predictive_quantiles.npz",
                   p05=np.nanquantile(ppc, 0.05, axis=0), p50=np.nanquantile(ppc, 0.50, axis=0),
                   p95=np.nanquantile(ppc, 0.95, axis=0), times_days=pipe.times_days)
        logger.info("Saved posterior-predictive files.")

    # ---- truth + posterior-median maps (so plotting never reruns SWAMP) ----
    logger.info("Computing truth + posterior-median terminal maps...")
    import jax.numpy as jnp
    truth_maps = pipe.compute_maps_for_theta(pipe.theta_truth)
    s = np.load(samples_path)
    theta_median = np.median(np.asarray(s["samples"]).reshape(-1, pipe.n_dim), axis=0).astype(pipe.npdtype)
    post_maps = pipe.compute_maps_for_theta(jnp.asarray(theta_median, pipe.dtype))
    P.save_npz(cfg.out_dir / "maps_truth_and_posterior_summary.npz",
               lon=np.asarray(pipe.lon), lat=np.asarray(pipe.lat),
               phi_truth=truth_maps["phi"], T_truth=truth_maps["T"], I_truth=truth_maps["I"], y_truth=truth_maps["y_dense"],
               phi_post=post_maps["phi"], T_post=post_maps["T"], I_post=post_maps["I"], y_post=post_maps["y_dense"],
               inferred_param_names=np.asarray(pipe.param_names, dtype="<U64"),
               inferred_param_truth=np.asarray(pipe.param_truth, dtype=np.float64),
               inferred_param_post_median=np.asarray(theta_median, dtype=np.float64))
    logger.info("Saved truth + posterior-median maps. DONE.")


if __name__ == "__main__":
    main()
