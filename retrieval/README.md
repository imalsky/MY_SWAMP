# Differentiable SWAMP → phase-curve retrieval

A downstream **application** of `my_swamp` (not part of the core package). It runs
end-to-end Bayesian retrievals that differentiate through the full shallow-water
time integration to recover a tidally-locked hot Jupiter's **governing forcing
timescales** (radiative `tau_rad`, drag `tau_drag`, and optionally other GCM
scalars) from a noisy thermal phase curve:

```
parameters → SWAMP (my_swamp) terminal Φ → brightness-temperature map → intensity map
           → starry/jaxoplanet spherical-harmonic phase curve → Gaussian likelihood
```

Because the forward model is differentiable, inference uses a **gradient-informed
sampler**: BlackJAX **adaptive tempered SMC** — a swarm of particles annealed from
prior to posterior with a MALA/HMC mutation kernel. SMC is the design target
because the particle swarm vmaps onto a GPU (run 32–256 at once).

## Layout

```
retrieval/
├── README.md          # this file
├── scripts/           # all code + launcher + style guide
│   ├── pipeline.py        # importable core (forward model, likelihood, SMC)
│   ├── run_smc.py         # driver: build → observe → SMC → write data/
│   ├── plot_smc.py        # data/ → figures in plots/
│   ├── make_dashboard.py  # one consolidated results figure
│   ├── summarize_run.py   # data/ → RETRIEVAL_SUMMARY.md
│   ├── coverage_study.py  # SBC / coverage (the "run N at once" workload)
│   ├── run.sh             # SLURM/local launcher
│   ├── science.mplstyle   # publication style guide (applied to all plots)
│   └── tests/             # pytest correctness suite
├── data/              # outputs: *.npz, config.json, logs, RETRIEVAL_SUMMARY.md
└── plots/             # figures (*.png)
```

| File | Purpose |
|------|---------|
| `scripts/pipeline.py` | **Importable, config-parameterized core.** `build_pipeline(cfg)` returns the forward model, starry projector, u-space transform, prior, likelihood (with a custom forward-mode-JVP VJP), and SMC helpers. Presets: `fast_cpu_config()`, `gpu_config()`. Unit-testable without running inference. |
| `scripts/run_smc.py`  | **Thin driver.** Picks a Config (preset + env overrides), builds the pipeline, generates/loads synthetic observations, runs SMC, writes the `.npz` bundles to `data/`. |
| `scripts/plot_smc.py` | Reads `data/` and produces posterior / diagnostic / map figures into `plots/`. Never re-runs SWAMP. |
| `scripts/make_dashboard.py` | One consolidated `results_dashboard.png` (fit + joint posterior + convergence + marginals + map). |
| `scripts/summarize_run.py` | Human-readable recovery + correlation report (`data/RETRIEVAL_SUMMARY.md`). |
| `scripts/coverage_study.py` | Injection-recovery / Simulation-Based-Calibration over many truths — the rigorous calibration check, and the natural "run N at once" GPU/cluster workload (SLURM array). |
| `scripts/tests/` | Pytest correctness suite for `pipeline.py` (forward parity, projector, u-space, prior, likelihood, gradient vs finite-difference, expanded-param path, end-to-end SMC). |
| `scripts/run.sh` | SLURM/local launcher. |
| `scripts/science.mplstyle` | The project publication matplotlib style; `plot_smc.py` and `make_dashboard.py` apply it so retrieval figures match the paper figures. |

## Environment

Use the project conda env (has `jaxoplanet` 0.1.0 + `blackjax` 1.3):

```bash
conda activate MY_SWAMP
```

`pipeline.py` prepends this working tree's `src/` to `sys.path`, so it always uses
the in-tree (current, differentiable, x64-aware) `my_swamp`, never a stale
pip-installed copy.

## Quickstart

```bash
cd retrieval/scripts

# fast local CPU smoke retrieval (~2-day spin-up, float32, ~30-55 min) -> data/
SWAMP_RETRIEVAL_PRESET=fast python run_smc.py

# figures -> ../plots/ ; recovery report -> ../data/RETRIEVAL_SUMMARY.md
python plot_smc.py
python make_dashboard.py
python summarize_run.py

# correctness tests (fast subset, ~1 min):
python -m pytest tests -q -m "not slow"
# include the end-to-end SMC recovery test (~6-10 min):
python -m pytest tests -q
```

Scripts resolve `data/` and `plots/` from their own location, so they work from
any working directory. Outputs always land in `retrieval/data/` and figures in
`retrieval/plots/`.

### Presets and overrides (env vars, read before JAX import)

- `SWAMP_RETRIEVAL_PRESET` — `fast` (default), `gpu`, or `prod`.
- `SWAMP_RETRIEVAL_USE_X64` — `0`/`1` to force precision (overrides preset).
- `SWAMP_RETRIEVAL_OVERRIDES` — JSON of any Config fields, e.g.
  `'{"model_days":3.0,"smc_num_particles":64,"obs_sigma":5e-5}'`.

## The science (what to expect)

The forward model is a forced-dissipative shallow-water hot Jupiter (Perez-Becker
& Showman 2013-style). Key facts that drive the experiment design:

- **`tau_rad` sets the day-night brightness-temperature amplitude** (a strong,
  clean signal). **`tau_drag` sets the eastward hot-spot offset** via the
  superrotating jet (a weaker handle). A disk-integrated phase curve constrains
  only ~2 longitudinal harmonics (amplitude + offset), so `tau_rad` is tightly
  recovered while `tau_drag` is broader and **partially degenerate with `tau_rad`
  and `Phibar`** (gravity-wave speed). This degeneracy is real and shown honestly
  in the corner plot.
- **Spin-up.** The thermal pattern equilibrates in a few `tau_rad`; the jet/offset
  needs ~10·`tau_drag` in shallow water. With short truth timescales (~hours) a
  **2-day run is converged** for both — the basis of the `fast` preset.
- **Emission layer** (`cfg.emission_temp_mode`). Default `"geopotential"`:
  `T = (Φ̄ + Φ)/R_d` with `R_d=3.78e3` — the SWAMPE-JAX paper / Perez-Becker 2013
  temperature proxy (~80–315 K, verified to recover both taus with a strong signal).
  Alternative `"linear"`: `T = T_ref + Φ/phi_to_T_scale` (`1000`/`600` → a ~1000–2500 K
  hot-Jupiter-like map). Intensity is `T⁴` (bolometric) or Planck.
- **Noise.** Two models (`cfg.noise_model`): `"white"` (constant `obs_sigma`,
  ~80 ppm, used by the fast/test path) and `"photon"` (heteroscedastic photon
  noise `σ_i = sigma_phot / √(1 + flux_i)`, ~50 ppm floor, used by the GPU full
  run). Photon noise makes brighter (dayside) points carry slightly smaller error;
  the per-point `σ_i` is computed once from the truth and held fixed in the
  likelihood, as real per-point uncertainties are. The truth amplitude is ~4000 ppm
  → amplitude/noise ~50–80.
- **Priors.** Timescales use **log-uniform** priors (standard for scale parameters).

Inferring more parameters (`Phibar`, `DPhieq`, `omega`, …) is supported via the
`infer_*` flags; those trigger the general path that rebuilds `static` each
evaluation (still differentiable). Shape-changing parameters (`M`, `dt`) are not
inferrable.

## Precision

`float32` is the default and is **validated as posterior-unbiased** for this
problem: across the (tau_rad, tau_drag) grid the forward flux agrees with float64
to ~1e-6 and the log-posterior shape to <0.03 (out of ~470 log-units), with
identical MAP. Develop/iterate in float32; flip to float64 with
`SWAMP_RETRIEVAL_USE_X64=1` for a final cross-check or production.

## Full GPU run (`sbatch run.sh`)

`scripts/run.sh` is a SLURM launcher for the JPL **edge** GPU cluster. The `gpu`
preset runs the **full, paper-aligned retrieval**: a **64-particle** SMC swarm
(`jax.vmap`-ed → the whole swarm advances at once), a **20-day** spin-up,
**heteroscedastic photon noise**, **float64**, and the paper's temperature mapping.

```bash
cd retrieval/scripts
sbatch run.sh                       # full GPU retrieval (preset=gpu) + figures
# tune without editing files, e.g. more mixing or a shorter spin-up:
SWAMP_RETRIEVAL_OVERRIDES='{"smc_num_mcmc_steps":32,"model_days":5.0}' sbatch run.sh
```

**Why 64 (not 512):** the SWAMPE-JAX paper measures A100 throughput **saturating at
a few dozen simultaneous trajectories**. A 512-particle swarm fits in memory but just
queues into ~8 batches (≈8× the wall-time, no better posterior). **64 is the
efficient sweet spot.** Smoothness instead comes from good mixing
(`num_mcmc_steps=20`) + **KDE-smoothed** posterior plots.

**Why it scales:** the likelihood uses a **custom forward-mode-JVP gradient** (the
paper's stated approach), so there is **no reverse-mode tape** through the 7200-step
scan — peak memory is `O(n_particles · J · I)`, not `O(n_steps · …)`. The launcher
(per JAX-GPU best practice) does **not** `module load cuda` or set `LD_LIBRARY_PATH`
(that shadows the bundled-CUDA wheel → silent CPU fallback), and **aborts** if the
backend isn't GPU so SLURM fails fast instead of burning CPU hours.

**Paper alignment:** `dt=240 s`, `model_days=20` → 7200 steps (the paper's
10-day-at-120 s benchmark step count; `dt=120` gives an *identical* phase curve at
2× cost). Truth `tau_rad=10 h`, `tau_drag=6 h`; `Phibar=3e5`, `DPhieq=1e6`,
`omega=3.2e-5`, `a=8.2e7`, `M=42`; temperature `T=(Φ̄+Φ)/R_d` (`R_d=3.78e3`,
Perez-Becker 2013). float64 for robustness (`SWAMP_RETRIEVAL_USE_X64=0` → faster,
validated-unbiased float32).

**Estimated wall-time:** ≈ **1–2 h on a single A100** (7200-step forward ≈ 1.7 s per
the paper; SMC ≈ N×mcmc×tempering value-and-grads; ~15–20 tempering stages) — well
within the 48 h SLURM limit. The forward is converged by ~5 days, so
`model_days=5` cuts it ~4×.

For **many independent retrievals at once** (a calibration study), use
`coverage_study.py` as a SLURM array — one task per injected truth:

```bash
cd retrieval/scripts
# array task: one simulation seeded by the array index
python coverage_study.py --n_sim 1 --seed $SLURM_ARRAY_TASK_ID --out_dir ../data/cov_out
# then aggregate + report coverage / SBC ranks
python coverage_study.py --aggregate --out_dir ../data/cov_out
```

## Outputs

`run_smc.py` writes to `retrieval/data/`: `config.json`, `observations.npz`,
`posterior_samples.npz`, `mcmc_extra_fields.npz` (SMC diagnostics),
`posterior_predictive*.npz`, `maps_truth_and_posterior_summary.npz`, plus
`run.log` and `RETRIEVAL_SUMMARY.md`. `plot_smc.py` + `make_dashboard.py` write
figures to `retrieval/plots/`: `results_dashboard.png` (everything in one image),
phase-curve fit + residuals, 1-D posteriors with prior overlays, a **corner plot**
(shows the tau_rad–tau_drag degeneracy), SMC diagnostics (tempering schedule, ESS,
acceptance, logZ), terminal Φ/T/I maps (truth vs posterior), and starry disk
renders. `data/` and `plots/` contents are regenerable (gitignored).

## Provenance

Restored from git history (`run_mala.py`/`plot_mala.py`, removed in commit
`9d0d400`), refactored into the `pipeline.py` + `run_smc.py` split, then
reorganized into `scripts/` + `data/` + `plots/`. The pre-refactor monolith was
removed (its numerics live, verified, in `pipeline.py`).
