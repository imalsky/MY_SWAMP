# Claude guide for `my_swamp`

This file is the in-repo briefing for an AI coding assistant working on this
package. It is short on purpose. If you only read one file before editing
code, read this one.

---

## 1. What this project is

`my_swamp` is a JAX rewrite of the NumPy/SciPy SWAMPE spectral shallow-water
model on the sphere. The numerical core runs inside `jax.lax.scan`, so the
forward simulation is end-to-end differentiable with respect to continuous
physical parameters and explicit initial conditions.

Two priorities, in order:

1. **Numerical parity with reference NumPy SWAMPE** for default settings.
   Every behavior described in §3 below is reproduced bit-for-bit modulo
   XLA reassociation (≤ 1e-10 atol on `eta`/`delta`, ≤ 5e-8 atol on `Phi`).
2. **Differentiability**: no `float(tracer)` coercions, no NumPy-on-tracer
   calls, no Python-side mutation in the scan body. Scalars that should be
   differentiable parameters (`DPhieq`, `taurad`, `taudrag`, `K6`, `Phibar`,
   `omega`, `a`, `dt`, `alpha`) are wrapped as JAX arrays inside `Static` /
   `RunFlags` so `jax.grad` and `jax.jvp` work end-to-end.

The reference NumPy implementation lives in the sibling directory
`../SWAMPE/` and is used by the parity tests + long-run comparison script.
It is **not** shipped inside this package.

---

## 2. Repo layout

```
MY_SWAMP/
├── CLAUDE.md                    # this file
├── README.md                    # user-facing docs, install, examples
├── CONTRIBUTING.md              # contribution conventions
├── LICENCE.txt                  # BSD-3-Clause
├── pyproject.toml               # build metadata, deps, ruff/pytest config
├── setup.py                     # legacy setuptools shim
├── src/my_swamp/
│   ├── __init__.py              # package entry; sets jax_enable_x64
│   ├── _version.py
│   ├── dtypes.py                # float_dtype / complex_dtype switch on x64
│   ├── branching.py             # cond/select/maybe_apply with static-pred specialization
│   ├── backend_preflight.py     # JAX backend / device validation
│   ├── spectral_transform.py    # Gauss–Legendre quadrature, Pmn/Hmn basis,
│   │                            #   FFT truncation, Legendre transforms,
│   │                            #   wind inversion (invrsUV)
│   ├── initial_conditions.py    # supported resolutions, analytic ICs, ABCDE
│   ├── time_stepping.py         # scheme dispatch + coefficient arrays + RMS_winds
│   ├── modEuler_tdiff.py        # modified-Euler scheme (default)
│   ├── explicit_tdiff.py        # explicit (leapfrog) scheme
│   ├── forcing.py               # Phieq, Q, R (radiative + drag) forcing
│   ├── filters.py               # diffusion filter coefficients + apply
│   ├── continuation.py          # pickle save/load + timestamp arithmetic
│   ├── plotting.py              # matplotlib helpers + GIF generation (lazy import)
│   ├── autodiff_utils.py        # forward-mode JVP helper (fwd_grad)
│   └── model.py                 # *** main driver: run_model, run_model_scan,
│                                #     run_model_scan_final, run_model_gpu,
│                                #     assert_finite_state, plus Static/RunFlags/State
├── unit_tests/                  # pytest suite (smoke/parity markers)
│   ├── conftest.py              # x64 + backend setup
│   ├── fixtures/*.npz           # SWAMPE-generated reference snapshots
│   └── test_*.py
├── testing/                     # not pytest-collected
│   ├── benchmark_scan.py             # forward-scan wall-clock microbenchmark
│   ├── benchmark_gradient.py         # reverse-mode grad cost + vmap throughput (paper numbers)
│   ├── compare_long_run_parity.py    # SWAMPE vs my_swamp parity + Figure 1 + speedup
│   ├── make_sensitivity_figure.py    # Figure 2 (100-day AD sensitivity maps)
│   ├── generate_reference_parity_fixtures.py
│   ├── science.mplstyle
│   └── long_run_parity_outputs/ # generated artifacts (gitignored)
└── paper/                       # JOSS paper (LaTeX): paper.tex, paper.bib, figures, Makefile, README
```

**The JOSS paper is LaTeX.** `paper/paper.tex` is the canonical, authoritative
source of the paper — there is no `paper.md`. Build with `cd paper && make`
(figures regenerate with `make figures`). JOSS's own pipeline ingests Markdown,
but this project keeps the paper in LaTeX only; a `paper.md` would be generated
on demand if an upload required it, never maintained as a second source.

The driver `model.py` is the only large file (~1900 lines). If you only have
time for one read, that's the one. Everything else is small and orthogonal.

---

## 3. Locked parity contract (do not change without re-baselining)

These behaviors are **deliberately** preserved from reference NumPy SWAMPE,
including its historical quirks. Each item has a unit test in
`unit_tests/test_parity_quirks.py`. If you change one, regenerate the
fixtures (`testing/generate_reference_parity_fixtures.py`) and bump the spec
section in this file.

1. **Modified-Euler `Phi`/`Delta` use the effective `/4` coefficient**
   (`tstepcoeff/4`, `tstepcoeff2/4`) inside `modEuler_tdiff.{phi,delta}_timestep`.
   The reference SWAMPE arrives at this by halving twice; we set it directly.
2. **Modified-Euler `Delta` uses `Bm+Fm` and `Am-Gm` even when `forcflag=False`**
   (`modEuler_tdiff.delta_timestep`). This is a SWAMPE historical quirk where
   the F/G terms leak into the divergence tendency through the unforced branch.
3. **Modified-Euler `Eta` forced/unforced asymmetry**: forced uses
   `tstepcoeff1` and `(Am-Gm, Bm+Fm)`; unforced uses `tstepcoeff1/2` and
   `(Am, Bm)`. (`modEuler_tdiff.eta_timestep`)
4. **Explicit `Delta` is carry-only**: `deltamntstep = deltacomp1` only,
   even though SWAMPE computes the other components and discards them.
   (`explicit_tdiff.delta_timestep`)
5. **Explicit `Eta`/`Delta` add extra drag-linked forcing terms**
   (`U/τ_drag`, `V/τ_drag`) on top of `Fm/Gm`, not in the modified-Euler scheme.
6. **Dayside mask in `Phieqfun` uses strict `<` inequality** at the terminator
   (`forcing.Phieqfun`). The exact terminator longitudes are nightside.
7. **`Q < 0` is clamped to 0 in `Rfun`**; `taudrag == -1` disables Rayleigh
   drag entirely (returns `Ru, Rv` only).
8. **Legendre normalization**: factorial-based scaling with a sign flip for
   odd `m`, `n>0`. Matches SciPy's `lpmn`-based convention up to
   Condon–Shortley phase, then is corrected. (`spectral_transform.PmnHmn`)
9. **Inverse-Legendre negative-`m` layout**: `[:, I-M:I]` slot, conjugate of
   positive-`m`, reversed order. (`spectral_transform.invrs_leg`)
10. **Two-level initialization** (`prev == curr == initial`) and
    **Robert-Asselin filtering** that is applied for `t > 2`. The filter
    affects only the *physical-space* `eta_prev`/`delta_prev`/`Phi_prev` carry,
    not the spectral coefficients (`etam_prev`, etc.). This deliberate
    desync is preserved.
11. **Float64 mode is required for parity-grade comparisons.**
    `SWAMPE_JAX_ENABLE_X64=1` (default) → `jax_enable_x64=True`. The
    reference SWAMPE uses NumPy/SciPy at float64 by default.
12. **`_forcing_phys` computes `F`/`G`/`PhiF` whenever `test is None`**,
    regardless of `forcflag`. This is required for parity with the SWAMPE
    historical quirk in item 2. The `forcflag` switch only gates whether
    those terms are *added* by the timestepper.

If you find yourself changing something in `modEuler_tdiff.py`,
`explicit_tdiff.py`, `spectral_transform.py`, or `_forcing_phys`, double-check
this list first.

---

## 4. Public API contract

Top-level entry points (all in `my_swamp.model`):

| Function | Use when |
|----------|----------|
| `run_model(...)` | SWAMPE-compatible call signature. Side-effecting (save/plot). Returns a dict with terminal fields and diagnostics. |
| `run_model_scan(...)` | Differentiable full-history scan. Returns `outs` time histories, `last_state`, `dead_first_idx`. **Memory cliff** for large `tmax`. |
| `run_model_scan_final(...)` | Recommended for AD/optimization. Same as above with `return_history=False`. Memory ∝ J·I, not tmax·J·I. |
| `run_model_gpu(...)` | Wrapper around `run_model` with GPU/AD-friendly defaults (`plotflag=False, saveflag=False, as_numpy=False, jit_scan=True`). |
| `assert_finite_state(last_state)` | Host-side NaN/Inf check after `diagnostics=False` runs. |
| `fwd_grad(loss, theta, chunk=None)` | In `my_swamp.autodiff_utils`. Forward-mode gradient with optional JVP chunking. |

Required minimum kwargs for any `run_model_scan*` call:
`M`, `dt`, `tmax`, `Phibar`, `omega`, `a`.

Test selectors: `None` (forced mode), `1`, `2` only. `0` maps to `None` in
the legacy `main_function.main`. Other values (3, 9, 10, 11) raise.

Resolutions: `M ∈ {42, 63, 106}` only (set by `initial_conditions.spectral_params`).

`K6Phi=None` means "inherit `K6`" — preserves SWAMPE's legacy default of
using the same hyperdiffusion coefficient for vorticity/divergence and
geopotential.

---

## 5. Differentiability rules

The single most common way to break this rewrite is to coerce a JAX tracer
to a Python value somewhere inside the scan body. Symptoms:

- `jax.grad` returns `0.0`.
- `jax.jit` recompiles every call with a slight parameter change.
- `TracerArrayConversionError` at runtime.

Rules to keep:

1. **Never call `float(...)`, `int(...)`, `bool(...)`, or `np.asarray(...)`
   on a tracer.** `_is_python_scalar` in `model.py` exists to gate Python-side
   validation behind a concrete-scalar check.
2. **Scalar physical parameters live in `Static`** (`dt`, `a`, `omega`, `g`,
   `Phibar`, `taurad`, `taudrag`) and are stored as JAX 0-D arrays. They
   appear as `children` in `Static.tree_flatten`, not in `aux_data`. This
   makes them reach `jax.grad` cleanly.
3. **Boolean flags live in `RunFlags.aux_data`** (`forcflag`, `diffflag`,
   `expflag`, `modalflag`, `diagnostics`). Changing them changes the
   pytree structure → triggers recompilation. That's intentional.
4. **`alpha` and `blowup_rms` live in `RunFlags.children`** (JAX arrays),
   not `aux_data`. They can vary between calls without recompilation.
5. **The `test` selector is a Python `int` or `None`** and is part of the
   `lru_cache` key for `_get_simulate_scan_jit` / `_get_simulate_scan_last_jit`.
   Different test values → different compiled scan.
6. **If you add a new branch in the scan body**, use `branching.cond` /
   `branching.maybe_apply` (which collapse to a Python branch when the
   predicate is statically known) rather than raw `jax.lax.cond` (which
   traces both branches).

The cross-check that will catch most regressions:
`unit_tests/test_autodiff.py::test_grad_matches_finite_difference_for_DPhieq`.
Run it first whenever you touch the scan body.

---

## 6. Validation commands

```bash
# All tests, x64 mode (the one that has to pass before merging anything):
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 JAX_ENABLE_X64=1 pytest -q

# Smoke only (fast):
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 pytest -q -m smoke

# Parity only:
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 pytest -q -m parity

# Confirm that the x64 gate actually fires (these parity tests are
# expected to FAIL — that's the validation of the gate):
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=0 JAX_ENABLE_X64=0 pytest -q -m parity

# Lint:
ruff check src unit_tests testing

# Long-run parity vs reference SWAMPE (requires ../SWAMPE/ to exist;
# not part of pytest because it takes minutes):
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 python testing/compare_long_run_parity.py --days 100

# Regenerate parity fixtures (after a deliberate numerics change):
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 python testing/generate_reference_parity_fixtures.py

# Benchmark:
python testing/benchmark_scan.py --M 42 --tmax 300 --timed-runs 3
```

Current test count: 36. Suite runtime: ~25–40s on CPU.

---

## 7. Test markers

- `smoke` — fast sanity checks. Run on every commit. Includes the autodiff
  smoke tests (these compile a JIT'd scan; they're slower than the rest of
  smoke but still under ~10s combined).
- `parity` — strict regression checks against fixtures generated from
  reference NumPy SWAMPE. Gated on x64 mode by `_assert_x64_enabled`.
- `perf` — benchmark/perf-oriented checks. Reserved; not currently used
  inside pytest (the benchmark script in `testing/` is run separately).

Tests with no marker run in both default and parity-gated invocations.

---

## 8. What the unit tests cover

| File | What it tests |
|------|---------------|
| `test_import_and_version.py` | Package imports cleanly, version string is well-formed. |
| `test_backend_preflight.py` | CPU backend is reported, invalid backend raises. |
| `test_static_spectral_params.py` | Grid sizes / lambdas / mus / w shapes for M=42. |
| `test_transform_stack.py` | Forward+inverse Legendre/FFT round-trips, `Pmn`/`Hmn` reference values, wind ↔ vorticity/divergence diagnostic. |
| `test_model_scan_smoke.py` | One end-to-end `run_model_scan` step, asserts finite outputs. |
| `test_parity_quirks.py` | One test per locked-parity item in §3. |
| `test_parity_reference_regression.py` | Compares terminal + snapshot fields against `unit_tests/fixtures/*.npz`. |
| `test_autodiff.py` | `jax.grad`/`jvp` smoke across 9 scalar parameters; FD cross-check on `DPhieq`; `jax.grad` over a `(J, I)` initial-Phi field. |
| `test_continuation_roundtrip.py` | contflag-resume reproduces an explicit-IC restart from the same single-level state. |
| `test_invalid_input.py` | `pytest.raises` for bad `tmax`, `dt`, `M`, `test`, partial/wrong-shape ICs, contflag without contTime, non-numeric contTime. |
| `test_vmap_smoke.py` | `jax.vmap` over a stack of `DPhieq` values; per-member agreement with direct calls. |

When adding a feature, add or extend the matching file. When adding a new
parity quirk, also extend `test_parity_quirks.py`.

---

## 9. Common pitfalls

- **The save filename arithmetic in `compute_timestamp(units, t, dt)` is
  multiplicative**, so SWAMPE's swapped-arg call `(units, dt, t)` happens to
  produce the right filename. MY_SWAMP fixed the call site (`model.py`
  calls it as `(units, t, dt)`). Don't "fix" the function signature back —
  it would break the SWAMPE-shipped pickle filenames.
- **The `Phi+Phibar` denominator in `forcing.Rfun` is guarded** with
  `jnp.where(jnp.abs(phi_total) > 0, phi_total, finfo.tiny)`. This is a
  small departure from SWAMPE that only triggers on pathological transients.
  Don't remove the guard.
- **`arccos` in `state_var_init(test=1)` is clipped** to `[-1, 1]` before
  the call, otherwise FP overshoot at boundary points produces NaN. This is
  a deliberate one-line departure from SWAMPE that is parity-safe (matches
  to within ULP for in-range inputs).
- **`return_history=True` is a memory cliff.** For long `tmax` use
  `run_model_scan_final` (≡ `return_history=False`). The readme has a
  sizing table. Defaulting to history is for SWAMPE compatibility, not
  because it's the right choice for long runs.
- **`donate_state=True` requires the `prev==curr` aliasing to be
  deduplicated**, which `_dedupe_state_for_donation` handles. If you add
  fields to `State` that share buffers at init, this helper will need to
  see them too.
- **`flags.diagnostics=False` skips the in-scan blowup gate.** Use
  `assert_finite_state(last_state)` after the run to detect silent NaN
  propagation. The new `dead_first_idx` field in the scan return tells you
  the first scan-step index at which the gate tripped (if it did).
- **Imports ordering matters for `jax_enable_x64`.** Setting
  `SWAMPE_JAX_ENABLE_X64=1` after JAX has already created arrays does
  nothing. The package `__init__.py` reads the env var and configures JAX
  before any of the model modules import `jax.numpy`. Don't move JAX
  imports above the config block.

---

## 10. Where to start when…

- **A parity test fails after a refactor** → first re-read §3 (locked
  parity contract). Most failures are caused by drifting from one of these
  quirks.
- **`jax.grad` returns 0** → run
  `pytest unit_tests/test_autodiff.py::test_grad_matches_finite_difference_for_DPhieq`.
  If it fails, grep the diff for `float(...)`, `int(...)`, `bool(...)`,
  `np.asarray(...)` — one of these slipped past somewhere.
- **A `lax.scan` recompiles every call** → check whether you changed
  something in `RunFlags.aux_data` (Python bool changes recompile) or
  introduced a new shape in `Static`. Use `jax.config.update("jax_log_compiles", True)`
  to see what triggered it.
- **OOM on long runs** → switch from `run_model_scan(..., return_history=True)`
  to `run_model_scan_final(...)`.
- **Adding a new test case** → add to
  `initial_conditions.state_var_init`/`velocity_init`, then update
  `_analytic_ic` in `model.py`, then extend the `if test in (None, 1, 2)`
  guard in the legacy `main_function.main`.
- **Adding a new differentiable parameter** → add to `Static`'s
  `tree_flatten` `children` (not `aux_data`), wire it through the timestepper,
  and add a row to `test_autodiff.py::test_grad_returns_finite_for_each_scalar_parameter`.

---

## 11. Change-control checklist

When you make a change that affects locked numerical behavior:

1. State the change plainly (in the PR description and the relevant docstring).
2. Update / add the appropriate `test_parity_quirks.py` entry.
3. Re-run `pytest -q -m parity` and `pytest -q -m smoke`.
4. If the change invalidates the stored fixtures, regenerate them via
   `testing/generate_reference_parity_fixtures.py` — and check in the new
   `.npz` files.
5. Re-run `testing/compare_long_run_parity.py --days 100` and confirm
   `Phi` agrees with reference SWAMPE to better than `~1e-6` max-fractional.
6. Update §3 in this file if the locked contract changed.

---

## 12. Sibling project: `gcmulator` (the ML emulator)

This package has one production downstream consumer in this workspace:
the `gcmulator` repository at `../gcmulator/`. It is **not** part of
`my_swamp`, but any non-trivial change here can break it. Read this
section before refactoring public APIs in `model.py`,
`spectral_transform.py`, or the forcing/diffusion modules.

### 12.1 What `gcmulator` is

A PyTorch-based emulator that learns a **direct-jump** transition
operator on the sphere:

```
(state0, params, transition_days)  →  state1  ≈  MY_SWAMP(state0, params, transition_days)
```

- Architecture: a Spherical Fourier Neural Operator (SFNO) from
  `torch_harmonics==0.8.1` (pinned), wrapped with a **FiLM
  conditioner** (`gcmulator.modeling.FiLMConditioner`) that injects the
  conditioning vector into each SFNO stage as per-channel scale/shift.
  Optional fixed big-skip via `residual_prediction=True`.
- State channels: `("Phi", "U", "V", "eta", "delta")` (the same five
  fields MY_SWAMP exposes as `last_state.{Phi,U,V,eta,delta}_curr`).
- Conditioning: 7 physical scalars
  `(a_m, omega_rad_s, Phibar, DPhieq, taurad_s, taudrag_s, g_m_s2)`
  plus a derived `log10_transition_days` channel.
- Loss: quadrature-weighted spherical MSE (`gcmulator.modeling.SphereLoss`,
  built on `torch_harmonics.examples.losses.get_quadrature_weights`).
- Training requires CUDA (asserted in `train_emulator`).

### 12.2 Two-stage workflow

Both stages use the same `config.json`. CLI:

```bash
# Stage 1: data generation (calls MY_SWAMP under the hood)
python -m gcmulator --gen --config config.json
# → writes data/raw_*/sim_NNNNNN.npy + manifest.json

# Stage 2: training (preprocess + fit, end-to-end)
python -m gcmulator --train --config config.json
# → writes models/<run>/{best,last}.pt + config_used.* + history.csv
```

The shipped `run.sh` (Slurm-friendly) wraps both stages, reinstalls
`my_swamp` and `torch_harmonics==0.8.1` from pinned package specs, and
chains gen→train conditionally on the dataset already existing.

### 12.3 How `gcmulator` couples to `my_swamp`

`gcmulator/src/gcmulator/my_swamp_backend.py` is the single integration
seam. Everything else in the emulator goes through it. The imports it
relies on:

| Import | Stability | Notes |
|---|---|---|
| `from my_swamp.model import RunFlags` | **Public** | OK to refactor only with a new field added compatibly. |
| `from my_swamp.model import run_model_scan` | **Public** | The data-generation entry point. Used with `return_history=False`, `donate_state=True`. |
| `from my_swamp.model import build_static` | **Public-ish** | Used by the diagnostic wind reconstruction. Currently exported from `model.py` but not in `__all__`. **If you remove or rename `build_static`, the emulator breaks at retrieval-time wind reconstruction.** |
| `from my_swamp.model import _step_once` | **Private** | Used to build a custom batched chunked scan inside `_get_reduced_carry_chunk_runner`. **Renaming or changing the signature of `_step_once` silently breaks `gcmulator` data generation.** |
| `from my_swamp.model import _step_once_state_only` | **Private** | Same hazard as `_step_once`. Used by the batched checkpoint runner. |
| `from my_swamp import spectral_transform as st` | **Public module** | Used to recompute winds from `(eta, delta)` via FFT + Legendre + `invrsUV`. |

**Treatment**: when you rename, restructure, or change the call
signature of `build_static`, `_step_once`, or `_step_once_state_only`,
either (a) keep a backward-compatible alias for one release, or (b)
update `gcmulator/src/gcmulator/my_swamp_backend.py` in the same PR.

`gcmulator` also calls `enforce_no_tpu_backend()`, which sets
`SWAMPE_JAX_ENABLE_X64=1` and strips `tpu` from `JAX_PLATFORMS`/
`JAX_PLATFORM_NAME` before any JAX import. If you change the env-var
contract in `my_swamp/__init__.py`, mirror the change in
`my_swamp_backend.py`.

### 12.4 Geometry contract

`gcmulator` stores all on-disk and in-memory state tensors in the
canonical orientation `(north→south, 0→2π)`. MY_SWAMP returns
`(south→north, -π→π)` from `state_var_init`. The bridge lives in
`gcmulator/src/gcmulator/geometry.py`:
`apply_geometry_state(state, flip_latitude_to_north_south=True,
roll_longitude_to_0_2pi=True)`.

If you change the latitude or longitude convention in MY_SWAMP
(`build_lambdas`, `gauss_legendre`, or `state_var_init`), the
emulator's geometry module will silently produce a permuted state
tensor and training will diverge in subtle ways. **Don't change those
conventions without coordinating.**

### 12.5 Internal-fixed parameters

`K6` and `K6Phi` are MY_SWAMP-side hyperdiffusion controls. The
emulator deliberately holds them fixed across all sims:

```python
INTERNAL_FIXED_K6 = 1.24e33     # gcmulator/sampling.py
INTERNAL_FIXED_K6PHI = None     # → MY_SWAMP inherits K6 for Phi diffusion
```

They are **not** part of the conditioning vector. If you change the
default `K6` in MY_SWAMP, you change the trained emulator's
out-of-distribution behavior — bump the dataset version
(`config.paths.dataset_dir`) so a fresh model gets trained.

### 12.6 Repo layout (sibling)

```
gcmulator/
├── config.json                  # default training config
├── pyproject.toml, requirements.txt, setup.py, run.sh, run.pbs
├── spec.md                      # detailed emulator design doc
├── src/gcmulator/
│   ├── __init__.py, __main__.py, main.py
│   ├── config.py                # typed config schema, JSON/YAML parsing
│   ├── data_generation.py       # --gen entry
│   ├── geometry.py              # north↔south / [-π,π)↔[0,2π) bridge
│   ├── modeling.py              # FiLM-SFNO + SphereLoss + coord channels
│   ├── my_swamp_backend.py      # *** the only integration seam ***
│   ├── normalization.py         # state + param + log10_transition_days stats
│   ├── sampling.py              # parameter draws + checkpoint schedules + live pair catalog
│   └── training.py              # preprocess + train + checkpointing
├── retrieval/
│   ├── README.md                # retrieval contract
│   ├── surrogate_backend.py     # TorchSurrogateRuntime (TorchScript loader)
│   └── run_surrogate_nss.py     # standalone inference benchmark
├── extra/
│   ├── pytorch_export.py        # checkpoint → TorchScript with embedded normalization
│   ├── predictions.py           # offline rollout / comparison vs MY_SWAMP
│   ├── swampe_parity_compare.py # emulator vs MY_SWAMP field comparison
│   ├── batch_size_benchmark.py
│   └── training_log.py
└── unit_tests/                  # 64 tests across 13 files
```

### 12.7 Tests in the sibling repo

`gcmulator` ships its own pytest suite (no shared markers with MY_SWAMP).
The 64 tests cover: config schema validation, geometry bridges, sampling
catalogs, normalization round-trip, training scheduler/logging,
modeling shapes (FiLM, big-skip, channel-weighted SphereLoss), and the
TorchScript retrieval contract.

`gcmulator/unit_tests/conftest.py` adds *both* `gcmulator/src/` and
`MY_SWAMP/src/` to `sys.path`, so emulator tests run against this
working tree's `my_swamp` (not the installed one). Keep that in mind
when running the emulator suite from a clean checkout — break this
project's `src/` and `gcmulator` tests will fail too.

### 12.8 Retrieval / inference path

Trained checkpoints can be exported to a self-contained TorchScript
bundle that no longer depends on `my_swamp`:

```bash
python gcmulator/extra/pytorch_export.py
# → models/<run>/model_export.torchscript.pt
# → models/<run>/model_export.meta.json
```

The exported module embeds the normalization tensors, so the runtime
contract is "physical state in, physical state out":
`forward(state0, params, transition_days) → state1`. The runtime
loader (`retrieval/surrogate_backend.py::TorchSurrogateRuntime`)
batches calls and applies `torch.jit.optimize_for_inference()`.

The retrieval path **does not load `my_swamp`** at inference time.
This is the supported way to use the emulator in downstream pipelines.
If a retrieval consumer reaches back into `my_swamp.model` directly,
that's a contract violation — flag it.

### 12.9 Common pitfalls when bridging the two

- **Don't change `_step_once` or `_step_once_state_only` signatures
  silently.** They look private but are imported by
  `gcmulator/src/gcmulator/my_swamp_backend.py`.
- **Don't change which fields are in `State`.** The reduced carry
  in `my_swamp_backend.ReducedCarrySnapshot` reads
  `Phi_curr, U_curr, V_curr, eta_curr, delta_curr,
  Phi_prev, eta_prev, delta_prev` directly from `last_state`. Renaming
  any of these fields breaks `gcmulator` checkpoint extraction.
- **Don't change geometry conventions** in MY_SWAMP without updating
  `gcmulator.geometry`. The emulator has no test for "MY_SWAMP returned
  the wrong orientation" — it would just train on a transposed sphere.
- **Don't tighten `K6` / `K6Phi` defaults** without bumping the
  emulator dataset name. The trained model has implicitly memorized the
  hyperdiffusion regime of its training data.
- **Don't relax the `donate_state=True` / `return_history=False`
  invariants** in `run_model_scan` — the emulator generates ~500
  trajectories × 100 days each per run, and donation is what keeps it
  inside GPU memory.
- **Don't change the `SWAMPE_JAX_ENABLE_X64` env-var name** without
  updating `enforce_no_tpu_backend()` in the emulator, or float64
  parity will silently degrade to float32 during data generation.
