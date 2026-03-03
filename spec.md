# MY_SWAMP Specification

Last updated: 2026-03-03 (PST)

## Environment Note

All verification in this document was run from:

1. Conda environment: `swamp_compare`
2. Python: `3.10.19`
3. Project root: `/Users/imalsky/Desktop/SWAMPE_Project/MY_SWAMP`

Command used for env verification:

1. `conda run -n swamp_compare python -V`
2. `conda run -n swamp_compare which python`


## 1) Goal and Scope

This specification covers:

1. Core solver parity + performance in `src/my_swamp`
2. Retrieval forward-model behavior in `notebooks/nss.py`

Primary goals:

1. Preserve SWAMPE physical outputs (strict parity target for default settings).
2. Keep implementation differentiable and JAX-accelerated.
3. Improve runtime/compile efficiency without changing physics.
4. Support steady-state terminal-map retrieval workflows.


## 1.1) Locked Decisions (2026-03-02)

1. Strict parity scope is default settings only.
2. Parity reference is original `SWAMPE` outputs.
3. `model_days` remains user-controlled.


## 2) Definition of "Exact Same Results"

"Exact" means:

1. Same equations/behavior as reference SWAMPE.
2. Tolerance-based parity in float64 (`jax_enable_x64=True`) for matched inputs.
3. Retrieval-level parity (terminal map and projected light curve) within strict tolerances.

Bitwise identity across NumPy CPU and XLA backends is not required.


## 3) Spec Adherence Status (Code-Verified)

### 3.1 Requirements now satisfied

1. Backend preflight checks exist and are exercised in tests/benchmark entry points.
2. End-to-end parity regression tests versus trusted SWAMPE fixtures exist.
3. Parity mode is explicitly x64-gated in tests (parity tests fail in x32 mode by design).
4. Deterministic benchmark harness exists and reports compile/runtime/per-step metrics.
5. Retrieval-level parity checks for terminal-map projection are in the test suite.
6. High-impact efficiency updates were applied in `MY_SWAMP`:
   - static-flag branch specialization
   - batched FFT/Legendre transforms
   - weighted Legendre basis precompute
   - dead explicit-delta transform removal
   - broadcast replacements for prior `tile` patterns
   - redundant mask/real pass cleanup
7. Deprecated SciPy `lpmn` runtime path was replaced with `assoc_legendre_p_all` (with fallbacks), eliminating deprecation warnings in verification runs.
8. Full-project lint check now passes on the validated paths (`src`, `tests`, `testing`, `notebooks/nss.py`).
9. Previously-open performance correctness blockers were fixed:
   - `donate_state=True` no longer triggers XLA donation alias errors.
   - `remat_step=True` no longer fails for numeric `test` values.
   - NSS outer-loop stop checks no longer force host sync on every non-logging step.

### 3.2 Still open or intentionally deferred

1. Local GPU runtime validation is blocked on this machine (`jax-metal` reports no supported GPU).


## 4) Parity-Critical Contract (Do Not Change Behavior)

These behaviors remain locked for SWAMPE parity:

1. Modified-Euler Phi/Delta effective `/4` coefficient behavior.
2. Modified-Euler Delta forced-term usage in both branches.
3. Modified-Euler Eta forced/unforced asymmetry.
4. Explicit Delta carry-over-only update.
5. Explicit Eta/Delta extra drag-linked forcing terms.
6. Dayside strict inequality in equilibrium forcing mask.
7. `Q < 0` clamp and `taudrag == -1` semantics in forcing.
8. Legendre normalization/sign conventions.
9. Inverse Legendre negative-m layout and symmetry.
10. Two-level initialization and Robert-Asselin behavior.
11. Float64 parity mode for reference-comparison runs.


## 5) Retrieval Pipeline Clarifications (Steady-State)

1. Terminal `Phi` usage is intentional (`swamp_terminal_phi` path).
2. This is a steady-state retrieval pipeline, not transient fitting.
3. `map_projection_mode="shape_plus_amplitude"` preserves monopole amplitude.
4. `shape_only` intentionally discards amplitude.
5. Optional adaptive convergence stopping is implemented and validated in config checks.
6. Saved run artifacts include projection/integration metadata and identifiability diagnostics.


## 6) Executed Verification Matrix (Environment + Model)

All commands below were run in `swamp_compare` on 2026-03-03 with `JAX_PLATFORMS=cpu` for stability.

### 6.1 Test collection (what exists)

1. `JAX_PLATFORMS=cpu conda run -n swamp_compare pytest --collect-only -q`
2. Result: 20 tests collected, including:
   - Environment/backend tests: `tests/test_backend_preflight.py`
   - Core model smoke: `tests/test_model_scan_smoke.py`
   - Static/grid checks: `tests/test_static_spectral_params.py`
   - Legacy transform/unit suite: `src/my_swamp/test_unit.py`
   - Parity quirk regressions: `tests/test_parity_quirks.py`
   - SWAMPE fixture parity regressions: `tests/test_parity_reference_regression.py`

### 6.2 Full suite

1. `JAX_PLATFORMS=cpu conda run -n swamp_compare pytest -q`
2. Result: `20 passed`

### 6.3 Marker suites

1. `JAX_PLATFORMS=cpu conda run -n swamp_compare pytest -q -m smoke`
2. Result: `5 passed, 15 deselected`
3. `JAX_PLATFORMS=cpu conda run -n swamp_compare pytest -q -m parity`
4. Result: `4 passed, 16 deselected`

### 6.4 x64 parity gating check

1. `JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=0 JAX_ENABLE_X64=0 conda run -n swamp_compare pytest -q -m parity`
2. Result: `4 failed` with explicit assertion requiring float64 mode.

### 6.5 Benchmark harness

1. `JAX_PLATFORMS=cpu conda run -n swamp_compare python testing/benchmark_scan.py --backend cpu --M 42 --dt 30 --tmax 30 --test 1 --forcflag false --diffflag false --timed-runs 2 --warmup-runs 1`
2. Result snapshot:
   - `result.compile_seconds=1.281907`
   - `result.runtime_median_seconds=0.048284`
   - `result.per_step_median_ms=1.724434`
3. Retrieval-oriented knob sweep (`tmax=120`, sequential runs):
   - baseline (`diagnostics=false, donate=false, remat=false`): `per_step_median_ms=1.425232`
   - `donate=true`: `per_step_median_ms=1.427316`
   - `remat=true`: `per_step_median_ms=1.432047`
4. Interpretation for this CPU runtime:
   - `donate` and `remat` are now stable but do not improve throughput at this workload scale.
   - Keep `diagnostics=false` for retrieval/inference loops unless explicit blow-up diagnostics are required.

### 6.6 Lint/dead-code checks

1. `conda run -n swamp_compare ruff check src tests testing notebooks/nss.py --select F401,F841`
2. Result: `All checks passed`
3. `conda run -n swamp_compare ruff check src tests testing notebooks/nss.py`
4. Result: `All checks passed`
5. `conda run -n swamp_compare python -m vulture src tests testing notebooks/nss.py`
6. Result: `no findings` after explicit ignore-list triage for known API hooks/callback surfaces.


## 7) Correctness Assessment

Given the current test matrix and parity fixtures:

1. Default-path behavior is correct relative to the documented parity target.
2. Retrieval projection behavior is covered by reference tests.
3. No correctness regressions were observed after latest cleanup edits.

Boundaries:

1. GPU correctness/performance validation remains pending on hardware with a working GPU backend.


## 8) Dead-Code Assessment

Removed in previous passes:

1. Unused import in `src/my_swamp/continuation.py`.
2. Unused local in `notebooks/nss.py` (`prior_width`).
3. Unused variable assignment in `src/my_swamp/plotting.py` (`colorbar` binding only).

Removed in latest pass:

4. Unused `_SCIPY_IMPORT_ERROR` variable in `src/my_swamp/spectral_transform.py`.
5. Dead `modal_splitting` function in `src/my_swamp/filters.py` (Robert-Asselin filter is computed inline in `model.py`).
6. Unused `INT_DTYPE`, `as_real`, `as_complex` from `src/my_swamp/dtypes.py`.
7. Extra blank lines and trailing whitespace in core source files.

Dead-code scan is now clean on the validated paths after explicit triage of known API-hook/callback symbols.


## 9) Known Intentional Divergences

1. Continuation timestamp call-ordering fix in `model.py`.
2. `invrsUV_with_coeffs` optimization helper to avoid redundant FFT work.
