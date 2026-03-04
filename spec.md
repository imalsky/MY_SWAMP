# MY_SWAMP Specification

Last updated: 2026-03-03 (America/Los_Angeles)
Project root: `/Users/imalsky/Desktop/SWAMPE_Project/MY_SWAMP`

This document is scoped to `MY_SWAMP` only.

## 1) Goal

`MY_SWAMP` is the JAX rewrite of SWAMPE. The project goal is to keep SWAMPE physics/numerics behavior for default runs while providing a differentiable, package-first implementation suitable for retrieval workflows.

Primary objectives:

1. Preserve reference behavior against legacy `SWAMPE` for default settings.
2. Keep the solver differentiable (JAX-first, scan-based time stepping).
3. Maintain stable, testable APIs for both history and terminal-state simulation.
4. Improve runtime efficiency without changing the locked parity behavior.

## 2) Scope

In scope:

1. `src/my_swamp/*` solver, transforms, forcing, filtering, and model drivers.
2. `tests/*` and `testing/*` parity/smoke/benchmark validation.
3. Retrieval-facing notebook utilities under `notebooks/` that consume `my_swamp`.

Out of scope for this spec:

1. `gcmulator` package architecture/training policy.
2. `torch-harmonics-main` library development.
3. Cross-repo orchestration scripts at workspace root.

## 3) Locked Parity Contract (Do Not Change Without Re-baselining)

The following behaviors are parity-critical and intentionally fixed:

1. Modified-Euler `Phi`/`Delta` effective `/4` coefficient behavior.
2. Modified-Euler `Delta` forced-term usage in both branches.
3. Modified-Euler `Eta` forced vs unforced asymmetry.
4. Explicit `Delta` carry-over-only update behavior.
5. Explicit `Eta`/`Delta` extra drag-linked forcing terms.
6. Dayside strict-inequality behavior in equilibrium forcing mask.
7. `Q < 0` clamp and `taudrag == -1` forcing semantics.
8. Legendre normalization/sign conventions used for transforms.
9. Inverse-Legendre negative-`m` layout and symmetry behavior.
10. Two-level initialization and Robert-Asselin update behavior.
11. Float64 parity mode requirement for reference-comparison runs.

## 4) Public API Contract

Key runtime APIs:

1. `my_swamp.model.run_model(...)`
2. `my_swamp.model.run_model_scan(...)`
3. `my_swamp.model.run_model_scan_final(...)`

Expected terminal state fields:

1. `Phi`
2. `U`
3. `V`
4. `eta`
5. `delta`

Initial-condition override support must remain available through explicit inputs (`eta0_init`, `delta0_init`, `Phi0_init`, `U0_init`, `V0_init`) where exposed by the scan drivers.

## 5) Retrieval Workflow Contract

Current retrieval workflow in `notebooks/nss.py` is steady-state oriented.

1. Terminal-map usage is intentional (`swamp_terminal_phi` path).
2. `map_projection_mode="shape_plus_amplitude"` keeps monopole amplitude.
3. `map_projection_mode="shape_only"` intentionally drops amplitude.
4. Adaptive convergence stopping in the outer loop is optional and config-driven.

## 6) Runtime and Precision Policy

1. Float64 mode is required for parity-grade comparisons (`SWAMPE_JAX_ENABLE_X64=1`).
2. Backend preflight checks are part of the supported runtime path.
3. CPU backend (`JAX_PLATFORMS=cpu`) is the default validation target for deterministic CI-style checks on this workspace.
4. Performance options (for example donation/remat toggles) are allowed only when behavior remains parity-safe.

## 7) Verification Baseline

Validation commands for this repository:

1. `JAX_PLATFORMS=cpu pytest --collect-only -q`
2. `JAX_PLATFORMS=cpu pytest -q`
3. `JAX_PLATFORMS=cpu pytest -q -m smoke`
4. `JAX_PLATFORMS=cpu pytest -q -m parity`
5. `JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=0 JAX_ENABLE_X64=0 pytest -q -m parity` (expected parity failures; validates x64 gate)
6. `ruff check src tests testing notebooks/nss.py`

## 8) Repository Map (MY_SWAMP Only)

Core package modules under `src/my_swamp/`:

1. `model.py` (main scan drivers and wrappers)
2. `spectral_transform.py` (quadrature, transforms, inversion helpers)
3. `time_stepping.py`, `modEuler_tdiff.py`, `explicit_tdiff.py`
4. `forcing.py`, `filters.py`, `initial_conditions.py`
5. `autodiff_utils.py`, `backend_preflight.py`, `plotting.py`

Validation paths:

1. `tests/` (packaging, smoke, parity regressions)
2. `testing/` (benchmark and fixture tooling)

## 9) Change Control

Any PR that changes locked numerical behavior must:

1. Explicitly call out the behavior change.
2. Update/add parity fixtures and regression tests.
3. Re-run parity + smoke markers.
4. Update this spec and `readme.md` sections affected by the change.
