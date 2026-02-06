# SWAMPE-JAX: Potential Remaining Changes / Open Technical Decisions

Document version: 2026-02-06

This is intentionally separate from the README. It focuses on unresolved decisions, remaining technical debt, and changes you may still want depending on whether your priority is (A) faithfulness to historical SWAMPE trajectories or (B) mathematical/JAX correctness.

---

## 1) Decide whether “faithfulness” or “correctness” is the top priority for modified-Euler

Current fixed behavior (in `modEuler_tdiff.py`):
- Uses `tstepcoeff1 /= 2` and `tstepcoeff2 /= 2` exactly once for all branches.
- This is mathematically consistent if `tstepcoeff1` was defined using `2*dt` and modified-Euler needs `dt`.

Historical SWAMPE / earlier port behavior:
- `phi_timestep` and `delta_timestep` effectively used “double-halving” (`tstepcoeff1/4`), because the first computed update was overwritten and both branches re-halved and recomputed.
- `eta_timestep` used inconsistent scaling between forced/unforced.

Implications:
- The fixed code will not reproduce historical trajectories for `expflag=False` unless you restore the old scaling.

If strict output matching is required:
- Add a clearly-named compatibility flag (e.g., `legacy_modeuler_coeff_scaling=True`) that restores the old effective coefficients.
- Default should be explicit and documented; do not silently switch behavior.

---

## 2) Make `model.py` fully traceable with respect to `test` (optional)

Current status:
- Inside the scan update, there is a Python branch:
  - `if test == 1: newU, newV = Uic, Vic`

This is fine if `test` is a Python int constant, but it prevents:
- treating `test` as a traced value
- vmap/jit across multiple test modes in a single compiled computation

If you want fully JAX-style control flow:
- Replace the Python branch with `jax.lax.cond(test == 1, ...)`.
- That requires representing `test` as a JAX scalar or passing a boolean `freeze_winds` flag instead.

---

## 3) Flags are currently forced to Python bools in `run_model_scan`

Current status:
- `RunFlags(forcflag=bool(forcflag), ...)` forces flags to be Python bools.
- This is simple and safe, but it prevents:
  - vmapping over different flag configurations as data
  - toggling flags inside a traced computation without recompilation

If you need more flexibility:
- Keep flags as `jnp.bool_` and make them part of the scan carry or part of a PyTree config passed into the JIT-compiled function.

Caveat:
- This will increase compilation complexity, and most physical applications do not need gradients with respect to these discrete switches.

---

## 4) Blowup stopping logic is data-dependent (piecewise gradients)

Current status:
- If RMS winds exceed a threshold, the scan enters a “dead” state and stops updating.
- This is implemented with `jax.lax.cond`.

This is differentiable along the taken branch, but introduces:
- a non-smooth boundary at the trigger threshold
- “gradient masking” after dead-state triggers (later steps don’t depend on earlier parameters)

If you need smooth optimization:
- disable the dead-state gate (or set `blowup_rms` extremely large) during gradient-based training, and enforce stability via a soft penalty term instead.

---

## 5) Remove `__pycache__/*.pyc` from the distributed archive

The fixed zip currently contains `__pycache__` files. These are not useful in source distribution and can confuse diffs/reviews.

Recommendation:
- exclude `__pycache__` when packaging/releasing.

---

## 6) Add trajectory-level regression tests against a reference SWAMPE run

Current tests validate spectral transforms, not end-to-end integration.

If you care about faithfulness:
- store a short reference trajectory (e.g., 10–50 steps) from numpy SWAMPE for each supported M and test mode
- compare time histories of key diagnostics (`rms`, min/max Phi, etc.) and/or a subset of fields

If you care about gradients:
- add a gradient-check test (finite differences vs `jax.grad`) for a small objective at low resolution.

---

## 7) Basis construction + caching is Python-side and not meant for JIT inside large traced programs

`Pmn/Hmn` and Gauss–Legendre nodes/weights are cached in Python dicts.
This is fine for typical usage (built once outside the scan), but it becomes limiting if you want:
- to JIT-compile a function where M/J are treated as dynamic
- to run many M values inside a single compiled program

If you need that:
- replace Python caching with a pure-JAX basis build (e.g., using `lax.scan`) and/or move basis precomputation outside the jitted function and pass in `Static` as an explicit argument.

---

## 8) Minor API/cleanup items

- `main_function.main(...)` accepts `use_scipy_basis` but the current codebase does not implement that option.
  - Either implement it (for debugging parity) or remove it to reduce confusion.
- Several legacy parameters are accepted for signature compatibility but unused (documented in `main_function.py`).
  - Consider explicitly warning on unused parameters if you want strictness.

