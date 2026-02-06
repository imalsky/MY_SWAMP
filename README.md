# SWAMPE-JAX (my_swamp)

A JAX rewrite of the SWAMPE spectral shallow-water model. This codebase keeps the original SWAMPE module structure and call signatures where practical, but rewires the driver around `jax.lax.scan` so at least one forward-simulation path is end-to-end differentiable.

Document version: 2026-02-06

---

## What this repository is

SWAMPE-JAX implements a barotropic (shallow-water) global spectral model on the sphere (triangular truncation M=N with Gaussian quadrature in latitude and FFT in longitude).

Key implementation traits:

- Spectral transforms (FFT + associated Legendre transforms) implemented in JAX (`jax.numpy`), including inverse transforms and wind inversion.
- Time integration performed in a side-effect-free loop using `jax.lax.scan` (see `my_swamp/model.py`), enabling autodiff through time.
- Compatibility wrappers preserve the legacy SWAMPE calling pattern (`my_swamp/main_function.py`, `my_swamp/model.py:run_model`).

---

## Typical use cases

1) Forward simulation (SWAMPE-like workflow)
- Run the model for `tmax` time steps and inspect end-state fields (`U`, `V`, `Phi`, `eta`, `delta`).
- Optional plotting and/or saving to disk (non-differentiable side-effect paths).

2) Differentiable simulation for inverse problems / sensitivity analysis
- Differentiate a scalar loss defined on the final state with respect to:
  - physical parameters (e.g., `DPhieq`, `taurad`, `taudrag`, `K6`, `Phibar`)
  - initial conditions (via `eta0_init`, `delta0_init`, `Phi0_init` in `run_model_scan`)
- Use cases: parameter estimation, gradient-based optimal control / nudging, training surrogate objectives, etc.

3) JIT-accelerated repeated runs
- Compile the scan-based simulation with `jax.jit` for repeated evaluation on CPU/GPU/TPU (subject to the caveats in “Differentiability and JIT caveats”).

---

## What’s in the package

Main modules:

- `my_swamp/model.py`
  - `run_model_scan(...)`: differentiable, side-effect-free run returning time histories (JAX arrays).
  - `run_model(...)`: SWAMPE-compatible wrapper that may plot/save; not designed for differentiation.
- `my_swamp/main_function.py`
  - SWAMPE-like `main(...)` signature and a small CLI (`python -m my_swamp.main_function ...`).
- `my_swamp/spectral_transform.py`
  - Gauss–Legendre quadrature, Pmn/Hmn basis construction, FFT truncation, forward/inverse Legendre transforms, wind inversion.
- `my_swamp/time_stepping.py`
  - High-level wrapper calling the selected time differencing scheme.
- `my_swamp/modEuler_tdiff.py`
  - Modified Euler time differencing (default).
- `my_swamp/explicit_tdiff.py`
  - Explicit scheme (optional, see notes on behavior change vs SWAMPE).
- `my_swamp/forcing.py`, `my_swamp/filters.py`, `my_swamp/initial_conditions.py`, `my_swamp/continuation.py`, `my_swamp/plotting.py`
  - Same conceptual roles as in SWAMPE.

Tests:

- `my_swamp/test_unit.py`: unit tests for spectral transforms and wind inversion (pytest-friendly).

---

## Requirements and environment

Minimum requirements (practical):
- Python 3.9+
- `jax`, `jaxlib`
- `numpy`
- `matplotlib` (only if you use plotting paths)

Precision:
- By default, the package enables JAX 64-bit mode at import time (see `my_swamp/__init__.py`).
- Disable x64 if you want faster GPU throughput and can tolerate numerical differences:
  - `SWAMPE_JAX_ENABLE_X64=0`

---

## How to run it

### Option A: run from a checkout/unzip (no packaging)

From the directory that contains the `my_swamp/` folder:

```bash
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 0 --no-plot
```

Notes:
- `--test 0` maps to “forced mode” (internally `test=None`).
- `--test 1` or `--test 2` runs idealized test cases.

### Option B: call from Python

Forward run (SWAMPE-compatible wrapper, includes optional plotting/saving and returns numpy arrays):

```python
from my_swamp.model import run_model

out = run_model(
    M=42,
    dt=600.0,
    tmax=200,
    Phibar=3.0e5,
    omega=7.292e-5,
    a=6.37122e6,
    test=None,          # forced mode
    forcflag=True,
    diffflag=True,
    modalflag=True,
    expflag=False,      # modified Euler (default)
    plotflag=False,
    saveflag=False,
    verbose=True,
)

U_final = out["U"]
V_final = out["V"]
Phi_final = out["Phi"]
```

Differentiable run (preferred for autodiff; returns JAX arrays and scan outputs):

```python
import jax
import jax.numpy as jnp
from my_swamp.model import run_model_scan

def loss_fn(DPhieq: float) -> jnp.ndarray:
    sim = run_model_scan(
        M=42,
        dt=600.0,
        tmax=200,
        Phibar=3.0e5,
        omega=7.292e-5,
        a=6.37122e6,
        test=None,
        forcflag=True,
        diffflag=True,
        modalflag=True,
        expflag=False,
        DPhieq=DPhieq,
        contflag=False,
    )
    # Example scalar objective: mean Phi at the last step
    Phi_last = sim["outs"]["Phi"][-1]   # (J,I)
    return jnp.mean(Phi_last)

dloss_dDPhieq = jax.grad(loss_fn)(4.0e6)
```

Important: `jax.grad` requires the loss to be a real scalar. Complex spectral coefficients are fine as intermediates.

---

## What EXACTLY changed relative to original SWAMPE (numpy)

**Mode note:** this distribution is **Option B (corrected physics)**: it prioritizes mathematically consistent
time-stepping and forcing over reproducing historical SWAMPE trajectories.

There are two categories of differences:

A) “Porting” differences: numpy → JAX rewiring (intended, mostly value-preserving)
B) “Post-review fixes”: deliberate changes after code review (behavior differences may exist)

### A) Porting differences (SWAMPE → SWAMPE-JAX)

1) Driver structure
- SWAMPE-JAX uses `jax.lax.scan` (in `model.py`) for the time loop so the forward simulation can be differentiated.
- Side effects (plotting, saving, continuation I/O) are kept outside the scan core.

2) Spectral basis / quadrature
- Replaces SciPy-based Gauss–Legendre quadrature with a JAX implementation (Golub–Welsch eigen decomposition).
- Replaces SciPy `lpmn`/special-function calls with a custom recurrence to build `Pmn/Hmn` in JAX.
- `Pmn/Hmn` and Gauss–Legendre nodes/weights are cached in Python dictionaries for reuse across runs.

3) Numerical kernels
- All core array math uses `jax.numpy` (JAX primitives), including FFTs (`jnp.fft.fft/ifft`) and `einsum`-based Legendre transforms.
- Several loops were vectorized (e.g., filters and initialization routines) to better match JAX performance patterns.

4) Forcing implementation style
- In-place numpy masking patterns were replaced with `jnp.where` to preserve JAX functional semantics.
- Some small numerical guards exist (e.g., tiny floors in denominators) to avoid accidental division-by-zero.

### B) Post-review fixes (changes vs original SWAMPE *and* vs the first JAX port)

These are the changes made after the SWAMPE-JAX code review and are present in this fixed version.

1) `explicit_tdiff.py` — explicit-scheme physics bug fixes (behavior change when `expflag=True`)
- The explicit divergence update now includes all computed terms:
  - `deltacomp1 + deltacomp2 + deltacomp3 + deltacomp4`
- Previously, only `deltacomp1` was applied (a likely inherited SWAMPE bug).
- Rayleigh drag is no longer double-counted in the explicit scheme forcing when `forcflag=True`.
  In this codebase, `forcing.Rfun` already includes drag in F and G (F = Ru - U/taudrag, G = Rv - V/taudrag),
  so the extra U/taudrag and V/taudrag terms that were being added in `explicit_tdiff` were removed.
- A one-time warning is emitted at runtime to make these trajectory changes explicit.

2) `modEuler_tdiff.py` — coefficient scaling and forcing consistency fixes (behavior change when `expflag=False`)
The modified-Euler implementation in this fixed version:
- Halves `tstepcoeff1` and `tstepcoeff2` exactly once (converting from a 2·dt convention to dt).
- Applies consistent coefficient scaling in both forced and unforced paths.
- Selects forced/unforced A/B terms with `jax.lax.select` instead of Python branching.

Compatibility note:
- This differs from the historical SWAMPE / earlier port behavior in which `phi_timestep` and `delta_timestep` effectively used a “double-halved” coefficient (equivalent to `tstepcoeff1/4`), and the forced-path `eta_timestep` used an un-halved coefficient.
- The new behavior is mathematically consistent with using dt rather than 2·dt inside the modified-Euler update, but it will not reproduce prior trajectories exactly. This is also surfaced via a one-time warning.

3) `time_stepping.py` — scheme dispatch made traceable
- Dispatch between explicit vs modified-Euler scheme now uses `jax.lax.cond`, so the choice can be traced if `expflag` is provided as a JAX boolean.

4) `__init__.py` — docstring / import hygiene
- The package docstring is placed at the top (so it is a real module docstring).
- x64 enabling is performed at import-time in a guarded way.

---

## Differentiability: what works, what breaks, and why

### “Good path” for differentiability

To keep a simulation differentiable end-to-end:
- Use `my_swamp.model.run_model_scan(...)`.
- Keep `contflag=False`, and do not call plotting/saving functions inside your differentiated function.
- Ensure the simulation does not trigger “blowup stopping” (see below).

On this path, the computation is composed of JAX primitives (FFTs, einsums, pointwise ops, `lax.scan`, `lax.cond`) and is differentiable with respect to real-valued scalar parameters and/or initial conditions.

### What can make it non-differentiable (exact cases)

1) Python-side side effects inside the differentiated function
- Any file I/O (loading/saving continuation states, pickles, numpy `.npz`, etc.)
- Plotting (matplotlib)
These occur in wrapper paths like `run_model(...)`, `continuation.py`, and `plotting.py`. They are intentionally outside the differentiable core.

Rule of thumb: differentiate through `run_model_scan`, not `run_model`.

2) Converting JAX values to Python scalars
Examples of operations that break tracing/autodiff if applied to tracers:
- `float(x)` / `int(x)` / `bool(x)` when `x` is a tracer
- `x.item()`
The driver tries to avoid this in the scan path, but if you add your own logic, this is a common failure mode.

3) Python control flow conditioned on JAX arrays
This breaks JIT/autodiff:
```python
if some_jax_array > 0:
    ...
```
The codebase mostly avoids this by using `jax.lax.cond` / `jnp.where`.
Remaining Python branches are on compile-time constants (e.g., `if test == 1:`), which is fine as long as `test` is a Python int and not a traced value.

4) Data-dependent stopping logic (piecewise gradients)
The scan uses a “blowup detection” gate:
- If RMS winds exceed `flags.blowup_rms`, the simulation enters a “dead” state and stops updating.
- This is implemented with `jax.lax.cond`, so it is still differentiable along the executed branch, but it introduces a non-smooth decision boundary.

If you need stable gradients, avoid running near this threshold (or set `blowup_rms` very large).

5) Differentiating with respect to discrete configuration
The model uses discrete values for:
- spectral resolution `M`
- boolean flags (`forcflag`, `diffflag`, `modalflag`, `expflag`)
Gradients with respect to these discrete choices are not meaningful.

6) Differentiating through static-basis construction (unusual use case)
`Pmn/Hmn` and Gaussian nodes/weights are built with Python-side caching and Python loops. This is not intended to be differentiated with respect to the quadrature nodes or the resolution. Normal model usage treats these as constants.

---

## “At least one fully differentiable branch”: what to use

If you want a clean, end-to-end differentiable forward model:
- Call `run_model_scan(...)` with:
  - `contflag=False`
  - no plotting/saving
  - parameter values that do not trigger blowup stopping

This is the intended fully differentiable branch.

---

## Notes on reproducibility and numerical parity

- This code enables 64-bit JAX mode by default for closer parity with numpy SWAMPE.
- Disabling x64 can change trajectories because spectral models are sensitive to precision.
- The post-review fixes intentionally change the modified-Euler and explicit-scheme behavior relative to historical SWAMPE.

---

## Running the unit tests

From the directory containing `my_swamp/`:

```bash
pytest -q
```

or

```bash
python -m my_swamp.test_unit
```

---

## Known limitations

- Only the `M` values explicitly supported in `initial_conditions.spectral_params` (42, 63, 106) are accepted.
- Some legacy parameters exist only for signature compatibility and are ignored (see `main_function.main(...)` docstring).
- The explicit scheme (`expflag=True`) now differs from historical SWAMPE because the divergence bug was fixed.

---

## Contact points in the code

If you need to modify the numerical core:
- Start at `my_swamp/model.py:simulate_scan` (the scan body).
- `my_swamp/time_stepping.py:tstepping` selects the scheme and runs `invrsUV`.
- `my_swamp/modEuler_tdiff.py` and `my_swamp/explicit_tdiff.py` implement the actual time updates.

