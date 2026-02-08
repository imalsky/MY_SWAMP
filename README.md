# SWAMPE-JAX (`my_swamp`)

A JAX rewrite of the SWAMPE spectral shallow-water model on the sphere. The numerical core runs inside `jax.lax.scan`, making the forward simulation end-to-end differentiable with respect to physical parameters and initial conditions.

Document version: 2026-02-08

---

## Table of Contents

1. [What This Code Does](#1-what-this-code-does)
2. [Package Layout](#2-package-layout)
3. [Requirements and Installation](#3-requirements-and-installation)
4. [Running the Model](#4-running-the-model)
5. [Using the Differentiable Path](#5-using-the-differentiable-path)
6. [Plotting and Visualization](#6-plotting-and-visualization)
7. [What Changed vs. NumPy SWAMPE (and What Didn't)](#7-what-changed-vs-numpy-swampe-and-what-didnt)
8. [Known Physics Quirks Preserved for Parity](#8-known-physics-quirks-preserved-for-parity)
9. [Candidate Corrected-Physics Changes](#9-candidate-corrected-physics-changes)
10. [Differentiability: What Works, What Breaks, and Why](#10-differentiability-what-works-what-breaks-and-why)
11. [GPU, Precision, and Performance Notes](#11-gpu-precision-and-performance-notes)
12. [Running the Unit Tests](#12-running-the-unit-tests)
13. [Known Limitations](#13-known-limitations)
14. [Code Navigation Guide](#14-code-navigation-guide)

---

## 1. What This Code Does

SWAMPE-JAX implements a barotropic (shallow-water) global spectral model on the sphere using triangular truncation (M = N) with Gaussian quadrature in latitude and FFT in longitude. The model solves the shallow-water equations for absolute vorticity (eta), divergence (delta), and geopotential (Phi) on a rotating sphere, with optional Newtonian relaxation forcing and Rayleigh drag.

Typical use cases:

- **Forward simulation** (SWAMPE-compatible): run for `tmax` time steps and inspect final fields (U, V, Phi, eta, delta), with optional plotting and saving.
- **Differentiable simulation**: compute gradients of a scalar loss (defined on the final state) with respect to physical parameters (`DPhieq`, `taurad`, `taudrag`, `K6`, `Phibar`, etc.) or initial conditions (`eta0`, `delta0`, `Phi0`). Useful for parameter estimation, sensitivity analysis, adjoint-based optimal control, and surrogate training.
- **JIT-accelerated repeated runs**: compile the scan-based simulation once with `jax.jit` for fast repeated evaluation on CPU, GPU, or TPU.

---

## 2. Package Layout

```
my_swamp/
├── __init__.py              # Package entry; enables float64 by default
├── _version.py              # Version string
├── dtypes.py                # Centralized dtype selection (float32/64)
├── model.py                 # Core driver: run_model_scan (differentiable),
│                            #   run_model (legacy wrapper), run_model_gpu
├── main_function.py         # CLI + legacy main() signature
├── spectral_transform.py    # Gauss–Legendre quadrature, Pmn/Hmn basis,
│                            #   FFT truncation, forward/inverse Legendre,
│                            #   wind inversion (invrsUV)
├── time_stepping.py         # Scheme dispatch (explicit vs modEuler),
│                            #   coefficient arrays, RMS wind diagnostic
├── modEuler_tdiff.py        # Modified-Euler time differencing
├── explicit_tdiff.py        # Explicit (leapfrog-style) time differencing
├── forcing.py               # Equilibrium geopotential, radiative forcing (Q),
│                            #   velocity forcing with Rayleigh drag (F, G)
├── filters.py               # Diffusion filters (sigma4, sigma6, modal splitting)
├── initial_conditions.py    # Analytic ICs, spectral params, ABCDE nonlinear terms
├── continuation.py          # Pickle I/O for save/load/continuation
├── plotting.py              # Matplotlib plotting + GIF generation
└── test_unit.py             # Spectral transform unit tests (pytest-friendly)
```

---

## 3. Requirements and Installation

**Minimum requirements:**

- Python 3.9+
- `jax` and `jaxlib` (CPU or GPU build)
- `numpy`

**Optional:**

- `scipy` — used for Gauss–Legendre quadrature and `lpmn` if available; the code falls back to pure NumPy/custom recurrence implementations when SciPy is absent.
- `matplotlib` — only needed for plotting paths.
- `imageio` — only needed for GIF generation.

**Precision:** By default, the package enables JAX 64-bit mode at import time (in `__init__.py`) for closest parity with NumPy SWAMPE. To disable this (for faster GPU throughput at the cost of numerical differences), set the environment variable before importing:

```bash
export SWAMPE_JAX_ENABLE_X64=0
```

**No installation step is required.** Just place the `my_swamp/` directory on your Python path (or run from its parent directory).

---

## 4. Running the Model

### 4a. Command line

From the directory containing the `my_swamp/` folder:

```bash
# Forced mode (test=0 maps to test=None internally)
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 0 --no-plot

# Idealized test case 1 (cosine bell advection)
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 1 --no-plot

# Idealized test case 2 (balanced zonal flow)
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 2 --no-plot
```

### 4b. From Python (SWAMPE-compatible wrapper)

`run_model(...)` preserves the legacy SWAMPE interface. It runs the differentiable scan internally, then optionally plots and saves results. Returns NumPy arrays by default.

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

# Final-step fields (NumPy arrays, shape (J, I))
U_final = out["U"]
V_final = out["V"]
Phi_final = out["Phi"]
eta_final = out["eta"]
delta_final = out["delta"]

# Diagnostic time series (shape (tmax, 2))
spinup = out["spinup"]    # [:,0] = min winds, [:,1] = RMS winds
geopot = out["geopot"]    # [:,0] = min Phi, [:,1] = max Phi

# Grid arrays
lambdas = out["lambdas"]  # longitudes in radians
mus = out["mus"]           # Gaussian latitudes (sin(phi))
```

### 4c. From Python (differentiable core)

`run_model_scan(...)` is the preferred entry point for autodiff. It returns JAX arrays and the full time history from `lax.scan`. See Section 5 for details.

### 4d. GPU-friendly wrapper

`run_model_gpu(...)` is a convenience wrapper around `run_model(...)` that defaults to `plotflag=False`, `saveflag=False`, `as_numpy=False`, `jit_scan=True`:

```python
from my_swamp.model import run_model_gpu

out = run_model_gpu(
    M=42, dt=600.0, tmax=200,
    Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
    test=None, forcflag=True,
)
```

---

## 5. Using the Differentiable Path

### 5a. Basic gradient computation

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
    # scalar objective: mean Phi at the last step
    Phi_last = sim["outs"]["Phi"][-1]   # (J, I)
    return jnp.mean(Phi_last)

# Compute gradient of the loss w.r.t. DPhieq
dloss = jax.grad(loss_fn)(4.0e6)
print("d(loss)/d(DPhieq) =", dloss)
```

### 5b. Differentiating with respect to initial conditions

```python
import jax
import jax.numpy as jnp
from my_swamp.model import run_model_scan
from my_swamp.initial_conditions import spectral_params

# Get grid dimensions for M=42
N, I, J, dt_default, lambdas, mus, w = spectral_params(42)

def loss_from_ic(Phi0: jnp.ndarray) -> jnp.ndarray:
    sim = run_model_scan(
        M=42, dt=600.0, tmax=50,
        Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
        test=None, forcflag=True, diffflag=True,
        modalflag=True, expflag=False,
        # Explicit initial conditions
        eta0_init=jnp.zeros((J, I)),
        delta0_init=jnp.zeros((J, I)),
        Phi0_init=Phi0,
    )
    return jnp.mean(sim["outs"]["Phi"][-1])

Phi0 = jnp.zeros((J, I))
grad_Phi0 = jax.grad(loss_from_ic)(Phi0)
print("Gradient shape:", grad_Phi0.shape)  # (64, 128)
```

### 5c. Differentiating with respect to multiple parameters

```python
import jax
import jax.numpy as jnp
from my_swamp.model import run_model_scan

def multi_param_loss(params):
    DPhieq, taurad, taudrag = params
    sim = run_model_scan(
        M=42, dt=600.0, tmax=100,
        Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
        test=None, forcflag=True, diffflag=True,
        modalflag=True, expflag=False,
        DPhieq=DPhieq, taurad=taurad, taudrag=taudrag,
    )
    return jnp.mean(sim["outs"]["Phi"][-1] ** 2)

params = jnp.array([4.0e6, 86400.0, 86400.0])
grads = jax.grad(multi_param_loss)(params)
print("Gradients:", grads)
```

### 5d. What `run_model_scan` returns

```python
result = run_model_scan(...)

result["static"]      # Static object (grid, basis, coefficients)
result["t_seq"]       # JAX array of time step indices
result["last_state"]  # State namedtuple at the final time step
result["starttime"]   # Integer start time

# Time histories (JAX arrays, leading axis = time):
outs = result["outs"]
outs["eta"]           # (T, J, I)  absolute vorticity
outs["delta"]         # (T, J, I)  divergence
outs["Phi"]           # (T, J, I)  geopotential perturbation
outs["U"]             # (T, J, I)  zonal wind
outs["V"]             # (T, J, I)  meridional wind
outs["rms"]           # (T,)       RMS winds
outs["spin_min"]      # (T,)       min wind speed
outs["phi_min"]       # (T,)       min geopotential
outs["phi_max"]       # (T,)       max geopotential
outs["dead"]          # (T,)       blowup detection flag
```

### 5e. Rules for keeping the simulation differentiable

1. **Use `run_model_scan`**, not `run_model`.
2. Set `contflag=False` (continuation uses Python file I/O, which breaks tracing).
3. Do not call plotting or saving functions inside your differentiated function.
4. Do not trigger blowup stopping (keep `blowup_rms` large or avoid divergent runs).
5. `jax.grad` requires the loss to be a **real scalar**. Complex intermediates are fine.
6. Do not wrap JAX tracers in `float()`, `int()`, `bool()`, or `.item()`.

---

## 6. Plotting and Visualization

The `my_swamp.plotting` module provides three main plot types and a GIF generator. Plotting is intentionally kept outside the differentiable core.

### 6a. Built-in plotting via `run_model`

The simplest way to plot is through the legacy wrapper:

```python
from my_swamp.model import run_model

out = run_model(
    M=42, dt=600.0, tmax=200,
    Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
    test=None, forcflag=True,
    plotflag=True,       # enable built-in plots
    plotfreq=10,         # plot every 10 steps
    minlevel=2.9e5,      # colorbar min for geopotential
    maxlevel=3.1e5,      # colorbar max for geopotential
)
```

This produces, at every `plotfreq` steps:
- A **mean zonal wind** profile (U averaged over all longitudes, plotted vs. latitude)
- A **geopotential quiver plot** (filled contour of Phi + Phibar with overlaid wind vectors)
- A **spinup diagnostic** plot (min wind speed and RMS winds vs. time)

### 6b. Manual plotting from `run_model_scan` output

For more control, run differentiably and plot afterward:

```python
import numpy as np
from my_swamp.model import run_model_scan
from my_swamp import plotting

sim = run_model_scan(
    M=42, dt=600.0, tmax=200,
    Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
    test=None, forcflag=True, diffflag=True,
    modalflag=True, expflag=False,
)

static = sim["static"]
outs = sim["outs"]

# Convert to numpy for plotting
lambdas = np.asarray(static.lambdas)
mus = np.asarray(static.mus)
U = np.asarray(outs["U"])
V = np.asarray(outs["V"])
Phi = np.asarray(outs["Phi"])

# Plot the final timestep
step = -1  # last step
Phibar = 3.0e5

plotting.mean_zonal_wind_plot(
    U[step], mus, timestamp="final", units="steps"
)

plotting.quiver_geopot_plot(
    U[step], V[step], Phi[step] + Phibar,
    lambdas, mus, timestamp="final", units="steps",
    minlevel=2.9e5, maxlevel=3.1e5,
)
```

### 6c. Generating GIFs

```python
import numpy as np
from my_swamp import plotting

# Assuming you have time-history arrays from run_model_scan:
Phibar = 3.0e5
timestamps = list(range(0, 200, 10))  # every 10 steps
indices = list(range(0, 200, 10))

plotting.write_quiver_gif(
    lambdas, mus,
    Phi[indices] + Phibar,          # (num_snapshots, J, I)
    U[indices],                      # (num_snapshots, J, I)
    V[indices],                      # (num_snapshots, J, I)
    timestamps,
    filename="simulation.gif",
    frms=5,                          # frames per second
    sparseness=4,                    # wind vector spacing
    minlevel=2.9e5,
    maxlevel=3.1e5,
    units="steps",
)
```

### 6d. Saving figures

All plot functions accept `savemyfig=True`, `filename="name.png"`, and optionally `custompath="path/"`:

```python
plotting.quiver_geopot_plot(
    U[step], V[step], Phi[step] + Phibar,
    lambdas, mus, timestamp="200",
    savemyfig=True,
    filename="geopot_200.png",
    custompath="my_plots/",
)
```

---

## 7. What Changed vs. NumPy SWAMPE (and What Didn't)

### Changes that are purely JAX porting (intended, value-preserving)

**Driver structure:** The time loop is a `jax.lax.scan` (in `model.py`) instead of a Python `for` loop. Side effects (plotting, saving, continuation I/O) are kept outside the scan.

**Spectral basis and quadrature:** SciPy-based Gauss–Legendre quadrature is replaced with a JAX-compatible implementation (Golub–Welsch eigen decomposition fallback). SciPy `lpmn` is replaced with a custom recurrence when SciPy is unavailable. Both produce results matching SciPy to floating-point roundoff.

**Numerical kernels:** All core array math uses `jax.numpy`, including FFTs (`jnp.fft.fft/ifft`) and `einsum`-based Legendre transforms. Several loops were vectorized. In-place NumPy masking in `forcing.py` was replaced with `jnp.where`.

**Blowup stopping:** NumPy SWAMPE uses `break` to exit the time loop when RMS winds exceed a threshold. Since `lax.scan` cannot break, the JAX version enters a "dead" (frozen) state and stops updating. Field evolution is identical up to the blowup point.

### What was NOT changed (legacy quirks preserved)

**This codebase is legacy-faithful.** It reproduces the reference NumPy SWAMPE behavior, including all known numerical quirks. See Section 8 for the full list.

> **Important note on README conflicts:** Earlier draft READMEs (particularly README.md shipped alongside this code) described this distribution as "Option B (corrected physics)" with bug fixes applied. **That description does not match the shipped code.** The actual `modEuler_tdiff.py` and `explicit_tdiff.py` in this archive implement legacy-faithful behavior. Similarly, a `legacy_modeuler_scaling` toggle described in some documentation does not exist in the shipped code. This unified README supersedes all prior documentation.

---

## 8. Known Physics Quirks Preserved for Parity

These behaviors exist in the reference NumPy SWAMPE and are faithfully reproduced in this JAX port. They are "quirks" from a physics/numerics standpoint, but they are part of strict trajectory parity.

### 8.1. Modified-Euler coefficient scaling (`expflag=False`)

`time_stepping.tstepcoeff(...)` returns coefficients defined with a `2*dt` convention. The modified-Euler scheme needs coefficients at a `dt` scale, but the reference SWAMPE's conversion is inconsistent:

- **phi_timestep** and **delta_timestep** effectively use `tstepcoeff1/4` and `tstepcoeff2/4` (a "double halving" due to control-flow overwrites in the original code).
- **eta_timestep** uses the **unscaled** `tstepcoeff1` when `forcflag=True`, and `tstepcoeff1/2` when `forcflag=False`.

This means different prognostic variables use different effective timestep scalings, which is inconsistent but matches the reference.

### 8.2. Modified-Euler delta uses forced terms even when unforced (`expflag=False`)

In `modEuler_tdiff.delta_timestep`, the code always uses `Bm + Fm` and `Am - Gm` (the forced expressions), even when `forcflag=False`. This is a copy-paste artifact from the original SWAMPE. It only affects runs with `forcflag=False`.

### 8.3. Explicit divergence drops computed components (`expflag=True`)

In `explicit_tdiff.delta_timestep`, the code computes `deltacomp2`, `deltacomp3`, and `deltacomp4` but then discards them, returning only `deltacomp1` (the carry-over from the previous time level). This drops key divergence tendencies.

### 8.4. Explicit scheme double-counts Rayleigh drag (`expflag=True`, `forcflag=True`)

`forcing.Rfun(...)` already includes Rayleigh drag in F and G: `F = Ru - U/taudrag`, `G = Rv - V/taudrag`. The explicit scheme forced branch in `explicit_tdiff.delta_timestep` and `explicit_tdiff.eta_timestep` adds additional `U/taudrag` and `V/taudrag` terms, effectively applying drag twice.

### 8.5. Modal splitting does not feed back into spectral state

The Robert–Asselin filter is applied to physical fields for diagnostics (phi_min/phi_max computation), but the spectral coefficients used for the next time step are not recomputed from the filtered fields. The filter is therefore largely diagnostic rather than dynamically active.

### 8.6. Forcing clamp for negative Q

In `forcing.Rfun`, negative `Q` values are clamped to zero (`Qclone = 0 where Q < 0`), and `Ru/Rv` are set to zero where `Q < 0`. This is non-smooth and non-physical but is part of the legacy behavior.

### 8.7. Unused `g` parameter in `Phieqfun`

`forcing.Phieqfun` accepts a `g` parameter that is not used in the function body. This is preserved for API compatibility.

---

## 9. Candidate Corrected-Physics Changes

If you want a "corrected physics" mode (at the cost of breaking trajectory parity with NumPy SWAMPE), the following changes are recommended. **None of these are applied in the shipped code.** Each should be implemented behind a clearly-named flag.

### 9.1. Fix explicit divergence update

In `explicit_tdiff.delta_timestep`, replace:
```python
deltamntstep = deltacomp1
```
with the intended combination:
```python
deltamntstep = deltacomp1 + deltacomp2 + deltacomp3 + deltacomp4
```
This restores all computed divergence tendency components.

### 9.2. Remove drag double-counting in the explicit scheme

Remove the extra `U/taudrag` and `V/taudrag` terms from `explicit_tdiff.delta_timestep._add_forcing` and `explicit_tdiff.eta_timestep._add_forcing`, since `forcing.Rfun` already includes drag.

### 9.3. Make modified-Euler coefficient scaling consistent

Replace the /4 and mixed scaling with a uniform /2 conversion:
```python
# In all three modEuler functions:
tstep1 = tstepcoeff1 / 2.0   # instead of /4 for phi/delta
tstep2 = tstepcoeff2 / 2.0   # instead of /4 for phi/delta
# For eta: always /2, regardless of forcflag
```
This gives mathematically consistent dt-scale coefficients for all variables.

### 9.4. Fix delta_timestep forced/unforced branching

In `modEuler_tdiff.delta_timestep`, use `jax.lax.select` to choose `Bm+Fm` vs `Bm` and `Am-Gm` vs `Am` based on `forcflag`, rather than always using forced terms.

### 9.5. Optionally activate modal splitting

Apply the Robert–Asselin filter to the prognostic spectral state (by recomputing spectral coefficients from the filtered physical fields), not just to diagnostic physical fields.

### 9.6. Replace Q < 0 clamp with a smooth alternative

Replace the hard `jnp.where(Q < 0, 0, ...)` clamp with a soft floor or physically motivated mass correction to avoid distorting momentum coupling and creating a non-smooth gradient landscape.

### 9.7. Add a `legacy_modeuler_scaling` toggle

To support both legacy-faithful and corrected modes, add a `legacy_modeuler_scaling: bool` parameter to `run_model_scan(...)` that is forwarded through `time_stepping.tstepping(...)` to the three `modEuler_tdiff` functions. When `True`, use the legacy /4 and mixed scaling; when `False`, use the consistent /2 scaling.

---

## 10. Differentiability: What Works, What Breaks, and Why

### What is differentiable

The core time integration in `model.simulate_scan()` is built around `jax.lax.scan` and uses JAX primitives throughout (`jax.numpy`, `jax.lax.cond`, `jax.lax.select`, FFTs, `einsum`). In the scan path (no plotting, no saving, no NumPy materialization), the simulation is differentiable with respect to:

- Continuous physical parameters: `dt`, `Phibar`, `omega`, `a`, `taurad`, `taudrag`, `K6`, `DPhieq`, `alpha`, and initial conditions (`eta0_init`, `delta0_init`, `Phi0_init`).
- Any smooth scalar objective defined on the scan outputs.

### What can break differentiability

**1. Python-side side effects.** File I/O (continuation, saving) and plotting (matplotlib) force host transfers and Python execution. These occur only in `run_model(...)`, not in the scan core. **Rule of thumb:** differentiate through `run_model_scan`, not `run_model`.

**2. Converting JAX tracers to Python scalars.** Calls like `float(x)`, `int(x)`, `bool(x)`, or `x.item()` on JAX tracers break tracing. The scan core avoids these, but user-added logic might introduce them.

**3. Python control flow conditioned on JAX arrays.** Writing `if some_jax_array > 0:` breaks JIT/autodiff. The codebase uses `jax.lax.cond` / `jnp.where` instead. Remaining Python branches (e.g., `if test == 1:`) are on compile-time Python constants, which is safe.

**4. Blowup gating (piecewise gradients).** The scan uses a `jax.lax.cond` gate: if RMS winds exceed `blowup_rms`, the state freezes. This is differentiable along the taken branch but introduces a non-smooth decision boundary. If you need stable gradients, set `blowup_rms` very large or disable the gate.

**5. Non-smooth diagnostics.** `jnp.min` and `jnp.max` used in diagnostics (`spin_min`, `phi_min`, `phi_max`) have undefined gradients where the argmin/argmax changes. If your loss depends on these, expect noisy gradients.

**6. Discrete configuration.** Spectral resolution `M` and boolean flags (`forcflag`, `diffflag`, etc.) are discrete. Gradients with respect to these are not meaningful.

**7. Static basis construction.** `Pmn/Hmn` and Gauss–Legendre nodes/weights are built with Python-side loops and caching. These are treated as constants during the scan and are not differentiated. This is the intended design.

### The "clean" differentiable path

Call `run_model_scan(...)` with `contflag=False`, no plotting/saving, and parameter values that do not trigger blowup stopping. This is the intended fully differentiable branch.

---

## 11. GPU, Precision, and Performance Notes

### GPU execution

The scan body uses JAX ops throughout and will place compute on GPU when you have GPU-enabled `jaxlib` and avoid host transfers (`as_numpy=False`, `saveflag=False`, `plotflag=False`). Use `run_model_gpu(...)` or `run_model_scan(..., jit_scan=True)`.

### Precision

The package defaults to float64 for parity with NumPy SWAMPE. On many GPUs, float64 is significantly slower. If performance matters more than exact parity, set `SWAMPE_JAX_ENABLE_X64=0` before import. Be aware that spectral models are sensitive to precision — disabling x64 will change trajectories.

### Batch/ensemble runs

For ensemble runs, precompute the `Static` object once via `model.build_static(...)`, build a batch of initial `State` objects (stack arrays along a leading batch axis), and `jax.vmap` a wrapper around `simulate_scan`. Adjust `in_axes` and ensure FFT/Legendre transforms operate on the correct axes.

### JIT caching

The `run_model_scan` function caches JIT-compiled variants keyed on `test` mode and `donate_state`. Changing `M`, `tmax`, or boolean flags triggers recompilation.

---

## 12. Running the Unit Tests

The unit tests validate spectral transforms and wind inversion, not end-to-end integration.

```bash
# From the directory containing my_swamp/
pytest -q

# Or run directly
python -m my_swamp.test_unit
```

Tests cover: spectral parameter initialization, `Pmn/Hmn` values at known points, forward Legendre transform of the Coriolis field, forward-then-inverse spectral round-trip, wind-to-vorticity-to-wind round-trip, and vorticity-to-wind-to-vorticity round-trip.

**Recommended additional tests** (not yet implemented):

- Trajectory-level regression tests against a short reference NumPy SWAMPE run (10–50 steps).
- Gradient checks: compare `jax.grad` output against finite differences for a small objective at low resolution.

---

## 13. Known Limitations

- **Supported resolutions:** Only M = 42, 63, and 106 are accepted (hardcoded in `initial_conditions.spectral_params`).
- **Test modes:** Only `test=None` (forced), `test=1`, and `test=2` are implemented. Other test codes from the original SWAMPE `main_function.py` are not supported.
- **Unused legacy parameters:** `main_function.main(...)` accepts `k1`, `k2`, `pressure`, `R`, `Cp`, `sigmaSB` for API compatibility but ignores them.
- **`use_scipy_basis` flag:** Accepted by the CLI but not implemented. Either implement it or remove it.
- **`__pycache__` in archive:** The distributed zip may contain `__pycache__/*.pyc` files. These are harmless but should be excluded from releases.
- **No `legacy_modeuler_scaling` toggle:** Despite references in some earlier documentation, this toggle does not exist in the shipped code. See Section 9.7 for how to implement it.

---

## 14. Code Navigation Guide

If you need to modify the numerical core, start here:

| What you want to change | Where to look |
|---|---|
| Time loop / scan body | `model.py` → `_step_once()`, `simulate_scan()` |
| Scheme selection (explicit vs. modEuler) | `time_stepping.py` → `tstepping()` |
| Modified-Euler update equations | `modEuler_tdiff.py` → `phi_timestep`, `delta_timestep`, `eta_timestep` |
| Explicit update equations | `explicit_tdiff.py` → same three functions |
| Forcing (Newtonian relaxation, drag) | `forcing.py` → `Phieqfun`, `Qfun`, `Rfun` |
| Spectral transforms (FFT, Legendre) | `spectral_transform.py` → `fwd_fft_trunc`, `fwd_leg`, `invrs_leg`, `invrs_fft`, `invrsUV` |
| Basis construction (Pmn, Hmn) | `spectral_transform.py` → `PmnHmn`, `_lpmn_fallback`, `_scaling_table` |
| Diffusion filters | `filters.py` → `sigma6`, `sigma6Phi`, `diffusion` |
| Initial conditions | `initial_conditions.py` → `state_var_init`, `velocity_init`, `ABCDE_init` |
| Nonlinear terms (A, B, C, D, E) | `initial_conditions.py` → `ABCDE_init` |
| Coefficient arrays | `time_stepping.py` → `tstepcoeff`, `tstepcoeff2`, `tstepcoeffmn`, `marray`, `narray` |
| Blowup detection | `model.py` → `_step_once()`, line `dead_next = ...` |
| Robert–Asselin filter | `model.py` → `_step_once()`, `apply_ra` inner function |
| Plotting | `plotting.py` → `mean_zonal_wind_plot`, `quiver_geopot_plot`, `spinup_plot`, `write_quiver_gif` |
| Save/load/continuation | `continuation.py` |
| CLI entry point | `main_function.py` → `cli_main()` |
