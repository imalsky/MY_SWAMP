# SWAMPE-JAX (`my_swamp`)

A JAX rewrite of the SWAMPE spectral shallow‑water model on the sphere. The numerical core runs inside `jax.lax.scan`, so the forward simulation is end‑to‑end differentiable with respect to continuous physical parameters and explicit initial conditions.

Document version: 2026-02-08

---

## Table of Contents

1. [What This Code Does](#1-what-this-code-does)  
2. [Package Layout](#2-package-layout)  
3. [Requirements and Installation](#3-requirements-and-installation)  
4. [Running the Model](#4-running-the-model)  
5. [Differentiable Simulation API](#5-differentiable-simulation-api)  
6. [Plotting and Visualization](#6-plotting-and-visualization)  
7. [Behavior Relative to NumPy SWAMPE](#7-behavior-relative-to-numpy-swampe)  
8. [Legacy Physics and Numerics Preserved for Parity](#8-legacy-physics-and-numerics-preserved-for-parity)  
9. [Physics and Numerics Changes Not Implemented Here](#9-physics-and-numerics-changes-not-implemented-here)  
10. [Differentiability Scope and Caveats](#10-differentiability-scope-and-caveats)  
11. [GPU, Precision, and Performance Notes](#11-gpu-precision-and-performance-notes)  
12. [Running the Unit Tests](#12-running-the-unit-tests)  
13. [Known Limitations](#13-known-limitations)  
14. [Code Navigation Guide](#14-code-navigation-guide)  

---

## 1. What This Code Does

SWAMPE-JAX implements a single‑layer (barotropic) global spectral shallow‑water model on the sphere using triangular truncation (M = N), Gaussian quadrature in latitude, and FFT in longitude. The model advances:

- `eta`: absolute vorticity (relative vorticity plus Coriolis)
- `delta`: divergence
- `Phi`: geopotential-like perturbation field (often interpreted as a `gH`-like quantity in shallow-water formulations)

Winds (`U`, `V`) are diagnosed from (`eta`, `delta`) via spectral inversion.

Two time-stepping schemes are available (selected by `expflag`):

- `expflag=False`: modified‑Euler scheme (`my_swamp/modEuler_tdiff.py`)
- `expflag=True`: explicit scheme (`my_swamp/explicit_tdiff.py`)

Optional physical/numerical terms:

- Newtonian relaxation of `Phi` toward a prescribed equilibrium `Phieq` (forced mode only, `test=None`)
- Rayleigh drag on winds (disabled when `taudrag == -1`)
- Implicit spectral hyperdiffusion filters (`sigma6`, `sigma6Phi`)
- Robert–Asselin / “modal splitting” filter (`modalflag`, `alpha`) applied in the driver as in SWAMPE

The time loop is implemented using `jax.lax.scan` in `my_swamp/model.py`, which is the basis for differentiability and JIT acceleration.

---

## 2. Package Layout

Reference implementation:

- `SWAMPE copy/`  
  Original NumPy/SciPy SWAMPE code (used as the parity baseline).

JAX implementation:

```
my_swamp/
├── __init__.py              # Package entry; enables float64 by default (configurable)
├── _version.py              # Version string
├── dtypes.py                # Centralized dtype selection (float32/64)
├── model.py                 # Core driver: run_model_scan (differentiable),
│                            #   run_model (wrapper), run_model_gpu
├── main_function.py         # CLI + legacy main() signature
├── spectral_transform.py    # Gauss–Legendre quadrature, Pmn/Hmn basis construction,
│                            #   FFT truncation, forward/inverse Legendre transforms,
│                            #   wind inversion (invrsUV)
├── time_stepping.py         # Scheme dispatch (explicit vs modEuler),
│                            #   coefficient arrays, RMS wind diagnostic
├── modEuler_tdiff.py        # Modified‑Euler time differencing (parity behavior)
├── explicit_tdiff.py        # Explicit time differencing (parity behavior)
├── forcing.py               # Phieq, radiative forcing Q, velocity forcing R (incl. drag + clamp)
├── filters.py               # Diffusion filters + diffusion operator
├── initial_conditions.py    # Supported resolutions, analytic ICs, nonlinear term construction
├── continuation.py          # Pickle I/O for save/load/continuation
├── plotting.py              # Matplotlib plotting helpers + GIF generation
└── test_unit.py             # Unit tests for transforms and wind inversion (pytest-friendly)
```

---

## 3. Requirements and Installation

Minimum requirements:

- Python 3.9+
- `jax`, `jaxlib`
- `numpy`

Optional dependencies:

- `scipy`  
  Used when available for Gauss–Legendre nodes/weights and associated Legendre polynomials. The code falls back to NumPy + recurrence implementations when SciPy is unavailable.
- `matplotlib` and `imageio`  
  Required only for plotting/GIF utilities.

Precision configuration:

- By default, this package enables JAX 64‑bit mode at import time for closer parity with NumPy SWAMPE.
- The environment variable `SWAMPE_JAX_ENABLE_X64` controls this behavior (read during import of `my_swamp`):

```bash
export SWAMPE_JAX_ENABLE_X64=1   # enable float64/complex128 (default behavior)
export SWAMPE_JAX_ENABLE_X64=0   # disable and use float32/complex64
```

Installation:

- No packaging/installation step is required. Running from a checkout/unzip with the `my_swamp/` directory on `PYTHONPATH` is sufficient.

---

## 4. Running the Model

### 4a. Command line

From the directory that contains the `my_swamp/` package folder:

```bash
# Forced mode (test=0 maps internally to test=None)
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 0 --no-plot

# Idealized test case 1
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 1 --no-plot

# Idealized test case 2
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 2 --no-plot
```

CLI defaults (from `my_swamp/main_function.py`):

- Saving is enabled by default (writes pickles under `data/`). Use `--no-save` to disable.
- Plotting is disabled by default. Use `--plot` to enable.
- `--plotfreq` controls plotting cadence.

### 4b. Python wrapper (SWAMPE-compatible)

Use `run_model(...)` for a SWAMPE-style workflow, including optional plotting/saving.

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
eta_final = out["eta"]
delta_final = out["delta"]

spinup = out["spinup"]    # (tmax, 2)
geopot = out["geopot"]    # (tmax, 2)
lambdas = out["lambdas"]  # (I,) longitudes [rad]
mus = out["mus"]          # (J,) sin(latitude)
```

### 4c. Differentiable driver (scan core)

Use `run_model_scan(...)` for differentiable/JIT-friendly execution. See Section 5.

### 4d. GPU/AD-friendly wrapper

`run_model_gpu(...)` is a convenience wrapper around `run_model(...)` that defaults to:

- `plotflag=False`
- `saveflag=False`
- `as_numpy=False`
- `jit_scan=True`

```python
from my_swamp.model import run_model_gpu

out = run_model_gpu(
    M=42, dt=600.0, tmax=200,
    Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
    test=None, forcflag=True,
)
```

---

## 5. Differentiable Simulation API

### 5a. Basic gradient computation

`run_model_scan(...)` returns JAX arrays and does not perform side effects inside the time loop. A scalar objective can be differentiated with `jax.grad`.

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
        jit_scan=True,
    )
    Phi_last = sim["outs"]["Phi"][-1]   # (J, I)
    return jnp.mean(Phi_last)           # real scalar objective

dL_dDPhieq = jax.grad(loss_fn)(4.0e6)
```

### 5b. Differentiating with respect to initial conditions

Explicit initial fields override analytic initialization and continuation loading.

```python
import jax
import jax.numpy as jnp
from my_swamp.model import run_model_scan
from my_swamp.initial_conditions import spectral_params

N, I, J, _, lambdas, mus, w = spectral_params(42)

def loss_from_ic(Phi0: jnp.ndarray) -> jnp.ndarray:
    sim = run_model_scan(
        M=42, dt=600.0, tmax=50,
        Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
        test=None,
        Phi0_init=Phi0,
        eta0_init=jnp.zeros((J, I)),
        delta0_init=jnp.zeros((J, I)),
        contflag=False,
    )
    return jnp.mean(sim["outs"]["Phi"][-1])

Phi0 = jnp.zeros((J, I))
dL_dPhi0 = jax.grad(loss_from_ic)(Phi0)
```

### 5c. Differentiating with respect to multiple parameters

```python
import jax
import jax.numpy as jnp
from my_swamp.model import run_model_scan

def multi_param_loss(params: jnp.ndarray) -> jnp.ndarray:
    DPhieq, taurad, taudrag = params
    sim = run_model_scan(
        M=42, dt=600.0, tmax=100,
        Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
        test=None, forcflag=True, diffflag=True,
        modalflag=True, expflag=False,
        DPhieq=DPhieq, taurad=taurad, taudrag=taudrag,
        contflag=False,
    )
    return jnp.mean(sim["outs"]["Phi"][-1] ** 2)

params = jnp.array([4.0e6, 86400.0, 86400.0])
grads = jax.grad(multi_param_loss)(params)
```

### 5d. Return structure and time indexing

`run_model_scan(...)` returns a dictionary:

- `static`: basis, grid, coefficients, filters (treated as constants by the scan)
- `t_seq`: time indices advanced by the scan (default `arange(2, tmax)`)
- `outs`: time history dictionary (leading axis = time)
- `last_state`: scan carry at the final step
- `starttime`: integer start time index (default `2`)

`outs` contains:

- `outs["eta"], outs["delta"], outs["Phi"], outs["U"], outs["V"]`: fields produced by each time step
- `outs["rms"], outs["spin_min"], outs["phi_min"], outs["phi_max"]`: diagnostics computed from the “current/middle” physical level inside `_step_once` (including Robert–Asselin filtering when enabled)
- `outs["dead"]`: blowup gating flag
- `outs["t"]`: the time index associated with each output sample

Two-level initialization:

- Time levels 0 and 1 are identical (the initial conditions).
- The default scan starts at `t=2`, matching SWAMPE’s loop structure.

Legacy diagnostic arrays in `run_model(...)`:

- `run_model(...)` reconstructs `spinup` and `geopot` arrays of shape `(tmax, 2)`.
- These arrays are filled at indices `(t-1)` for scan times `t` in `t_seq`.
- With default start time 2, index 0 remains 0.0 unless separately filled.

---

## 6. Plotting and Visualization

Plotting utilities live in `my_swamp/plotting.py` and use matplotlib and imageio.

Plotting is intentionally outside the differentiable scan body.

### 6a. Built-in plotting via `run_model(...)`

```python
from my_swamp.model import run_model

out = run_model(
    M=42, dt=600.0, tmax=200,
    Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
    test=None, forcflag=True,
    plotflag=True,
    plotfreq=10,
    minlevel=2.9e5,
    maxlevel=3.1e5,
    saveflag=False,
)
```

### 6b. Manual plotting from `run_model_scan(...)` output

```python
import numpy as np
from my_swamp.model import run_model_scan
from my_swamp import plotting

sim = run_model_scan(
    M=42, dt=600.0, tmax=200,
    Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
    test=None, forcflag=True, diffflag=True,
    modalflag=True, expflag=False,
    contflag=False,
)

static = sim["static"]
outs = sim["outs"]

lambdas = np.asarray(static.lambdas)
mus = np.asarray(static.mus)
U = np.asarray(outs["U"])
V = np.asarray(outs["V"])
Phi = np.asarray(outs["Phi"])

step = -1
Phibar = 3.0e5

plotting.mean_zonal_wind_plot(U[step], mus, timestamp="final", units="steps")
plotting.quiver_geopot_plot(U[step], V[step], Phi[step] + Phibar, lambdas, mus, timestamp="final", units="steps")
```

### 6c. GIF generation

```python
from my_swamp import plotting

Phibar = 3.0e5
indices = list(range(0, 200, 10))
timestamps = indices

plotting.write_quiver_gif(
    lambdas, mus,
    Phi[indices] + Phibar,
    U[indices],
    V[indices],
    timestamps,
    filename="simulation.gif",
    frms=5,
    sparseness=4,
    minlevel=2.9e5,
    maxlevel=3.1e5,
    units="steps",
)
```

Default output directories:

- Plot helpers write to `plots/` by default when saving figures.
- Continuation/save helpers write to `data/` by default.

---

## 7. Behavior Relative to NumPy SWAMPE

The JAX implementation preserves the reference SWAMPE stepping behavior closely, including several legacy quirks that affect trajectories (Section 8).

Additional differences arise from the JAX driver architecture.

Driver structure:

- The time loop is implemented using `jax.lax.scan` rather than a Python `for` loop.
- Plotting/saving/continuation I/O are performed outside the scan.

Blowup stopping semantics:

- Reference SWAMPE exits early using `break` when RMS winds exceed a threshold.
- `lax.scan` cannot break; the JAX version enters a “dead/frozen” state and stops updating the dynamics. Outputs remain constant for subsequent steps.

Basis and portability:

- Gaussian quadrature uses SciPy (`roots_legendre`) when available and falls back to NumPy (`leggauss`) otherwise.
- Associated Legendre values use SciPy (`lpmn`) when available and fall back to a recurrence-based implementation otherwise.
- Basis construction occurs during `build_static(...)` and is treated as constant data during the scan.

Forcing computation edge case (`forcflag=False` in forced mode):

- In the reference driver, forcing fields may still be computed upstream in forced mode, and kernels decide whether to apply them.
- In the JAX driver, `_forcing_phys` returns zero forcing fields when `forcflag` is disabled.

Saved-file timestamp naming:

- The JAX wrapper uses `compute_timestamp(units, t, dt)` (matching `continuation.py`).
- The reference SWAMPE driver calls this function with swapped `(t, dt)` arguments, producing different filenames.

---

## 8. Legacy Physics and Numerics Preserved for Parity

These behaviors exist in the reference NumPy SWAMPE and are reproduced here. They are trajectory‑relevant.

1. Modified‑Euler coefficient scaling is internally inconsistent  
   In `modEuler_tdiff.py`:
   - `Phi` and `delta` use an effective `/4` conversion of the `2*dt` coefficient convention.
   - `eta` uses unscaled `tstepcoeff1` when `forcflag=True` and `/2` when `forcflag=False`.

2. Modified‑Euler divergence uses forced-form A/B terms even when unforced  
   In `modEuler_tdiff.delta_timestep`, the A/B coupling uses `(Bm + Fm)` and `(Am - Gm)` even when `forcflag=False`.

3. Explicit divergence tendency terms are dropped  
   In `explicit_tdiff.delta_timestep`, components 2/3/4 are computed but discarded; the update uses only the carry-over term.

4. Explicit scheme applies drag-like terms in addition to `Rfun` drag  
   `forcing.Rfun` includes Rayleigh drag when `taudrag != -1`. The explicit scheme adds additional terms proportional to `U/taudrag` and `V/taudrag`.

5. Modal splitting does not feed back into the spectral prognostic state  
   Robert–Asselin filtering is applied to the physical “previous” level used for diagnostics/state bookkeeping, while the spectral coefficients advanced by the step are not recomputed from filtered physical fields.

6. Forcing clamp for negative `Q`  
   In `forcing.Rfun`, negative `Q` is clamped to zero and momentum tendencies are zeroed where `Q < 0`.

7. Spectral transform conventions  
   `Pmn/Hmn` normalization uses factorial-based scaling and includes an additional sign flip for odd `m` to match SWAMPE’s convention. `invrsUV` zeros the `n=0` modes of `eta` and `delta` prior to inversion.

---

## 9. Physics and Numerics Changes Not Implemented Here

This section describes common changes that produce a more internally consistent shallow‑water solver at the cost of breaking strict SWAMPE trajectory parity. These changes are not applied in this code snapshot.

1. Explicit scheme: include the full divergence tendency  
   Replace the explicit divergence update with the sum of all computed components.

2. Explicit scheme: remove drag double-counting  
   Apply Rayleigh drag in exactly one location (either `forcing.Rfun` or explicit forcing expressions), not both.

3. Modified‑Euler: make coefficient scaling consistent  
   Replace the mixed `/4`, `/2`, and branch-dependent scalings with one documented discretization.

4. Modified‑Euler: correct unforced `delta` A/B coupling  
   Use `Bm` and `Am` when `forcflag=False`, rather than forced-form expressions.

5. Modal splitting: make filtering dynamically effective  
   Apply Robert–Asselin filtering to the prognostic state used by the next step (including spectral coefficients), not only to stored physical fields.

6. Replace the hard `Q < 0` clamp with a smoother correction  
   A smooth limiter or a physically motivated correction reduces non-smoothness and can improve gradient-based inference behavior.

---

## 10. Differentiability Scope and Caveats

The intended differentiable execution path is:

- `run_model_scan(..., contflag=False, jit_scan=True)` inside a scalar loss,
- `jax.grad(loss_fn)` or `jax.value_and_grad(loss_fn)`.

Differentiation is supported with respect to:

- Continuous scalar parameters that enter the scan (e.g., `DPhieq`, `taurad`, `taudrag`, `K6`, `K6Phi`, `Phibar`, `omega`, `a`, `dt`, `alpha`).
- Initial conditions, when passed explicitly as `eta0_init`, `delta0_init`, `Phi0_init` (and optionally `U0_init`, `V0_init`).

Discrete configuration is not a meaningful differentiation target:

- `M`
- boolean flags (`forcflag`, `diffflag`, `modalflag`, `expflag`)
- test mode selection

Non-smooth or piecewise components (AD works, but gradients can be kinked/noisy):

- `min/max` diagnostics (`phi_min`, `phi_max`, `spin_min`)
- `Q < 0` clamp in `forcing.Rfun`
- blowup gating threshold branch

Side effects and host materialization:

- Plotting, saving, and continuation I/O are intentionally outside the scan and should not be executed inside a differentiated function.

---

## 11. GPU, Precision, and Performance Notes

GPU execution:

- The scan body (FFTs, `einsum`, pointwise ops) runs on accelerators when GPU/TPU-enabled `jaxlib` is installed and host transfers are avoided.
- `run_model_gpu(...)` defaults to the configuration most compatible with accelerator execution.

Precision:

- The package defaults to float64 for closer parity with NumPy SWAMPE.
- Float32 mode increases throughput (especially on GPUs) but changes trajectories due to spectral sensitivity to roundoff.

JIT recompilation behavior:

- JIT compilation specializes on array shapes and Python booleans. Changing `M`, `tmax`, or flags typically triggers recompilation.

Ensembles and vectorization:

- Static basis construction occurs once in `build_static(...)`.
- Vectorized/ensemble execution typically wraps `run_model_scan` in `jax.vmap` over batched initial conditions or parameters, with care taken to preserve array axis conventions.

---

## 12. Running the Unit Tests

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q
```

Notes:

- The package defaults to enabling JAX 64-bit mode for closer parity with the NumPy/SciPy SWAMPE reference.  
  To run tests in 32-bit mode (faster, lower memory), set:
  ```bash
  export SWAMPE_JAX_ENABLE_X64=0
  ```

- You can also run the legacy spectral-transform test module directly:
  ```bash
  python -m my_swamp.test_unit
  ```

---

## 13. Known Limitations

- Supported resolutions are limited to `M in {42, 63, 106}` as defined in `initial_conditions.spectral_params`.
- Supported test modes are `test=None` (forced), `test=1`, and `test=2`.
- The CLI accepts `--use-scipy-basis`, but the current code does not wire this flag into basis construction; it is a placeholder.
- Continuation saving defaults to `data/` (relative to the working directory) and plotting defaults to `plots/`.

---

## 14. Code Navigation Guide

| Topic | Primary locations |
|---|---|
| Time loop / scan body | `model.py` → `_step_once()`, `simulate_scan()` |
| Scheme selection | `time_stepping.py` → `tstepping()` |
| Modified‑Euler update equations | `modEuler_tdiff.py` → `phi_timestep`, `delta_timestep`, `eta_timestep` |
| Explicit update equations | `explicit_tdiff.py` → `phi_timestep`, `delta_timestep`, `eta_timestep` |
| Forcing (Phieq/Q/R, clamp, drag) | `forcing.py` → `Phieqfun`, `Qfun`, `Rfun` |
| Spectral transforms (FFT, Legendre) | `spectral_transform.py` → `fwd_fft_trunc`, `fwd_leg`, `invrs_leg`, `invrs_fft`, `invrsUV` |
| Basis construction (`Pmn`, `Hmn`) | `spectral_transform.py` → `PmnHmn`, `_lpmn_fallback`, `_scaling_table` |
| Diffusion filters | `filters.py` → `sigma6`, `sigma6Phi`, `diffusion` |
| Analytic initial conditions | `initial_conditions.py` → `spectral_params`, `state_var_init`, `velocity_init` |
| Nonlinear terms (A–E) | `initial_conditions.py` → `ABCDE_init` |
| Continuation save/load | `continuation.py` |
| Plotting | `plotting.py` |
| CLI entry point | `main_function.py` → `cli_main()` |
