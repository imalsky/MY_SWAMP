# SWAMPE-JAX (`my_swamp`)

A JAX rewrite of the SWAMPE spectral shallow‑water model on the sphere. The numerical core runs inside `jax.lax.scan`, so the forward simulation is end‑to‑end differentiable with respect to continuous physical parameters and explicit initial conditions.

Document version: 2026-02-13

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

- `expflag=False`: modified‑Euler scheme (`src/my_swamp/modEuler_tdiff.py`)
- `expflag=True`: explicit scheme (`src/my_swamp/explicit_tdiff.py`)

The model supports:

- Forced mode (`test=None`) with Newtonian relaxation + drag
- Two idealized test cases (`test=1` and `test=2`), matching SWAMPE-style initial conditions and numerics as closely as possible

---

## 2. Package Layout

This repository uses a `src/` layout. The import name is `my_swamp`, but the source lives under `src/my_swamp/`.

Repository (high level):

```
MY_SWAMP/
├── readme.md
├── LICENCE.txt
├── pyproject.toml
├── setup.py
├── src/
│   └── my_swamp/
│       ├── __init__.py              # Package entry; enables float64 by default (configurable)
│       ├── _version.py              # Version string
│       ├── dtypes.py                # Centralized dtype selection (float32/64)
│       ├── model.py                 # Core driver: run_model_scan (history) +
│       │                            #   run_model_scan_final (terminal-only),
│       │                            #   run_model (wrapper), run_model_gpu
│       ├── main_function.py         # CLI + legacy main() signature
│       ├── spectral_transform.py    # Gauss–Legendre quadrature, Pmn/Hmn basis construction,
│       │                            #   FFT truncation, forward/inverse Legendre transforms,
│       │                            #   wind inversion (invrsUV)
│       ├── time_stepping.py         # Scheme dispatch (explicit vs modEuler),
│       │                            #   coefficient arrays, RMS wind diagnostic
│       ├── modEuler_tdiff.py        # Modified‑Euler time differencing (parity behavior)
│       ├── explicit_tdiff.py        # Explicit time differencing (parity behavior)
│       ├── forcing.py               # Phieq, radiative forcing Q, velocity forcing R (incl. drag + clamp)
│       ├── filters.py               # Diffusion filters + diffusion operator
│       ├── initial_conditions.py    # Supported resolutions, analytic ICs, nonlinear term construction
│       ├── continuation.py          # Pickle I/O for save/load/continuation
│       ├── plotting.py              # Matplotlib plotting helpers + GIF generation
│       ├── autodiff_utils.py        # Forward-mode utilities (JVP chunking)
│       └── test_unit.py             # Unit tests for transforms and wind inversion (pytest-friendly)
├── tests/                           # Pytest suite for packaging + smoke tests
├── notebooks/                       # Example notebooks
├── paper/                           # Paper / writeup assets
└── dist/                            # Build artifacts (sdist/wheels), if present
```

Reference (NumPy/SciPy) SWAMPE code is not shipped inside this archive. When this README refers to “parity with NumPy SWAMPE”, it means parity with the upstream SWAMPE reference implementation, not a directory contained here.

---

## 3. Requirements and Installation

Requirements (as packaged by `setup.py`):

- Python 3.9+
- `numpy>=1.24`
- `scipy>=1.10`  
  Used for Gauss–Legendre nodes/weights and associated Legendre polynomials. (There is fallback code for SciPy-free environments, but the default package installation includes SciPy.)
- `jax>=0.4,<1.0`  
  `jaxlib` is intentionally not pinned here; follow JAX’s recommended install method for your platform (CPU/GPU/TPU).
- `matplotlib>=3.7` and `imageio>=2.31`  
  Used by `my_swamp.plotting` (the module is lazily imported, but these dependencies are included in the default install requirements).

Editable install from the repository root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

If you already manage your own JAX/JAXLIB installation (common on GPU/HPC), you can prevent pip from changing it by installing this package without dependencies and ensuring the dependencies above are already installed:

```bash
python -m pip install -e . --no-deps
```

Precision configuration:

- By default, this package enables JAX 64‑bit mode at import time for closer parity with NumPy SWAMPE.
- The environment variable `SWAMPE_JAX_ENABLE_X64` controls this behavior (read during import of `my_swamp`):

```bash
export SWAMPE_JAX_ENABLE_X64=1   # enable float64/complex128 (default behavior)
export SWAMPE_JAX_ENABLE_X64=0   # disable and use float32/complex64
```

---

## 4. Running the Model

### 4a. Command line

Recommended (after installing, from anywhere on your PATH):

```bash
# Forced mode (test=0 maps internally to test=None)
my-swamp --M 42 --dt 600 --tmax 200 --test 0 --no-plot

# Idealized test case 1
my-swamp --M 42 --dt 600 --tmax 200 --test 1 --no-plot

# Idealized test case 2
my-swamp --M 42 --dt 600 --tmax 200 --test 2 --no-plot
```

Alternative (module execution). This works once the package is installed, but may emit a Python `RuntimeWarning` because `src/my_swamp/__init__.py` imports `main_function` eagerly; prefer `my-swamp` for a clean CLI run:

```bash
python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 0 --no-plot
```

No-install development run from the repository root (adds `src/` to `PYTHONPATH`):

```bash
PYTHONPATH=src python -m my_swamp.main_function --M 42 --dt 600 --tmax 200 --test 0 --no-plot
```

CLI defaults (from `src/my_swamp/main_function.py`):

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

Use `run_model_scan(...)` when you need a full time history (`outs`).

For optimization/inference where you only need the terminal state (e.g. the final `Phi`), use `run_model_scan_final(...)` (or `run_model_scan(..., return_history=False)`). This avoids stacking a `(t, J, I)` history inside `jax.lax.scan`.

```python
import jax
import jax.numpy as jnp
from my_swamp.model import run_model_scan

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
    jit_scan=True,
)

outs = sim["outs"]   # dict of time histories
Phi = outs["Phi"]    # (t_len, J, I)
```

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

### 5a. Final-only loss (recommended)

For optimization/inference you usually only need the terminal state, not the full trajectory. Use `run_model_scan_final(...)` (or `run_model_scan(..., return_history=False)`) to avoid stacking a `(t, J, I)` history inside `jax.lax.scan`.

```python
import jax
import jax.numpy as jnp
from my_swamp.model import run_model_scan_final

def loss_fn(DPhieq: float) -> jnp.ndarray:
    sim = run_model_scan_final(
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
        jit_scan=True,
        diagnostics=False,
    )
    Phi_final = sim["last_state"].Phi_curr  # (J, I)
    return jnp.mean(Phi_final**2)

# Reverse-mode (good for many parameters):
g = jax.grad(loss_fn)(4.0e6)

# Forward-mode (good when differentiating wrt a small parameter vector):
g_fwd = jax.jacfwd(loss_fn)(4.0e6)
```

### 5b. Differentiating with respect to initial conditions

To differentiate with respect to explicit initial conditions, you must provide all three of:

- `eta0_init` (shape `(J, I)`)
- `delta0_init` (shape `(J, I)`)
- `Phi0_init` (shape `(J, I)`)

Optionally, you may also provide `U0_init` and `V0_init` (both shape `(J, I)`). If you provide one of `U0_init` or `V0_init`, you must provide both.

Example: reverse-mode gradient of a scalar loss with respect to the full initial geopotential field `Phi0_init` (using the same analytic IC construction as `run_model(...)` when `contflag=False`):

```python
import jax
import jax.numpy as jnp

from my_swamp.initial_conditions import (
    spectral_params,
    test1_init,
    state_var_init,
    velocity_init,
)
from my_swamp.model import run_model_scan_final

M = 42
N, I, J, dt_default, lambdas, mus, w = spectral_params(M)

a = 6.37122e6
omega = 7.292e-5
Phibar = 3.0e5
a1 = 0.05

# Build a consistent analytic IC (mirrors run_model(..., contflag=False))
SU0, sina, cosa, etaamp, Phiamp = test1_init(a, omega, a1)
eta0, _, delta0, _, Phi0, _ = state_var_init(I, J, mus, lambdas, test=None, etaamp=etaamp)
U0, V0 = velocity_init(I, J, SU0, cosa, sina, mus, lambdas, test=None)

def loss_ic(Phi0_init: jnp.ndarray) -> jnp.ndarray:
    sim = run_model_scan_final(
        M=M,
        dt=dt_default,
        tmax=50,
        Phibar=Phibar,
        omega=omega,
        a=a,
        test=None,
        forcflag=True,
        diffflag=True,
        modalflag=True,
        expflag=False,
        eta0_init=eta0,
        delta0_init=delta0,
        Phi0_init=Phi0_init,
        U0_init=U0,
        V0_init=V0,
        diagnostics=False,
        jit_scan=True,
    )
    return jnp.mean(sim["last_state"].Phi_curr)

gPhi0 = jax.grad(loss_ic)(Phi0)  # shape (J, I)
```

Practical note: differentiating with respect to a full `(J, I)` field is expensive. For inverse problems, it is usually better to parameterize the initial condition with a small number of parameters and differentiate with respect to those.

### 5c. Forward-mode gradients for a small parameter vector

When your parameter vector is small (e.g., 1–10 scalars), forward-mode can be competitive and often uses less memory than reverse-mode.

If you want a Jacobian-vector product (JVP) or want to avoid `jax.jacfwd` (which computes all tangent directions at once), compute forward-mode gradients in small chunks via JVPs.

This repo provides a helper in `my_swamp.autodiff_utils`:

```python
from my_swamp.autodiff_utils import fwd_grad

# Full jacfwd (fine for ~5 params)
g_fwd = fwd_grad(loss, theta0)

# Chunked JVPs (lower peak memory)
g_fwd_chunked = fwd_grad(loss, theta0, chunk=2)
```

### 5d. Return structure and time indexing

`run_model_scan(...)` returns a dictionary. By default (`return_history=True`) it contains:

- `static`: basis, grid, coefficients, filters (treated as constants by the scan)
- `t_seq`: time indices at which diagnostics are recorded (integers)
- `outs`: dict of time histories (each of shape `(len(t_seq), ...)`)
- `last_state`: terminal scan carry containing the final physical fields
- `starttime`: the effective start time (used for continuation)

`outs` contains:

- `eta`, `delta`, `Phi`: physical-space fields (each `(t, J, I)`)
- `U`, `V`: physical winds (each `(t, J, I)`)
- `rms`: RMS wind (shape `(t,)`)
- `spin_min`: minimum wind speed (shape `(t,)`)
- `phi_min`, `phi_max`: min/max geopotential perturbation (shape `(t,)`)

---

## 6. Plotting and Visualization

### 6a. Built-in plotting via `run_model(...)`

If you call `run_model(...)` with `plotflag=True`, it will generate:

- geopotential contour plots (optionally with wind quivers)
- spinup diagnostics plots

Plots are written under `plots/` by default. This mirrors SWAMPE behavior.

### 6b. Manual plotting from `run_model_scan(...)` output

If you prefer to generate plots manually:

```python
from my_swamp.model import run_model_scan
from my_swamp import plotting

sim = run_model_scan(
    M=42, dt=600.0, tmax=200,
    Phibar=3.0e5, omega=7.292e-5, a=6.37122e6,
    test=None, forcflag=True, diffflag=True, modalflag=True,
)

outs = sim["outs"]
static = sim["static"]

U = outs["U"]
V = outs["V"]
Phi = outs["Phi"]

lambdas = static.lambdas
mus = static.mus
Phibar = 3.0e5

step = -1
plotting.quiver_geopot_plot(
    U[step],
    V[step],
    Phi[step] + Phibar,
    lambdas,
    mus,
    timestamp="final",
    units="steps",
)
```

### 6c. GIF generation

The plotting module provides helpers for GIF generation using `imageio`. See `src/my_swamp/plotting.py`.

---

## 7. Behavior Relative to NumPy SWAMPE

This implementation aims to preserve:

- the spectral transform conventions
- the modified Euler time-differencing logic (including Robert–Asselin-like filtering)
- diffusion operators and filters
- forcing/clamping semantics and hard-stability protections

Differences can arise due to:

- JAX’s XLA compilation and algebraic reassociation
- different default dtype behavior if `SWAMPE_JAX_ENABLE_X64=0`
- small differences in Legendre basis construction depending on SciPy availability/version

---

## 8. Legacy Physics and Numerics Preserved for Parity

The following behaviors are preserved for parity with SWAMPE-style workflows:

- Triangular truncation with M = N
- Gaussian quadrature in latitude
- FFT truncation in longitude
- Spectral inversion of winds
- Newtonian relaxation forcing (`Phieq`) and drag forcing (`R`)
- Diffusion filtering (`sigma6`, `sigma6Phi`) and diffusion operator

---

## 9. Physics and Numerics Changes Not Implemented Here

This codebase is focused on parity and differentiability; it does not implement:

- adaptive time stepping or variable resolution

---

## 10. Differentiability Scope and Caveats

The simulation is differentiable with respect to:

- Continuous scalar parameters that enter the scan (e.g., `DPhieq`, `taurad`, `taudrag`, `K6`, `K6Phi`, `Phibar`, `omega`, `a`, `dt`, `alpha`).
- Explicit initial conditions (`eta0_init`, `delta0_init`, `Phi0_init`) as long as you avoid side effects and keep array shapes static.

Non-differentiable aspects include:

- File I/O (saving/loading continuation pickles)
- Plotting side effects
- Any control-flow that depends on data in a way that changes shapes or scan structure

---

## 11. GPU, Precision, and Performance Notes

- For closest parity with NumPy SWAMPE, leave `SWAMPE_JAX_ENABLE_X64` enabled (default).
- For faster runs, disable x64 (`SWAMPE_JAX_ENABLE_X64=0`), but expect larger numerical drift.
- Use `run_model_scan_final` for training/inference loops where you only need the terminal state.
- `jit_scan=True` is usually best for performance; disable only for debugging.

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
  To run tests in 32-bit mode (faster, lower precision), set:

```bash
export SWAMPE_JAX_ENABLE_X64=0
pytest -q
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
| Model driver (`run_model*`) | `model.py` |
| CLI / legacy interface | `main_function.py` |
| Spectral transforms | `spectral_transform.py` |
| Time stepping | `time_stepping.py`, `modEuler_tdiff.py`, `explicit_tdiff.py` |
| Forcing | `forcing.py` |
| Filters / diffusion | `filters.py` |
| Initial conditions | `initial_conditions.py` |
| Continuation save/load | `continuation.py` |
| Plotting | `plotting.py` |
| Forward-mode AD utils | `autodiff_utils.py` |
| Transform/unit tests | `test_unit.py`, `tests/` |
