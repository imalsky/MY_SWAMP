# SWAMPE-JAX (`my_swamp`)

A JAX rewrite of the SWAMPE spectral shallow‑water model on the sphere. The
numerical core runs inside `jax.lax.scan`, so the forward simulation is
end‑to‑end differentiable with respect to continuous physical parameters and
explicit initial conditions.

- **Numerical parity** with reference NumPy SWAMPE for default settings
  (≤ 1e-10 atol on `eta`/`delta`, ≤ 5e-8 atol on `Phi`).
- **Drop-in API**: `from my_swamp.model import run_model` works as a
  replacement for `from SWAMPE.model import run_model`.
- **Differentiable**: `jax.grad`, `jax.jvp`, `jax.jit`, `jax.vmap` work on
  the scan core out of the box.

Working on this codebase as an AI assistant or human contributor? Read
[`CLAUDE.md`](CLAUDE.md) first — it's the short, in-repo developer briefing
covering the locked parity contract, differentiability rules, validation
commands, and common pitfalls.

Document version: 2026-06-24

## Citation

If you use `my_swamp` in your research, please cite the accompanying software
paper (in preparation for the Journal of Open Source Software; the LaTeX source
lives in [`paper/paper.tex`](paper/paper.tex)) together with the original SWAMPE
model on which this port is based:

> Landgren, E., & Nadeau, A. (2022). SWAMPE: A Shallow-Water Atmospheric Model in
> Python for Exoplanets. *Journal of Open Source Software*, 7(80), 4872.
> <https://doi.org/10.21105/joss.04872>

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
12. [Reliability Helpers](#12-reliability-helpers)
13. [Testing and Parity Checks](#13-testing-and-parity-checks)
14. [Known Limitations](#14-known-limitations)
15. [Code Navigation Guide](#15-code-navigation-guide)

---

## 1. What This Code Does

SWAMPE-JAX implements a single‑layer global spectral shallow‑water model on the sphere using triangular truncation (M = N), Gaussian quadrature in latitude, and FFT in longitude. The model advances:

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
├── README.md                        # this file (user-facing)
├── CLAUDE.md                        # in-repo developer briefing (locked parity, AD rules, validation)
├── CONTRIBUTING.md                  # contribution conventions
├── LICENCE.txt                      # BSD-3-Clause
├── pyproject.toml                   # build metadata, deps, ruff/pytest config
├── setup.py                         # legacy setuptools shim
├── src/
│   └── my_swamp/
│       ├── __init__.py              # Package entry; sets jax_enable_x64 (configurable via env var)
│       ├── _version.py              # Version string
│       ├── dtypes.py                # Centralized dtype selection (float32/64)
│       ├── branching.py             # cond/select/maybe_apply with static-pred specialization
│       ├── backend_preflight.py     # JAX backend / device validation
│       ├── model.py                 # Core driver: run_model_scan (history),
│       │                            #   run_model_scan_final (terminal-only),
│       │                            #   run_model (SWAMPE-compatible wrapper),
│       │                            #   run_model_gpu, assert_finite_state
│       ├── main_function.py         # CLI + legacy main() signature
│       ├── spectral_transform.py    # Gauss–Legendre quadrature, Pmn/Hmn basis,
│       │                            #   FFT truncation, forward/inverse Legendre transforms,
│       │                            #   wind inversion (invrsUV)
│       ├── time_stepping.py         # Scheme dispatch (explicit vs modEuler),
│       │                            #   coefficient arrays, RMS wind diagnostic
│       ├── modEuler_tdiff.py        # Modified‑Euler time differencing (parity behavior)
│       ├── explicit_tdiff.py        # Explicit time differencing (parity behavior)
│       ├── forcing.py               # Phieq, radiative forcing Q, velocity forcing R
│       ├── filters.py               # Diffusion filter coefficients + apply
│       ├── initial_conditions.py    # Supported resolutions, analytic ICs, ABCDE nonlinear products
│       ├── continuation.py          # Pickle I/O for save/load/continuation
│       ├── plotting.py              # Matplotlib helpers + GIF generation (lazy import)
│       └── autodiff_utils.py        # Forward-mode utilities (JVP chunking)
├── unit_tests/                      # Pytest suite (see §13 for the test matrix)
│   └── fixtures/                    # SWAMPE-generated reference snapshots (.npz)
├── testing/                         # Benchmarks, fixture generation, parity tooling
│   └── long_run_parity_outputs/     # Generated artifacts (only summary.json is committed)
└── paper/                           # JOSS paper (LaTeX: paper.tex, paper.bib, figures, Makefile)
```

Reference (NumPy/SciPy) SWAMPE code is not shipped inside this archive. When this README refers to “parity with NumPy SWAMPE”, it means parity with the upstream SWAMPE reference implementation, not a directory contained here.

---

## 3. Requirements and Installation

Requirements (from `pyproject.toml`):

- Python 3.10+
- `numpy>=1.26,<2.0`
- `scipy>=1.10`  
  Used for Gauss–Legendre nodes/weights and associated Legendre polynomials. (There is fallback code for SciPy-free environments, but the default package installation includes SciPy.)
- `jax>=0.4.31,<0.5`  
  This repository's validated CPU test matrix is the JAX 0.4 line with NumPy 1.x. `jaxlib` is intentionally not pinned here; follow JAX’s recommended install method for your platform (CPU/GPU/TPU).
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
- `--g` defaults to `9.8` to match the function-level default in
  `my_swamp.model.run_model`.

> Note: the **function-level** defaults of
> `my_swamp.model.run_model(...)` are `plotflag=True` and `saveflag=True`
> (kept for backwards compatibility with the upstream SWAMPE
> `model.run_model` signature). The **CLI** instead defaults plotting to
> `False` because most terminal users don't want figures popping up
> mid-run. Both defaults are deliberate; pass the flag you want
> explicitly when in doubt. The example below shows the typical
> headless/HPC configuration (both off).

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
- `dead_first_idx`: scalar `int32` — the first scan-step index at which
  the blowup gate tripped, or `-1` if the run completed cleanly. Always
  present when `return_history=True`.

`outs` contains:

- `eta`, `delta`, `Phi`: physical-space fields (each `(t, J, I)`)
- `U`, `V`: physical winds (each `(t, J, I)`)
- `rms`: RMS wind (shape `(t,)`)
- `spin_min`: minimum wind speed (shape `(t,)`)
- `phi_min`, `phi_max`: min/max geopotential perturbation (shape `(t,)`)
- `dead`: per-step blowup-gate boolean (shape `(t,)`). Monotonic
  non-decreasing once tripped.

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

This implementation preserves the SWAMPE numerics for default settings.
Cross-validated against the NumPy SWAMPE reference, the JAX rewrite agrees
to within:

- ≤ 1e-10 absolute on `eta` and `delta`
- ≤ 5e-8 absolute on `Phi`
- ≤ 1e-9 absolute on `U` and `V`

Specifically preserved:

- Spectral transform conventions (triangular truncation, Gauss–Legendre
  quadrature, FFT truncation in longitude).
- Modified-Euler time-differencing logic, including the Robert–Asselin
  three-level filter.
- Diffusion filter coefficients and the diffusion operator.
- Forcing semantics: `Q < 0` clamp, `taudrag == -1` no-drag branch,
  strict-inequality dayside mask.
- Two-level initialization (`prev == curr == initial`) and the deliberate
  desync between RA-filtered physical-space carries and the unfiltered
  spectral coefficients.
- Legendre basis sign convention (factorial-based scaling with odd-`m`
  flip; matches SciPy's `lpmn` after Condon–Shortley correction).

Differences can arise due to:

- JAX/XLA compilation and algebraic reassociation (~1e-10 atol drift).
- Different default dtype behavior if `SWAMPE_JAX_ENABLE_X64=0` (much
  larger drift; not parity-grade).
- SciPy version differences in `assoc_legendre_p_all` vs `lpmn`
  (negligible at our supported `M` range).

The full enumerated parity contract — including the historical SWAMPE
quirks deliberately reproduced — lives in [`CLAUDE.md`](CLAUDE.md) §3.

---

## 8. Legacy Physics and Numerics Preserved for Parity

The high-level numerical method preserved:

- Triangular truncation with M = N.
- Gaussian quadrature in latitude, FFT truncation in longitude.
- Spectral inversion of winds from absolute vorticity and divergence.
- Newtonian relaxation forcing (`Phieq`) and drag forcing (`R`).
- Sixth-order hyperdiffusion filtering (`sigma6`, `sigma6Phi`).
- The tidally-locked dayside `Phieq` model with substellar point at
  (λ=0, μ=0).

For the full enumerated list of historical quirks (e.g., the modified-Euler
delta tendency that uses `Bm+Fm` even in its unforced branch), see
[`CLAUDE.md`](CLAUDE.md) §3.

---

## 9. Physics and Numerics Changes Not Implemented Here

This codebase is focused on parity and differentiability; it does not
implement:

- adaptive time stepping or variable resolution
- multi-layer extensions
- substellar-point relocation (substellar is hard-coded at λ=0, μ=0)

---

## 10. Differentiability Scope and Caveats

The simulation is differentiable with respect to:

- Continuous scalar parameters that enter the scan (e.g., `DPhieq`,
  `taurad`, `taudrag`, `K6`, `K6Phi`, `Phibar`, `omega`, `a`, `dt`,
  `alpha`). Verified by `unit_tests/test_autodiff.py`.
- Explicit initial conditions (`eta0_init`, `delta0_init`, `Phi0_init`,
  optional `U0_init`/`V0_init`) as long as you avoid side effects and
  keep array shapes static.

`K6Phi=None` is a deliberate API default meaning "inherit `K6`". This
preserves SWAMPE's legacy behavior where geopotential diffusion uses the
same coefficient as vorticity/divergence unless you explicitly override
it.

Non-differentiable aspects include:

- File I/O (saving/loading continuation pickles).
- Plotting side effects.
- Boolean run-mode flags (`forcflag`, `diffflag`, `expflag`, `modalflag`,
  `diagnostics`) — these are static configuration, not parameters.
- Any control-flow that depends on data in a way that changes shapes or
  scan structure.

Rules and pitfalls (`float(tracer)` coercions, JIT cache keys, etc.) are
listed in [`CLAUDE.md`](CLAUDE.md) §5.

---

## 11. GPU, Precision, and Performance Notes

- For closest parity with NumPy SWAMPE, leave `SWAMPE_JAX_ENABLE_X64` enabled (default).
- For faster runs, disable x64 (`SWAMPE_JAX_ENABLE_X64=0`), but expect larger numerical drift.
- Use `run_model_scan_final` for training/inference loops where you only need the terminal state.
- `jit_scan=True` is usually best for performance; disable only for debugging.

### Memory cost of `return_history=True`

`run_model_scan(..., return_history=True)` (the default) materializes a
`(len(t_seq), J, I)` array for each of the five physical fields plus four
`(len(t_seq),)` scalar diagnostics inside `jax.lax.scan`. The dominant
memory footprint is roughly:

```
bytes ≈ 5 * len(t_seq) * J * I * itemsize
```

with `itemsize = 8` for float64 and `itemsize = 4` for float32. At the
default M=42 grid (J=64, I=128) this is roughly:

| `len(t_seq)` | float64 | float32 |
|--------------|---------|---------|
| 1,000        | 328 MB  | 164 MB  |
| 10,000       | 3.3 GB  | 1.6 GB  |
| 100,000      | 33 GB   | 16 GB   |

For long integrations, optimization, or inference, use
`run_model_scan_final(...)` (or pass `return_history=False`) — it discards
the per-step trajectory and returns only the terminal state, which scales
with `J * I` rather than `len(t_seq) * J * I`.

---

## 12. Reliability Helpers

### 12a. Blowup gating during the scan

When `diagnostics=True` (the default for `run_model_scan`), the scan body
checks per-step RMS wind speed against `RunFlags.blowup_rms` (default
`8000.0` m/s) and switches to a "frozen" branch on the first step that
exceeds it. The scan still runs to completion (you cannot change shape
mid-scan), but no further physics is computed from that step onward.

The first scan-step index at which the gate tripped is returned as
`dead_first_idx` in the result dict (or `-1` if the run completed cleanly):

```python
sim = run_model_scan(M=42, dt=600.0, tmax=200, Phibar=3.0e5,
                     omega=7.292e-5, a=6.37122e6, test=None,
                     diagnostics=True)

if int(sim["dead_first_idx"]) >= 0:
    print(f"blowup at scan step {int(sim['dead_first_idx'])}")
```

### 12b. Post-run NaN check (recommended for `diagnostics=False`)

`run_model_scan_final(...)` (and `run_model_scan(..., diagnostics=False)`)
skip the in-scan blowup gate for performance. Use `assert_finite_state`
on the host side to catch silent NaN/Inf propagation:

```python
from my_swamp.model import run_model_scan_final, assert_finite_state

sim = run_model_scan_final(M=42, dt=600.0, tmax=10_000, Phibar=3.0e5,
                           omega=7.292e-5, a=6.37122e6, test=None)
assert_finite_state(sim["last_state"])  # raises if any field has NaN/Inf
```

By default `assert_finite_state` raises `RuntimeError` on detection. Pass
`raise_on_nan=False` to get a `bool` return instead.

---

## 13. Testing and Parity Checks

There are three levels of testing: the fast pytest suite for everyday
development, a long-run parity script for validating numerical agreement
against the NumPy SWAMPE reference, and a benchmark harness for measuring
performance. All three are described below.

Current status: **36 tests, all passing on CPU x64 in ~30s.**

---

### 13a. Unit Tests (pytest)

Install the dev dependencies and run the full suite on CPU:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 pytest -q
```

You can also run specific subsets using pytest markers:

```bash
# Just the smoke tests (fast)
JAX_PLATFORMS=cpu pytest -q -m smoke

# Just the parity regression tests
JAX_PLATFORMS=cpu pytest -q -m parity

# List all collected tests without running them
JAX_PLATFORMS=cpu pytest --collect-only -q
```

The package defaults to JAX 64-bit mode for closer numerical parity with
the NumPy/SciPy SWAMPE reference. To run tests in 32-bit mode (faster,
less precise):

```bash
export SWAMPE_JAX_ENABLE_X64=0
JAX_PLATFORMS=cpu pytest -q
```

To verify that the parity tests correctly gate on x64 (they should fail
without it):

```bash
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=0 JAX_ENABLE_X64=0 pytest -q -m parity
```

Parity failures here are expected — this just confirms the x64 guard is
working.

To lint the source and test directories:

```bash
ruff check src unit_tests testing
```

The test suite lives under `unit_tests/` and covers:

| File | What it tests |
|---|---|
| `test_import_and_version.py` | Package imports and `_version.py`. |
| `test_backend_preflight.py` | JAX backend detection and visibility. |
| `test_static_spectral_params.py` | Grid sizes and Gauss–Legendre nodes for M=42. |
| `test_transform_stack.py` | Forward/inverse Legendre and FFT round-trips, wind ↔ vorticity/divergence diagnostic. |
| `test_model_scan_smoke.py` | One end-to-end `run_model_scan` step, finite outputs. |
| `test_parity_quirks.py` | One test per locked-parity item (see [`CLAUDE.md`](CLAUDE.md) §3). |
| `test_parity_reference_regression.py` | Regression against stored reference fixtures (NumPy SWAMPE snapshots). |
| `test_autodiff.py` | `jax.grad`/`jax.jvp` smoke across 9 scalar parameters; finite-difference cross-check on `DPhieq`; `jax.grad` over a `(J, I)` initial-Phi field. |
| `test_continuation_roundtrip.py` | `contflag` resume reproduces an explicit-IC restart from the same single-level state. |
| `test_invalid_input.py` | `pytest.raises` for bad `tmax`/`dt`/`M`/`test`, partial/wrong-shape ICs, contflag without contTime, non-numeric contTime. |
| `test_vmap_smoke.py` | `jax.vmap` over a stack of `DPhieq` values; per-member agreement with direct calls. |

---

### 13b. SWAMPE vs. MY_SWAMP Long-Run Parity Check (`compare_long_run_parity.py`)

This is the main tool for checking that `my_swamp` stays numerically close to the original NumPy SWAMPE reference over long integrations. It is not part of the pytest suite because a useful horizon (100 days) can take several minutes.

Run it from the repository root:

```bash
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 python testing/compare_long_run_parity.py --days 100
```

What it does:
- Runs both `SWAMPE` (NumPy) and `my_swamp` (JAX) with the same forced-mode parameter set.
- Prints per-field error statistics (relative L2, max fractional, RMS fractional, max absolute) to the console.
- Writes `summary.json` with the full error breakdown and run parameters.
- Saves `comparison_fields.npz` with the SWAMPE fields, MY_SWAMP fields, and absolute error arrays for `eta`, `delta`, `Phi`, `U`, and `V`.
- Generates `field_comparison.png` — a grid of side-by-side maps showing the SWAMPE fields, MY_SWAMP fields, and signed fractional differences for each field.

All output lands in `testing/long_run_parity_outputs/forced_default_100d/` by default.

Key options:

```bash
# Change integration horizon or timestep
python testing/compare_long_run_parity.py --days 200 --dt 600

# Run an idealized test case instead of forced mode (1 or 2)
python testing/compare_long_run_parity.py --days 50 --test 1

# Write outputs to a custom directory
python testing/compare_long_run_parity.py --days 100 --out-dir /tmp/parity_check
```

The script requires that the SWAMPE reference package is importable. It looks for it at `../SWAMPE` relative to the `MY_SWAMP` root.

---

### 13c. Regenerating Reference Fixtures (`generate_reference_parity_fixtures.py`)

The regression tests in `test_parity_reference_regression.py` compare against stored `.npz` fixtures generated from the NumPy SWAMPE reference. If you change the numerics intentionally, regenerate them:

```bash
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 python testing/generate_reference_parity_fixtures.py
```

What it does:
- Runs the NumPy SWAMPE reference model for two cases: an unforced test case 1 run and a forced default run.
- Saves field snapshots at multiple intermediate steps plus the final state for each case.
- Also computes and saves a phase curve derived from the final `Phi` field.
- Writes two compressed `.npz` fixture files to `unit_tests/fixtures/`.

This script requires the SWAMPE reference package at `../SWAMPE`. Commit the updated fixtures alongside your code change so the regression baseline stays current.

---

### 13d. Performance Benchmarking (`benchmark_scan.py`)

The benchmark harness in `testing/benchmark_scan.py` measures wall-clock time for `run_model_scan_final` across multiple timed runs after a JIT warmup. It prints backend info (device, x64 status), compile time, and per-run statistics including mean, median, min, max, and per-step median time in milliseconds.

Basic usage:

```bash
python testing/benchmark_scan.py --M 42 --tmax 300 --timed-runs 3
```

Key options:

```bash
# Run on GPU (if available)
python testing/benchmark_scan.py --backend gpu --require-gpu

# Higher resolution
python testing/benchmark_scan.py --M 63 --tmax 500

# Forced mode with diffusion
python testing/benchmark_scan.py --M 42 --tmax 300 --forcflag true --diffflag true

# Adjust warmup and timed run counts
python testing/benchmark_scan.py --warmup-runs 2 --timed-runs 5

# Fail fast if x64 is not enabled
python testing/benchmark_scan.py --require-x64
```

---

## 14. Known Limitations

- Supported resolutions are limited to `M in {42, 63, 106}` as defined in
  `initial_conditions.spectral_params`.
- Supported test modes are `test=None` (forced), `test=1`, and `test=2`.
  Legacy SWAMPE selectors `test=9, 10, 11` are not implemented; passing
  them via the legacy `main_function.main(...)` raises
  `NotImplementedError`.
- Continuation saving defaults to `data/` (relative to the working
  directory) and plotting defaults to `plots/`. Both directories are in
  `.gitignore`.
- Continuation saves a single time level of physical state and re-derives
  winds + spectral coefficients on resume. The leapfrog two-level memory
  is therefore not preserved across a save/load boundary (matches
  reference SWAMPE; verified by `test_continuation_roundtrip.py`).
- Single-device only — no `pmap`/`pjit`/`shard_map`/`Mesh` use. `jax.vmap`
  works for ensemble forward simulations (verified by `test_vmap_smoke.py`).

---

## 15. Code Navigation Guide

| Topic | Primary locations |
|---|---|
| Developer briefing (parity contract, AD rules, validation) | [`CLAUDE.md`](CLAUDE.md) |
| Model driver (`run_model*`, `assert_finite_state`, `Static`/`RunFlags`/`State`) | `src/my_swamp/model.py` |
| CLI / legacy interface | `src/my_swamp/main_function.py` |
| Spectral transforms | `src/my_swamp/spectral_transform.py` |
| Time stepping | `src/my_swamp/time_stepping.py`, `modEuler_tdiff.py`, `explicit_tdiff.py` |
| Forcing | `src/my_swamp/forcing.py` |
| Filters / diffusion | `src/my_swamp/filters.py` |
| Initial conditions | `src/my_swamp/initial_conditions.py` |
| Continuation save/load | `src/my_swamp/continuation.py` |
| Plotting | `src/my_swamp/plotting.py` |
| Forward-mode AD utils | `src/my_swamp/autodiff_utils.py` |
| Static/dynamic branching helpers | `src/my_swamp/branching.py` |
| Backend detection / preflight | `src/my_swamp/backend_preflight.py` |
| Dtype switch (float32/64) | `src/my_swamp/dtypes.py` |
| Transform/unit tests | `unit_tests/test_transform_stack.py` |
| Autodiff tests | `unit_tests/test_autodiff.py` |
| Continuation round-trip test | `unit_tests/test_continuation_roundtrip.py` |
| Validation tests for invalid input | `unit_tests/test_invalid_input.py` |
| `vmap` ensemble smoke test | `unit_tests/test_vmap_smoke.py` |
| Long-run parity vs NumPy SWAMPE | `testing/compare_long_run_parity.py` |
| Reference fixture generation | `testing/generate_reference_parity_fixtures.py` |
| Performance benchmark | `testing/benchmark_scan.py` |
