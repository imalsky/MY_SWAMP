# SWAMPE (numpy/scipy) vs my_swamp (JAX): exact behavior differences + parity toggle

This repository contains:
- `SWAMPE copy/` (baseline, numpy/scipy implementation)
- `my_swamp/` (JAX implementation aimed at being differentiable + GPU-friendly)

This README is a *behavioral* diff: it calls out every known change that can affect the physics/trajectory, and where it lives in code. A full line-by-line diff is also provided (see the last section).

---

## 1) The main trajectory difference you observed: modified-Euler effective timestep scaling (expflag=False)

Baseline SWAMPE’s modified-Euler implementation (`SWAMPE copy/modEuler_tdiff.py`) contains a control-flow overwrite that changes the *effective* timestep coefficients.

### 1.1 What baseline SWAMPE actually does

`time_stepping.tstepcoeff(...)` returns:
- `tstepcoeff1 = 2*dt / (a*(1-mu^2))`
- `tstepcoeff2 = 2*dt / a^2`

In baseline SWAMPE:

A) `phi_timestep(...)` and `delta_timestep(...)`:
- At the start of each function: `tstepcoeff1 = tstepcoeff1/2; tstepcoeff2 = tstepcoeff2/2`
- Then (both in the `forcflag==True` *and* the `else` branch): **they divide by 2 again** before recomputing the returned tendency.
- The earlier computation is overwritten, so the returned update always uses:

  - `tstepcoeff1_eff = tstepcoeff1 / 4`
  - `tstepcoeff2_eff = tstepcoeff2 / 4`

I.e. relative to the “2*dt” convention, baseline SWAMPE ends up using:
- `tstepcoeff1_eff = dt / (2*a*(1-mu^2))`
- `tstepcoeff2_eff = dt / (2*a^2)`

B) `eta_timestep(...)`:
- If `forcflag==True`: **no halving is applied at all** (uses the full `2*dt` coefficient).
- If `forcflag==False`: they do `tstepcoeff1 = tstepcoeff1/2`.

So baseline SWAMPE uses different effective time coefficients depending on `forcflag` for `eta`, and a different (double-halved) coefficient for `phi/delta`.

### 1.2 What the earlier JAX port did (before this patch)

The prior JAX port standardized the conversion and used:
- `tstepcoeff1_eff = tstepcoeff1 / 2`
- `tstepcoeff2_eff = tstepcoeff2 / 2`
for *all* modified-Euler updates.

That is a major trajectory change vs baseline SWAMPE when `expflag=False`.

### 1.3 Parity toggle added in this patch: `legacy_modeuler_scaling`

In `my_swamp/modEuler_tdiff.py`, all three modified-Euler stepping functions now accept:

    legacy_modeuler_scaling: bool = False

Behavior:
- `legacy_modeuler_scaling=False` (default): “corrected / consistent” scaling
  - phi/delta: use `/2`
  - eta: use `/2` (independent of forcflag)
- `legacy_modeuler_scaling=True`: emulate baseline SWAMPE’s historical quirks
  - phi/delta: use `/4`
  - eta: use `1.0` if `forcflag=True`, else `/2`

This is the “bool switch” you asked for, and it directly targets the major difference (A).

How to enable from a notebook:

```python
from my_swamp.model import run_model_scan

result = run_model_scan(
    M=42, dt=30, tmax=100,
    Phibar=4e6, omega=3.2e-5, a=8.2e7,
    taurad=3600*24, taudrag=10*3600*24,
    DPhieq=4e6,
    expflag=False,                   # modified-Euler
    legacy_modeuler_scaling=True,    # <- emulate baseline SWAMPE dt scaling
)
```

---

## 2) Other *physics/trajectory* differences (independent of dt-scaling)

### 2.1 Modified-Euler: delta unforced branch used forced expressions in baseline SWAMPE (forcflag=False case)

Baseline SWAMPE (`SWAMPE copy/modEuler_tdiff.py`), `delta_timestep(...)`:
- In the `else:` (forcflag False) branch, it still uses `(Bm+Fm)` and `(Am-Gm)` (copy/paste bug).
- Since forcing terms are computed upstream even if `forcflag=False`, this changes the “unforced” dynamics.

JAX version uses:
- `Bm` and `Am` when `forcflag=False`
- `Bm+Fm` and `Am-Gm` only when `forcflag=True`

This only affects runs with `forcflag=False`.

### 2.2 Explicit scheme (expflag=True): divergence update completeness (bug fix)

Baseline SWAMPE’s explicit scheme computes several divergence-related spectral terms but then returns only the carry-over term:

    deltamntstep = fwd_leg(deltam0)

The JAX version restores the full update:

    deltamntstep = deltacomp1 + deltacomp2 + deltacomp3 + deltacomp4

This only affects runs with `expflag=True`.

### 2.3 Explicit scheme (expflag=True): Rayleigh drag was double-counted when forcing enabled (bug fix)

In this codebase, `forcing.Rfun(...)` already includes Rayleigh drag in the returned (F,G) forcing tendencies. The explicit scheme was adding extra `U/taudrag` and `V/taudrag` terms, effectively applying drag twice.

The JAX version removes the extra drag terms and uses only `Fm/Gm` (which already encode drag). This only affects runs with `expflag=True` and `forcflag=True`.

---

## 3) Numerical differences that can still change trajectories (even with legacy dt scaling enabled)

Even if you enable `legacy_modeuler_scaling=True`, two differences can still produce visible (but usually smaller) divergence over time:

### 3.1 Default dtype: float32 vs float64

- Baseline SWAMPE uses numpy/scipy defaults (typically float64).
- The JAX port defaults to float32 for performance unless x64 is enabled.

For closest parity, enable x64 before importing/using JAX code:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

(or set environment variable `JAX_ENABLE_X64=1`).

### 3.2 Associated Legendre / basis generation implementation

Baseline SWAMPE uses scipy’s special functions (and scipy Gauss–Legendre nodes).
The JAX port uses a recurrence/vectorized implementation so it can be JIT compiled.

This should be mathematically equivalent, but:
- rounding differs (especially in float32),
- evaluation order differs,
- small differences in Pmn/Hmn propagate nonlinearly over time.

---

## 4) Where the new toggle is wired in the JAX code

Changes are intentionally minimal and localized:

- `my_swamp/modEuler_tdiff.py`
  - adds `legacy_modeuler_scaling` arg to:
    - `phi_timestep`
    - `delta_timestep`
    - `eta_timestep`
  - implements the two scaling modes described above

- `my_swamp/time_stepping.py`
  - `tstepping(...)` now accepts `legacy_modeuler_scaling` and forwards it only to the modified-Euler module

- `my_swamp/model.py`
  - adds `legacy_modeuler_scaling` to `RunFlags`
  - exposes it in `run_model_scan(..., legacy_modeuler_scaling=...)`
  - passes it into `time_stepping.tstepping(...)`

---

## 5) Full exact diff (line-by-line)

A full recursive diff is provided alongside this README:

- `swampe_numpy_vs_jax_patched.diff`

This is the authoritative “exactly what changed” source (including all refactors done for JAX/JIT/differentiability).
