# SWAMPE ⇄ my_swamp (JAX) parity audit

This document is a compiled, single-place checklist of:
1) Differentiability status (with emphasis on the `lax.scan` integration core).
2) Legacy quirks that must be preserved to match the reference SWAMPE behavior.
3) Candidate “corrected physics” changes (NOT applied when strict SWAMPE parity is the goal).
4) Dependency / portability issues observed in the code and recommended fixes.
5) Code bloat / unused-argument inventory.
6) GPU/vectorization readiness notes.

The intent is *legacy-faithfulness*: “match SWAMPE identically” means matching the reference implementation’s behavior, including known quirks, unless explicitly running a separate corrected-physics mode.

---

## 1) Differentiability audit (core scan mode)

### What is differentiable
The core time integration in `my_swamp.model.simulate_scan()` is built around `jax.lax.scan` and uses JAX primitives (`jax.numpy`, `jax.lax.cond`, FFTs, `einsum`, etc.) inside the scan body.

In a pure “scan path” configuration (no plotting, no saving, no NumPy materialization), the integration is differentiable with respect to:
- Continuous parameters that enter the scan body (e.g., `dt`, `Phibar`, `omega`, `a`, `taurad`, `taudrag`, `K6`, `DPhieq`, `alpha`, and initial conditions).
- Any objective that is a smooth function of the scan outputs (e.g., sums/means/norms of fields).

### What is NOT differentiable / where gradients become piecewise
These are not “JAX failures,” but they are important if you’re doing AD end-to-end:

1) Blow-up gating introduces a discrete branch  
   In `simulate_scan.step()` the predicate `rms > blowup_rms` drives a `lax.cond` that either:
   - updates normally, or
   - “freezes” the state (skip update).

   This is piecewise differentiable and discontinuous at the threshold. If you want gradients for long integrations, use a very high `blowup_rms` or disable the blow-up logic for optimization runs.

2) Diagnostics that use `min/max` are non-smooth  
   `spin_min = min(sqrt(U^2+V^2))`, `phi_min/max`, etc. are not smooth at points where argmin/argmax changes. If your loss uses these values, expect noisy/undefined gradients.

3) Any host materialization breaks JAX tracing  
   In `run_model()`:
   - `as_numpy=True` converts to NumPy.
   - saving/plotting paths force host transfers and Python loops.

   For differentiable execution: use `run_model_scan()` / `run_model_gpu()` with `as_numpy=False`, `plotflag=False`, `saveflag=False`.

4) Static basis precomputation is not a JAX graph  
   `Pmn/Hmn` and Gauss–Legendre nodes/weights are computed in Python (SciPy/NumPy) and injected as constants. That’s fine for differentiating through the time integration, but you cannot differentiate w.r.t. the basis-generation inputs (e.g., grid definition) without rewriting that part in JAX.

### Bottom line for differentiability
- **Core scan mode is differentiable** in the usual sense (JAX AD through `lax.scan`) provided you stay on the pure-JAX execution path and avoid objectives dominated by `min/max` or the blow-up threshold branch.
- The scan does not contain any obvious “hard” AD breakers (no host callbacks, no side-effecting prints, no NumPy inside the scan body).

---

## 2) SWAMPE legacy quirks that must be preserved for identical matching

These behaviors exist in the reference SWAMPE and are reproduced in the JAX port. They are “quirks” from a physics/numerics standpoint, but they are part of strict parity.

### 2.1 Explicit scheme: divergence tendency terms are dropped
File: `explicit_tdiff.py` (`delta_timestep`)

The explicit divergence update computes `deltacomp2/3/4` but then sets:
- `deltamntstep = deltacomp1`

This drops key divergence tendencies. If parity is the goal, keep it as-is.

### 2.2 Explicit scheme: drag is double-counted
Files:
- `forcing.py` (`Rfun`) includes Rayleigh drag: `F = Ru - U/taudrag`, `G = Rv - V/taudrag` (unless `taudrag == -1`)
- `explicit_tdiff.py` adds additional `U/taudrag` and `V/taudrag` terms inside divergence/vorticity forcing in the explicit branch.

This over-damps winds when drag is enabled, but it matches SWAMPE.

### 2.3 Modified Euler scheme: historical coefficient scaling quirks
File: `modEuler_tdiff.py`

Preserved quirks include (see in-file comments):
- “/4 rather than /2” effective conversion scalings in multiple places.
- forced/unforced branches using inconsistent coefficient scalings.
- divergence update using forced A/B terms even when `forcflag=False`.

### 2.4 Modal splitting (Robert–Asselin-style) filter does not feed back into spectral state
File: `model.py`

In both SWAMPE and this JAX port:
- modal splitting is applied to *physical* time level 1 (`eta_curr`, `delta_curr`, `Phi_curr`) for diagnostics,
- but the spectral coefficients used for time stepping are *not* recomputed from the filtered fields.

This makes the filter largely diagnostic rather than dynamically damping the leapfrog computational mode.

### 2.5 Forcing clamp for negative Q (“mass loss prevention”)
File: `forcing.py` (`Rfun`)

- Negative `Q` values are clamped (`Qclone = 0` where `Q<0`).
- `Ru/Rv` are explicitly set to 0 where `Q<0`.

This is non-smooth and non-physical, but part of the legacy behavior.

### 2.6 Spectral transform conventions
Files: `spectral_transform.py`, `initial_conditions.py`

- `Pmn/Hmn` normalization uses factorial-based scaling and an additional sign flip for odd `m` (cancels SciPy’s Condon–Shortley phase).
- `invrsUV` zeros out `n=0` modes before diagnostics (`deltamn[:,0]=0`, `etamn[:,0]=0`), matching the reference note “do not sum over n=0”.

### 2.7 Blow-up stopping behavior differs because `lax.scan` cannot “break”
Reference SWAMPE breaks out of the Python time loop when RMS winds exceed a threshold.

The JAX scan uses a “dead/frozen” state:
- after threshold exceedance, it stops updating and returns constant fields for subsequent steps.

For parity in *field evolution up to blow-up*, this is fine. For parity in *output length semantics*, you may want a post-processing truncation on the host.

---

## 3) Candidate “corrected physics” changes (do NOT apply for strict parity)

If you ever want a corrected-physics mode, these are the most obvious candidates:

1) Explicit divergence update: include all computed components  
   Replace `deltamntstep = deltacomp1` with the intended combination of comps 1–4.

2) Remove drag double-counting  
   Decide whether Rayleigh drag lives in `forcing.Rfun` or in the explicit branch, but not both.

3) Make modified Euler scalings internally consistent  
   Replace legacy /4 and forced/unforced mismatches with a single consistent discretization.

4) Make modal splitting actually affect dynamics  
   Apply the filter consistently to the prognostic state used by the next step (including spectral coefficients), not only to stored physical diagnostics.

5) Replace the `Q<0` clamp with a physically consistent mass correction  
   The current clamp is a hard nonlinearity and can distort momentum coupling.

6) Clarify geopotential vs height conventions  
   `forcing.Qfun` docstrings mention a factor of `g`, but the code does not apply it. If you intend a height-based formulation, you may need to revisit this. (Do not change if strict SWAMPE parity is required.)

7) Optionally reintroduce ramp-up forcing  
   SWAMPE contains a `Qfun_with_rampup` helper (unused in the main loop). Ramp-up can improve stability for short radiative timescales.

---

## 4) Dependency / portability issues

### 4.1 SciPy `special.lpmn` availability across SciPy versions
The reference SWAMPE (and the original JAX port snapshot) relied on `scipy.special.lpmn`.

If you are in an environment where `scipy.special.lpmn` is missing (as reported in your audit notes), model construction fails before time stepping.

Mitigation implemented in this snapshot:
- `spectral_transform.PmnHmn` now:
  - uses SciPy `lpmn` when present,
  - otherwise falls back to a stable recurrence-based implementation `_lpmn_fallback` that matches SciPy’s ordering and Condon–Shortley convention up to floating round-off.
- `spectral_transform.gauss_legendre` now:
  - uses SciPy `roots_legendre` when present,
  - otherwise falls back to `numpy.polynomial.legendre.leggauss`.

This removes the *hard* SciPy requirement for core execution and avoids breakage when SciPy changes its API.

### 4.2 JAX runtime dependency (`jaxlib`)
The JAX port requires `jaxlib` at runtime. If only `jax` is installed (without `jaxlib`), imports fail early.

For GPU execution you must install the GPU-enabled `jaxlib` matching your CUDA stack.

---

## 5) Code bloat / unused argument inventory

A quick static scan of the JAX port found the following *function parameters that are unused in the function body* (often due to legacy-signature parity). Examples:

- `forcing.Phieqfun(..., g)` : `g` unused (matches SWAMPE legacy signature).
- `spectral_transform.fwd_leg(data, J, M, N, Pmn, w)` : `J, M, N` unused.
- `spectral_transform.invrs_leg(legcoeff, I, J, M, N, Pmn)` : `N` unused.
- `time_stepping.tstepcoeff(J, M, dt, mus, a)` : `J` unused.
- Many scheme functions accept full “scheme API” signatures but only use a subset of the parameters (e.g., `explicit_tdiff.phi_timestep` does not use `etam0`, etc.).

Recommendation for de-bloating without breaking external APIs:
- Keep public/legacy signatures at the module boundary.
- Introduce internal “_core” functions with minimal argument lists and have the legacy functions call them.

This reduces cognitive load and can reduce JIT compile time by not threading unused arrays through JAX traces.

---

## 6) GPU and vectorization readiness

### 6.1 GPU execution
The scan body uses JAX ops throughout; it should place the heavy compute (FFTs, einsums) on GPU when:
- you have GPU-enabled `jaxlib`, and
- you avoid host transfers (`as_numpy=False`, `saveflag=False`, `plotflag=False`).

Use:
- `run_model_gpu(...)` or
- `run_model_scan(..., jit_scan=True, as_numpy=False, ...)`

### 6.2 Batch/ensemble vectorization
For ensemble runs:
- Precompute `static = build_static(...)` once.
- Build a batch of initial `State` objects (stack arrays along a leading batch axis).
- `vmap` a wrapper around `simulate_scan(state0, t_seq, static, flags, ...)`.

You may need to adjust `in_axes` and ensure FFT/Legendre transforms operate on the correct axes (typically keeping the last two dims as `(J, I)` / `(M+1, N+1)`).

### 6.3 Precision
The package defaults to enabling float64 for parity with NumPy SWAMPE. On many GPUs float64 is slower. If performance matters more than exact parity, set:
- `SWAMPE_JAX_ENABLE_X64=0` before import.

---

## 7) Review of the provided audit comments

Your audit notes map cleanly onto the reference behavior:

- **[P0] SciPy `lpmn` missing**: this is a real portability failure if your SciPy build lacks `lpmn`. The fallback added in `spectral_transform.py` addresses this.
- **[P1] Explicit divergence tendency dropped**: confirmed; it’s a SWAMPE quirk preserved for parity.
- **[P1] Explicit forcing double-counts drag**: confirmed; preserved for parity.
- **[P1] Modified Euler uses legacy coefficient quirks**: confirmed; preserved for parity.
- **[P2] README/setup mismatches**: not verifiable from the provided zip snapshots (no packaging/README files were included). If those exist in your full repo, the fix is documentation alignment: clearly separate “legacy-faithful” vs “corrected physics” modes and ensure toggles are implemented or removed.

