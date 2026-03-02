# MY_SWAMP Specification: SWAMPE Parity & Optimization Guide

A comprehensive specification for the JAX port of SWAMPE (`MY_SWAMP`), clearly
distinguishing **intentional parity choices** from **safe optimizations**.

Every item is tagged:

- **PARITY-CRITICAL**: Intentional choice to match reference SWAMPE output.
  Changing this will cause numerical divergence from the original. Do NOT modify
  unless you explicitly intend to break parity.
- **PARITY-SAFE**: Can be changed without affecting physical outputs.
- **PARITY-DIVERGENCE**: Already differs from reference SWAMPE (intentional bug fix
  or improvement).

---

## Section 0: Intentional SWAMPE Parity Choices (DO NOT CHANGE)

These are behaviors in MY_SWAMP that **deliberately** reproduce quirks,
conventions, or implementation details of the original NumPy SWAMPE. They exist
to guarantee bitwise-identical (up to float64 round-off) physical outputs.

### 0.1 modEuler coefficient /4 scaling [PARITY-CRITICAL]

**Files:** `modEuler_tdiff.py:79-80,164-165`

In the original SWAMPE `modEuler_tdiff.py`, the leapfrog coefficient
`tstepcoeff` (which encodes `2*dt/(a*(1-mu^2))`) is divided by 2 at the top of
`phi_timestep`. The first use of this halved coefficient computes a result that
is immediately overwritten. The second use effectively divides by 2 again,
yielding an overall factor of `/4`. MY_SWAMP reproduces this directly:

```python
tstep1 = tstepcoeff1 / 4.0   # matches the double-halving in reference
tstep2 = tstepcoeff2 / 4.0
```

The same `/4` applies to `delta_timestep`.

### 0.2 modEuler delta always uses Bm+Fm, Am-Gm [PARITY-CRITICAL]

**File:** `modEuler_tdiff.py:168-169`

In the reference SWAMPE, the modified-Euler `delta_timestep` uses `Bm+Fm` and
`Am-Gm` (the forced nonlinear terms) **regardless** of whether `forcflag` is
`True` or `False`. This is a historical quirk -- the forced terms leak into the
unforced branch. MY_SWAMP preserves this:

```python
B_force = Bm + Fm   # always includes Fm
A_force = Am - Gm   # always includes Gm
```

### 0.3 modEuler eta asymmetric coefficient scaling [PARITY-CRITICAL]

**File:** `modEuler_tdiff.py:251-253`

In the reference SWAMPE, the forced `eta_timestep` uses the **unscaled**
`tstepcoeff1` (the full `2*dt/(a*(1-mu^2))`), while the unforced branch uses
`tstepcoeff1 / 2`. This differs from the phi/delta timesteppers which use `/4`.
MY_SWAMP preserves this asymmetry:

```python
tstep1 = jax.lax.select(forc_pred, tstepcoeff1, tstepcoeff1 / 2.0)
```

### 0.4 modEuler phi forced/unforced A,B coupling [PARITY-CRITICAL]

**File:** `modEuler_tdiff.py:83-84`

The reference SWAMPE phi_timestep uses `Am-Gm` and `Bm+Fm` when forced, but
plain `Am` and `Bm` when unforced, for the delta-contribution terms inside the
Phi update. MY_SWAMP matches:

```python
B_eff = jax.lax.select(forcflag, Bm + Fm, Bm)
A_eff = jax.lax.select(forcflag, Am - Gm, Am)
```

### 0.5 Explicit delta uses ONLY deltacomp1 (carry-over) [PARITY-CRITICAL]

**File:** `explicit_tdiff.py:146-160`

The reference SWAMPE explicit branch computes `deltacomp2`, `deltacomp3`,
`deltacomp4` (3 additional `fwd_leg` calls) and then sets
`deltamntstep = deltacomp1`, discarding all three. The additional terms are
commented out in the original. MY_SWAMP preserves this carry-over-only
behavior. The wasted transforms are kept for parity of the traced computation
graph (see Section 1.1 for the optimization note).

### 0.6 Explicit eta/delta forcing includes U/taudrag terms [PARITY-CRITICAL]

**Files:** `explicit_tdiff.py:162-178,241-255`

When `forcflag=True`, the explicit scheme adds forcing terms proportional to
`U/taudrag` and `V/taudrag` **in addition to** `Fm` and `Gm` (which already
include Rayleigh drag via `forcing.Rfun`). This is a double-counting that exists
in the reference SWAMPE and is preserved in MY_SWAMP for parity.

### 0.7 Condon-Shortley phase handling [PARITY-CRITICAL]

**File:** `spectral_transform.py:228-231`

The reference SWAMPE uses SciPy's `lpmn` (which includes the Condon-Shortley
phase `(-1)^m`) and then flips the sign for odd m with `n>0`. MY_SWAMP
reproduces this exactly:

```python
odd_m = (np.arange(M + 1) % 2) == 1
Pmn[:, odd_m, 1:] *= -1.0
Hmn[:, odd_m, 1:] *= -1.0
```

The fallback `_lpmn_fallback` also includes the Condon-Shortley phase to match
SciPy's output.

### 0.8 Factorial-based normalization scaling [PARITY-CRITICAL]

**File:** `spectral_transform.py:98-113`

The scaling uses `math.factorial` for exact integer arithmetic, matching the
reference:

```python
sqrt(((2*n+1) * factorial(n-m)) / (2 * factorial(n+m)))
```

This is computed in Python/NumPy at initialization time (not JIT-traced).

### 0.9 Test 1 wind override [PARITY-CRITICAL]

**File:** `model.py:604-607`

When `test==1`, the physical winds U,V are reset to the initial condition each
step, and the spectral wind coefficients are kept fixed. This matches the
reference SWAMPE test 1 behavior.

### 0.10 Dayside strict inequality in forcing [PARITY-CRITICAL]

**File:** `forcing.py:29`

The reference SWAMPE uses strict inequalities for the dayside mask:
`lambdas > -pi/2` and `lambdas < pi/2` (not `>=` or `<=`). MY_SWAMP matches:

```python
dayside = (lambdas > -jnp.pi / 2) & (lambdas < jnp.pi / 2)
```

### 0.11 Q<0 mass-loss prevention and taudrag==-1 [PARITY-CRITICAL]

**File:** `forcing.py:67-87`

The reference SWAMPE zeros out `Q` where `Q < 0` (mass loss prevention), and
handles `taudrag == -1` as a special "no Rayleigh drag" case. MY_SWAMP
reproduces both behaviors using `jnp.where` (JAX-compatible branching).

### 0.12 invrs_leg conjugate symmetry [PARITY-CRITICAL]

**File:** `spectral_transform.py:286-301`

The reference SWAMPE computes negative-m modes via:
```python
negPmn = ((-1)**m) * Pmn[:,m,m:N+1]
negXileg = ((-1)**m) * conj(legcoeff[m,m:N+1])
```
The `(-1)^m` factors cancel (`(-1)^{2m} = 1`), so MY_SWAMP correctly skips
them. The mathematical result is identical. MY_SWAMP also sums over the full
`0:N+1` range rather than `m:N+1`; this is correct because `Pmn[j,m,n]=0` for
`n<m`, so the extra terms contribute zero.

### 0.13 Robert-Asselin / modal splitting filter [PARITY-CRITICAL]

**File:** `filters.py:11-21`, `model.py:610-622`

The Robert-Asselin filter is applied identically to the reference:
```
X_filtered = X(t) + alpha * (X(t-1) - 2*X(t) + X(t+1))
```
Applied to eta, delta, and Phi when `modalflag=True` and `t > 2`.

### 0.14 float64 default [PARITY-CRITICAL]

**File:** `__init__.py`, `dtypes.py`

MY_SWAMP enables `jax_enable_x64=True` at import time and defaults all arrays
to float64/complex128. This matches the reference SWAMPE's NumPy float64
behavior and is essential for numerical parity.

### 0.15 Two-level initialization (leapfrog bootstrap) [PARITY-CRITICAL]

**File:** `model.py` (build_static, state initialization)

The reference SWAMPE initializes two time levels (t=0 and t=1) using the same
initial conditions, as required by the leapfrog scheme. MY_SWAMP preserves this.

### 0.16 fwd_leg Gaussian quadrature weights [PARITY-CRITICAL]

**File:** `spectral_transform.py:270`

The forward Legendre transform uses Gauss-Legendre weights `w[j]` in the
summation, matching the reference:
```python
out[m,n] = sum_j w[j] * data[j,m] * Pmn[j,m,n]
```

### 0.17 Diffusion filter formulation (implicit 4th/6th order) [PARITY-CRITICAL]

**File:** `filters.py:35-105`

The diffusion filters use the implicit formulation `1 / (1 + 2*dt*K*ncoeff)`
matching the reference SWAMPE. The `sigma` function subtracts `factor1` from
`ncoeff` while `sigmaPhi` does not -- this asymmetry is intentional and matches
the original.

---

## Section 1: Parity Divergences (Intentional Improvements)

These are places where MY_SWAMP **intentionally** differs from the reference.

### 1.1 `continuation.compute_timestamp` argument order fix [PARITY-DIVERGENCE]

**Files:** `model.py` (call site), `continuation.py`

The original SWAMPE calls `compute_timestamp(timeunits, dt, t)` but the function
signature is `compute_timestamp(units, t, dt)` -- the `dt` and `t` arguments
are **swapped at the call site** in the original. MY_SWAMP passes them in the
correct order: `(timeunits, int(t), dt)`.

**Impact:** Saved-file timestamps will differ from the original for the same
run. This is a bug fix. If exact file-naming parity is needed for continuation
from original SWAMPE data, add a compatibility shim.

### 1.2 invrsUV_with_coeffs avoids redundant FFTs [PARITY-DIVERGENCE]

**File:** `spectral_transform.py:333-378`

MY_SWAMP adds `invrsUV_with_coeffs` which returns Fourier coefficients
alongside physical-space winds, avoiding a redundant physical-to-spectral FFT
in `_step_once`. The physical outputs (U, V) are identical; this only affects
which intermediate arrays are reused.

---

## Section 2: GPU Efficiency Optimizations (All PARITY-SAFE)

### 2.1 Batch FFTs instead of serial calls [PARITY-SAFE]

**Files:** `model.py:633-635,637`, `spectral_transform.py`

**Current:** Each time step performs 11+ separate `fwd_fft_trunc` calls and
2-6 `invrs_fft` calls. On GPU, each small FFT has significant kernel launch
overhead.

**Fix:** Stack input arrays along a batch dimension and perform a single
`jnp.fft.fft` call:

```python
stacked = jnp.stack([A, B, C, D, E, F_phys, G_phys, PhiF, ...], axis=0)
stacked_m = jnp.fft.fft(stacked / I, n=I, axis=2)[:, :, :M+1]
Am, Bm, Cm, Dm, Em, Fm, Gm, PhiFm, ... = stacked_m
```

**Est. speedup:** 10-20% at M=42, 5-10% at M=63 (kernel launch overhead
fraction decreases with problem size).

### 2.2 Batch Legendre transforms [PARITY-SAFE]

**Files:** `explicit_tdiff.py`, `modEuler_tdiff.py`

**Current:** Each timestep function calls `fwd_leg` 3-8 times with the same
Pmn/Hmn basis.

**Fix:** Stack inputs and use `jax.vmap`:

```python
batched = jnp.stack([data1, data2, data3], axis=0)
results = jax.vmap(lambda d: jnp.einsum("j,jm,jmn->mn", w, d, Pmn))(batched)
```

**Est. speedup:** 5-15% at M=42, 3-8% at M=63.

### 2.3 Eliminate `jnp.tile` -- use broadcasting [PARITY-SAFE]

**Files:** `filters.py:51,68,88,105`, `time_stepping.py:332,336,342,350,355,360`

**Current:** Nine arrays created with `jnp.tile` to broadcast 1-D vectors to 2-D.

**Fix:** Keep as 1-D arrays and let JAX broadcasting handle expansion:

```python
# Instead of: jnp.tile(sigmas[None, :], (M+1, 1))  # shape (M+1, N+1)
# Do:         sigmas[None, :]                        # shape (1, N+1)
```

**Est. speedup:** Marginal runtime improvement but significant memory reduction
in the `Static` pytree carried through every scan step.

### 2.4 Python `if/else` for compile-time flags [PARITY-SAFE]

**Files:** `explicit_tdiff.py:88-89,181-182,257-258`,
`modEuler_tdiff.py:111-112,199-200,262`, `time_stepping.py:323`

**Current:** `forcflag`, `diffflag`, `expflag` are compile-time constants (stored
as `aux_data` in `RunFlags`), but `jax.lax.cond` is used, forcing XLA to trace
and compile **both** branches.

**Fix:** Use plain Python `if/else`:

```python
if forcflag:
    Phiforcing = st.fwd_leg(2.0 * dt * PhiFm, J, M, N, Pmn, w)
    Phimntstep = Phimntstep + Phiforcing
```

**Note:** The `lax.select` calls in items 0.2, 0.3, 0.4 that depend on
`forcflag` should also use Python `if/else` for the same reason. These are
marked PARITY-CRITICAL for their *behavior*, not for their *branching
mechanism* -- changing from `lax.select` to `if/else` preserves parity.

**Est. speedup:** ~2x smaller compiled HLO graph, faster compilation, marginal
runtime improvement.

### 2.5 Remove dead-state `lax.cond` [PARITY-SAFE]

**File:** `model.py:694`

**Current:** `jax.lax.cond(dead_next, skip_update, do_update, ...)` forces XLA
to compile both the full timestepping path and the skip path.

**Fix:** Replace with `jax.lax.select` on outputs, or remove entirely and rely
on Python-level RMS blowup detection.

**Est. speedup:** ~2x smaller compiled graph.

### 2.6 Remove dead-code Legendre transforms in explicit delta [PARITY-SAFE]

**File:** `explicit_tdiff.py:148-157`

**Current:** Three `fwd_leg` calls compute `deltacomp2/3/4` which are
immediately discarded (see item 0.5).

**Fix:** Remove the three dead `fwd_leg` calls. The carry-over behavior
(`deltamntstep = deltacomp1`) is unaffected. The discarded results do not
influence any downstream computation.

**Important:** This IS parity-safe because the discarded transforms have no
effect on the output. The parity-critical aspect (item 0.5) is the
carry-over-only assignment, which is preserved.

**Est. speedup:** 3 fewer `fwd_leg` per explicit delta step (significant when
using the explicit scheme).

### 2.7 Redundant `jnp.real()` calls [PARITY-SAFE]

**Files:** `model.py:597-601` and downstream `ABCDE_init`

**Current:** `jnp.real()` applied to eta, delta, Phi, U, V in `_step_once`,
then applied **again** to U, V, eta, Phi inside `_nonlinear_spectral`.

**Fix:** Remove the redundant second set.

**Est. speedup:** Minor (eliminates redundant array allocations).

### 2.8 FFT normalization optimization [PARITY-SAFE]

**Files:** `spectral_transform.py:240,247`

**Current:** `fwd_fft_trunc` divides `data / I` before FFT (touching all
`J*I` elements), then truncates. `invrs_fft` multiplies by `I` before IFFT.

**Fix:** Use `norm='forward'`/`norm='backward'`:

```python
jnp.fft.fft(data, n=I, axis=1, norm='forward')[:, :M+1]
jnp.fft.ifft(approxXim, n=I, axis=1)  # norm='backward' is default
```

**Est. speedup:** Minor (eliminates one elementwise pass).

### 2.9 Precompute `1j * mJarray` [PARITY-SAFE]

**Files:** `explicit_tdiff.py`, `modEuler_tdiff.py`

**Current:** `(1j) * mJarray` appears ~6 times per time step.

**Fix:** Compute `im_mJarray = 1j * mJarray` once in `build_static` and store
in `Static`.

**Est. speedup:** Minor.

### 2.10 Shared subexpressions in `invrsUV_with_coeffs` [PARITY-SAFE]

**File:** `spectral_transform.py:333-378`

**Current:** `(etamn - fmn) * tstepcoeffmn` and `deltamn * tstepcoeffmn` are
computed implicitly twice (once for Pmn-based leg, once for Hmn-based leg).

**Fix:** Compute once and reuse:

```python
delta_scaled = deltamn * tstepcoeffmn
eta_f_scaled = (etamn - fmn) * tstepcoeffmn
```

**Est. speedup:** Minor (two fewer elementwise multiplications per step).

### 2.11 `fwd_leg` einsum vs explicit matmul [PARITY-SAFE]

**File:** `spectral_transform.py:270`

**Current:** Uses `jnp.einsum("j,jm,jmn->mn", w, data, Pmn)`.

**Consideration:** On GPU, explicit `matmul` via cuBLAS can outperform einsum.
Profile before changing -- XLA may already optimize the einsum path.

---

## Section 3: Code Quality (All PARITY-SAFE)

### 3.1 Extreme parameter lists (37+ positional args) [PARITY-SAFE]

**Files:** `explicit_tdiff.py`, `modEuler_tdiff.py`, `time_stepping.py`

**Fix:** Pass `Static` and `RunFlags` dataclasses directly instead of
unpacking into dozens of positional arguments. This eliminates the 37-arg
copy-paste in `time_stepping.tstepping` (repeated 6 times across
`do_explicit`/`do_modeuler`).

### 3.2 Module naming (PEP 8) [PARITY-SAFE]

**File:** `modEuler_tdiff.py`

PEP 8 requires `snake_case` for module names. Consider renaming to
`mod_euler_tdiff.py` or `modified_euler.py`.

### 3.3 Manual pytree registration [PARITY-SAFE]

**File:** `model.py:116-270`

`RunFlags` and `Static` implement manual `tree_flatten`/`tree_unflatten`.
Consider `jax.tree_util.register_dataclass` (JAX >= 0.4.26) for automatic
registration.

### 3.4 State NamedTuple has 24 fields [PARITY-SAFE]

**File:** `model.py:272-309`

Consider grouping into sub-tuples (`SpectralState`, `PhysicalState`,
`NonlinearState`, `ForcingState`) for readability.

### 3.5 Broad exception handling [PARITY-SAFE]

**File:** `spectral_transform.py:48-50`

`except Exception` should be `except ImportError` for the SciPy import guard.

### 3.6 Redundant `int()` casts in internal functions [PARITY-SAFE]

**Files:** Throughout `spectral_transform.py`, `initial_conditions.py`

Defensive `I = int(I)` casts are useful at API boundaries but unnecessary in
internal JIT-called functions where dimensions are already Python ints.

### 3.7 `_cond` wrapper is unnecessary [PARITY-SAFE]

**Files:** `explicit_tdiff.py:21-23`, `modEuler_tdiff.py:31-33`

The `_cond` wrapper adds `jnp.asarray(pred)` which `jax.lax.cond` already
handles internally. Remove these wrappers (and convert to `if/else` per 2.4).

### 3.8 Redundant Q<0 masking in Rfun [PARITY-SAFE]

**File:** `forcing.py:67-76`

`Qclone` is zeroed where `Q < 0`, making `Ru` zero at those locations. The
subsequent `jnp.where(Q < 0, 0.0, Ru)` is redundant. Remove the second pair.

**Note:** This is a code cleanup, not a parity issue -- the output is identical
either way.

### 3.9 `diagnostic_eta_delta` only used in tests [PARITY-SAFE]

**File:** `spectral_transform.py:381-411`

Not called in the main model loop. Document as test-only or move to test module.

### 3.10 `lru_cache` on JIT wrappers [PARITY-SAFE]

**File:** `model.py:80-113`

JAX already caches compiled functions. The `lru_cache` prevents garbage
collection of compiled functions, which can accumulate memory across many
parameter configurations in optimization sweeps.

---

## Section 4: Summary Table

| # | Item | Tag | Impact | Effort |
|---|------|-----|--------|--------|
| 0.1-0.4 | modEuler coefficient/coupling quirks | PARITY-CRITICAL | -- | -- |
| 0.5 | Explicit delta carry-over only | PARITY-CRITICAL | -- | -- |
| 0.6 | Explicit eta/delta U/taudrag terms | PARITY-CRITICAL | -- | -- |
| 0.7-0.8 | Condon-Shortley + factorial scaling | PARITY-CRITICAL | -- | -- |
| 0.9 | Test 1 wind override | PARITY-CRITICAL | -- | -- |
| 0.10 | Dayside strict inequality | PARITY-CRITICAL | -- | -- |
| 0.11 | Q<0 + taudrag==-1 handling | PARITY-CRITICAL | -- | -- |
| 0.12 | invrs_leg conjugate symmetry | PARITY-CRITICAL | -- | -- |
| 0.13 | Robert-Asselin filter | PARITY-CRITICAL | -- | -- |
| 0.14 | float64 default | PARITY-CRITICAL | -- | -- |
| 0.15 | Two-level initialization | PARITY-CRITICAL | -- | -- |
| 0.16 | fwd_leg Gaussian quadrature | PARITY-CRITICAL | -- | -- |
| 0.17 | Diffusion filter formulation | PARITY-CRITICAL | -- | -- |
| 1.1 | compute_timestamp arg fix | PARITY-DIVERGENCE | File naming | -- |
| 1.2 | invrsUV_with_coeffs | PARITY-DIVERGENCE | Avoids FFT | -- |
| 2.1 | Batch FFTs | PARITY-SAFE | 10-20% | Medium |
| 2.2 | Batch Legendre transforms | PARITY-SAFE | 5-15% | Medium |
| 2.3 | Eliminate jnp.tile | PARITY-SAFE | Memory | Low |
| 2.4 | Python if/else for flags | PARITY-SAFE | 2x smaller HLO | Low |
| 2.5 | Remove dead-state lax.cond | PARITY-SAFE | 2x smaller graph | Low |
| 2.6 | Remove dead fwd_leg calls | PARITY-SAFE | 3 fewer fwd_leg | Low |
| 2.7 | Remove redundant jnp.real() | PARITY-SAFE | Minor | Low |
| 2.8 | FFT norm argument | PARITY-SAFE | Minor | Low |
| 2.9 | Precompute 1j*mJarray | PARITY-SAFE | Minor | Low |
| 2.10 | Shared subexpressions in invrsUV | PARITY-SAFE | Minor | Low |
| 2.11 | einsum vs matmul | PARITY-SAFE | Profile first | Low |
| 3.1 | Refactor parameter lists | PARITY-SAFE | Maintainability | Medium |
| 3.2 | PEP 8 module naming | PARITY-SAFE | Convention | Low |
| 3.3 | Simplify pytree registration | PARITY-SAFE | Maintainability | Low |
| 3.4 | Group State fields | PARITY-SAFE | Readability | Medium |
| 3.5-3.10 | Code cleanup items | PARITY-SAFE | Polish | Low |

---

## Section 5: Recommended Implementation Order

### Phase 1: Low-hanging fruit (no parity risk, low effort)

1. Python `if/else` for compile-time flags (2.4)
2. Remove dead `fwd_leg` calls in explicit delta (2.6)
3. Remove dead-state `lax.cond` (2.5)
4. Eliminate `jnp.tile` (2.3)
5. Remove redundant `jnp.real()` (2.7)
6. Remove redundant Q masking (3.8)
7. Remove `_cond` wrapper (3.7)
8. Precompute `1j*mJarray` (2.9)
9. Shared subexpressions in invrsUV (2.10)
10. FFT norm argument (2.8)

### Phase 2: Medium effort, high impact

11. Batch FFTs (2.1)
12. Batch Legendre transforms (2.2)
13. Refactor parameter lists to dataclasses (3.1)

### Phase 3: Polish

14. Module renaming (3.2)
15. Simplify pytree registration (3.3)
16. Group State fields (3.4)
17. Profile einsum vs matmul (2.11)

---

## Estimated Combined Speedups

| Resolution | Phase 1 | Phase 1+2 | All |
|-----------|---------|-----------|-----|
| M=42 | 1.5-2x | 4-7x | 4-7x |
| M=63 | 1.3-1.8x | 3-5x | 3-5x |
| M=106 | 1.2-1.5x | 2-4x | 2-4x |

Speedup decreases at higher resolution because GPU compute increasingly
dominates over kernel launch overhead.
