# MY_SWAMP: Needed Updates

A comprehensive analysis of the JAX SWAMPE port (`MY_SWAMP`) compared to the
reference NumPy SWAMPE, covering physical parity, GPU efficiency, and code
quality.

---

## 1. Physical Parity Issues

These are places where MY_SWAMP may diverge numerically from the original
SWAMPE. Some are intentional bug fixes; all should be verified.

### 1.1 `continuation.compute_timestamp` argument order mismatch

**Files:** `model.py:1395`, original `SWAMPE/model.py:409`

The original SWAMPE calls `compute_timestamp(timeunits, dt, t)` but the function
signature is `compute_timestamp(units, t, dt)` -- the arguments `dt` and `t` are
**swapped at the call site** in the original. MY_SWAMP correctly passes
`(timeunits, int(t), dt)`, which means saved-file timestamps will **differ**
from the original for the same run. This is a bug fix, but if exact file-naming
parity is needed for continuation from original SWAMPE data, a compatibility
shim should be considered.

### 1.2 `explicit_tdiff.delta_timestep` computes then discards terms

**Files:** `explicit_tdiff.py:146-160`

The original SWAMPE has `deltamntstep = deltacomp1` (with the additional terms
commented out). MY_SWAMP faithfully reproduces this quirk, **but it still
computes** `deltacomp2`, `deltacomp3`, and `deltacomp4` and then throws them
away. These are wasted forward Legendre transforms (3 extra `fwd_leg` calls per
explicit delta timestep). Since the results are discarded, they should be
removed entirely -- keeping them "for parity" serves no purpose and wastes
significant compute.

### 1.3 Redundant Q<0 masking in `forcing.Rfun`

**File:** `forcing.py:67-76`

The code zeros `Qclone` where `Q < 0`, then computes `Ru = -U * Qclone / phi_total`,
then zeros `Ru` where `Q < 0` again. The second `jnp.where` is redundant since
`Qclone` is already zero at those locations (making `Ru` zero). The duplicate
masking wastes a comparison + select per element. Remove the second pair of
`jnp.where` calls.

### 1.4 `invrs_leg` negative-m symmetry -- mathematically equivalent but verify

**File:** `spectral_transform.py:286-301`

The original SWAMPE applies `(-1)^m` to both `Pmn` and `conj(legcoeff)` for
negative m, which cancel (`(-1)^{2m} = 1`). MY_SWAMP correctly skips this
cancellation and directly computes:
```python
neg = jnp.einsum("jmn,mn->jm", Pmn[:, 1:, :], jnp.conj(legcoeff[1:, :]))
```
This is mathematically equivalent and more efficient. However, the original
sums only from `m:N+1` (since `Pmn[j,m,n]=0` for `n<m`), while MY_SWAMP sums
over the full `0:N+1` range. This is still correct (zero contributions), but
does extra multiply-adds on the GPU. Consider masking or slicing to `m:N+1` for
efficiency if profiling shows it matters at higher resolutions.

---

## 2. GPU Efficiency -- Critical Optimizations

These are the highest-impact changes for GPU throughput.

### 2.1 Batch FFTs instead of serial calls

**Files:** `model.py:433-437,458-466,633-635`, `spectral_transform.py`

Each time step performs **11+ separate `fwd_fft_trunc` calls** (5 for ABCDE
nonlinear terms, 3 for forcing F/G/PhiF, and 3 for eta/delta/Phi prognostic
fields) and **2-6 `invrs_fft` calls**. On a GPU, each small FFT has
significant kernel launch overhead relative to the actual computation.

**Fix:** Stack the input arrays along a new batch dimension and perform a single
`jnp.fft.fft` call over the batch. Then unstack. For M=42, this replaces 11
kernel launches with 1.

```python
# Instead of:
Am = fwd_fft_trunc(A, I, M)
Bm = fwd_fft_trunc(B, I, M)
# ...

# Do:
stacked = jnp.stack([A, B, C, D, E, F_phys, G_phys, PhiF, ...], axis=0)
stacked_m = jnp.fft.fft(stacked / I, n=I, axis=2)[:, :, :M+1]
Am, Bm, Cm, Dm, Em, Fm, Gm, PhiFm, ... = stacked_m
```

Similarly batch the `invrs_fft` calls.

### 2.2 Batch Legendre transforms

**Files:** `explicit_tdiff.py`, `modEuler_tdiff.py`

Each time-stepping function calls `fwd_leg` 3-8 times with the same `Pmn` or
`Hmn` basis. These are independent matrix multiplications that can be batched
using `vmap` or by stacking inputs:

```python
# Instead of 3 separate fwd_leg calls with Pmn:
# comp1 = fwd_leg(data1, ..., Pmn, w)
# comp2 = fwd_leg(data2, ..., Pmn, w)
# comp3 = fwd_leg(data3, ..., Pmn, w)

# Do:
batched = jnp.stack([data1, data2, data3], axis=0)  # (3, J, M+1)
results = jax.vmap(lambda d: jnp.einsum("j,jm,jmn->mn", w, d, Pmn))(batched)
comp1, comp2, comp3 = results
```

This is particularly impactful in `modEuler_tdiff.phi_timestep` which has 7
`fwd_leg` calls.

### 2.3 Eliminate `jnp.tile` -- use broadcasting instead

**Files:** `filters.py:51,68,88,105`, `time_stepping.py:332,336,342,355,360`

Nine separate arrays are created using `jnp.tile` to broadcast a 1-D vector
into a 2-D array. JAX and XLA handle broadcasting natively; the tiled arrays
waste memory (e.g., `sigma` stores `M+1` copies of the same length-`N+1`
vector).

**Fix:** Keep these as 1-D arrays and let broadcasting handle the rest:

```python
# Instead of:
return jnp.tile(sigmas[None, :], (M + 1, 1))  # shape (M+1, N+1)

# Do:
return sigmas[None, :]  # shape (1, N+1), broadcasts with (M+1, N+1) inputs
```

This halves memory for `sigma`, `sigmaPhi`, `tstepcoeffmn`, `narray`, `marray`,
`mJarray`, `tstepcoeff`, `tstepcoeff2` -- all of which are carried in the
`Static` pytree through every scan step.

### 2.4 `lax.cond` for compile-time-constant flags traces both branches

**Files:** `explicit_tdiff.py:88-89,181-182,257-258`,
`modEuler_tdiff.py:111-112,199-200,262`, `time_stepping.py:323`

`forcflag`, `diffflag`, and `expflag` are stored as `aux_data` in `RunFlags`
(compile-time constants that don't change between JIT calls with the same
flags). Using `jax.lax.cond` forces XLA to trace and compile **both** branches,
doubling the HLO graph size and compilation time.

**Fix:** Since these flags are Python bools known at trace time, use plain
Python `if/else`:

```python
# Instead of:
Phimntstep = _cond(forcflag, _add_forcing, lambda x: x, Phimntstep)

# Do:
if forcflag:
    Phiforcing = st.fwd_leg(2.0 * dt * PhiFm, J, M, N, Pmn, w)
    Phimntstep = Phimntstep + Phiforcing
```

This eliminates the dead branch from the compiled XLA program entirely.

### 2.5 Dead-state branching in `_step_once` doubles the compiled graph

**File:** `model.py:694`

The `jax.lax.cond(dead_next, skip_update, do_update, ...)` forces XLA to
compile both the full time-stepping path and the skip path. The skip path is
essentially a no-op but the full `do_update` closure is enormous.

**Fix:** Replace with `jax.lax.select` on the outputs, or remove the dead-state
check entirely and rely on the RMS blowup detection at the Python level (since
`diagnostics=True` already tracks RMS). For optimization loops where
`diagnostics=False`, the dead-state branch is already skipped.

### 2.6 Redundant `jnp.real()` calls

**Files:** `model.py:597-601` and `model.py:424-428`

`jnp.real()` is applied to eta, delta, Phi, U, V in `_step_once`, and then
`jnp.real()` is applied **again** to U, V, eta, Phi inside
`_nonlinear_spectral` (via `initial_conditions.ABCDE_init`). Each call
allocates a new array. Remove the redundant set.

### 2.7 FFT normalization wastes FLOPs

**Files:** `spectral_transform.py:240,247`

`fwd_fft_trunc` divides `data / I` **before** the FFT (touching all `J * I`
elements), then truncates to `M+1` columns. `invrs_fft` multiplies by `I`
before the inverse FFT.

**Fix:** Use `jnp.fft.fft(data, n=I, axis=1) / I` instead -- the division
happens on the output which has the same size. Or better, use the `norm`
argument: `jnp.fft.fft(data, n=I, axis=1, norm='forward')` which applies the
`1/I` normalization automatically without an extra pass. Similarly for
`invrs_fft`, use `norm='backward'` and skip the explicit `I *` multiplication.

### 2.8 `invrsUV_with_coeffs` -- shared subexpressions not reused

**File:** `spectral_transform.py:333-378`

`invrsUV_with_coeffs` computes 4 separate `invrs_leg` calls. Two of them use
`marray * deltamn * tstepcoeffmn` and `(etamn - fmn) * tstepcoeffmn` which
are also used by the other two (with different bases Pmn vs Hmn). The
intermediate products `(etamn - fmn) * tstepcoeffmn` and
`deltamn * tstepcoeffmn` should be computed once and reused.

### 2.9 `fwd_leg` einsum may not be optimal on GPU

**File:** `spectral_transform.py:270`

`jnp.einsum("j,jm,jmn->mn", w, data, Pmn)` is a weighted contraction. On GPU,
explicit `matmul` is typically faster because cuBLAS is highly optimized:

```python
# Equivalent using matmul:
weighted_data = w[:, None] * data  # (J, M+1), broadcast multiply
# For each m, contract over j: result[m,n] = sum_j weighted_data[j,m] * Pmn[j,m,n]
# This is a batched dot product along m:
result = jnp.sum(weighted_data[:, :, None] * Pmn, axis=0)
```

Or reshape and use `jnp.tensordot`. Profile to confirm, as XLA may already
optimize the einsum path.

---

## 3. Code Quality / Pythonic Issues

### 3.1 Extreme parameter lists (37+ positional args)

**Files:** `explicit_tdiff.py`, `modEuler_tdiff.py`, `time_stepping.py`

Every time-stepping function takes 37+ positional arguments that are passed
through multiple levels of indirection. This is error-prone and hard to
maintain.

**Fix:** Pass the `Static` and `RunFlags` dataclasses directly instead of
unpacking them into dozens of positional arguments:

```python
# Instead of:
def phi_timestep(etam0, etam1, deltam0, deltam1, Phim0, Phim1,
                 I, J, M, N, Am, Bm, Cm, Dm, Em, Fm, Gm, Um, Vm,
                 Pmn, Hmn, w, tstepcoeff1, tstepcoeff2, mJarray, narray,
                 PhiFm, dt, a, Phibar, taurad, taudrag, forcflag, diffflag,
                 sigma, sigmaPhi, test, t):

# Do:
def phi_timestep(state_m, static, flags, t):
```

This also eliminates the massive copy-paste of argument lists in
`time_stepping.tstepping` (which repeats the 37-arg list **6 times** across
`do_explicit` and `do_modeuler`).

### 3.2 `modEuler_tdiff.py` / `explicit_tdiff.py` file naming

**Files:** `modEuler_tdiff.py`

PEP 8 requires `snake_case` for module names. Rename to `mod_euler_tdiff.py`
(or `modified_euler.py`).

### 3.3 `RunFlags` / `Static` manual pytree registration

**Files:** `model.py:116-148,150-270`

Both `RunFlags` and `Static` implement manual `tree_flatten` / `tree_unflatten`
with explicit child/aux_data splitting. This is verbose and fragile (adding a
field requires updating 3 places).

**Fix:** Use `jax.tree_util.register_dataclass` (available since JAX 0.4.26) or
`flax.struct.dataclass`:

```python
@jax.tree_util.register_dataclass
@dataclass
class Static:
    M: int = jax.tree_util.static_field()
    N: int = jax.tree_util.static_field()
    ...
    dt: jnp.ndarray
    a: jnp.ndarray
    ...
```

### 3.4 `State` NamedTuple has 24 fields

**File:** `model.py:272-309`

The `State` carries 24 arrays through every `lax.scan` iteration. Some
grouping would improve readability:

```python
class SpectralState(NamedTuple):
    etam_prev: jnp.ndarray
    etam_curr: jnp.ndarray
    deltam_prev: jnp.ndarray
    deltam_curr: jnp.ndarray
    Phim_prev: jnp.ndarray
    Phim_curr: jnp.ndarray

class PhysicalState(NamedTuple):
    eta_prev: jnp.ndarray
    eta_curr: jnp.ndarray
    ...

class State(NamedTuple):
    spectral: SpectralState
    physical: PhysicalState
    nonlinear: NonlinearState
    forcing: ForcingState
    dead: jnp.ndarray
```

### 3.5 Unused imports and broad exception handling

**Files:** `spectral_transform.py:48-50`, `dtypes.py:62`, `__init__.py:44`

- `except Exception` is overly broad for SciPy import failures. Use
  `except ImportError`.
- `_SCIPY_IMPORT_ERROR` is stored but never referenced.

### 3.6 `_is_python_scalar` and `_tree_has_tracer` should be private utilities

**File:** `model.py:48-56`

These are module-level utility functions that could be in a shared `_utils.py`
or at minimum should have leading underscores (which they do). But
`_tree_has_tracer` iterates all leaves of a pytree on every call, which is
O(n) in the number of State fields. Consider caching or simplifying.

### 3.7 Inconsistent use of `int()` casts

**Files:** Throughout `spectral_transform.py`, `initial_conditions.py`,
`filters.py`

Many functions start with `I = int(I)`, `J = int(J)`, etc. These are defensive
casts that guard against JAX tracers being passed for dimensions. While
reasonable at public API boundaries, they are unnecessary in internal functions
called from within a JIT context where dimensions are already Python ints.
Consider removing from internal functions and keeping only at entry points.

---

## 4. Additional Efficiency Opportunities

### 4.1 Precompute and cache `1j * mJarray` and `1j * marray`

**Files:** `explicit_tdiff.py`, `modEuler_tdiff.py`

The expression `(1j) * mJarray` appears ~6 times per time step across the
eta/delta/Phi timestep functions. Since `mJarray` is static, precompute
`im_mJarray = 1j * mJarray` once in `build_static` and store it in `Static`.

### 4.2 The `_cond` wrapper function is unnecessary indirection

**Files:** `explicit_tdiff.py:21-23`, `modEuler_tdiff.py:31-33`

```python
def _cond(pred, true_fun, false_fun, operand):
    return jax.lax.cond(jnp.asarray(pred), true_fun, false_fun, operand)
```

The `jnp.asarray(pred)` is already done inside `jax.lax.cond`. And as noted in
2.4, these should be plain `if/else` anyway. Remove these wrappers.

### 4.3 `diagnostic_eta_delta` is only used in `test_unit.py`

**File:** `spectral_transform.py:381-411`

This function is not called in the main model loop. If it's only for testing,
it should be documented as such or moved to the test module.

### 4.4 `run_model` recomputes initial conditions for diagnostics

**File:** `model.py:1328-1388`

After the full scan completes, `run_model` recomputes `test1_init`,
`state_var_init`, and `velocity_init` to populate the index-0 spinup
diagnostics. These initial conditions were already computed earlier. Cache or
pass them through.

### 4.5 `run_model_scan_final` is a trivial wrapper

**File:** `model.py:1110-1194`

`run_model_scan_final` is just `run_model_scan(return_history=False, ...)`.
It adds no logic. Consider removing and documenting the `return_history`
parameter instead.

### 4.6 `lru_cache` on JIT wrappers creates subtle issues

**Files:** `model.py:80-113`

`_get_simulate_scan_jit` and `_get_simulate_scan_last_jit` use `lru_cache` to
avoid re-JIT-ting. But JAX already caches compiled functions via its own
tracing cache. The `lru_cache` here prevents garbage collection of the compiled
function, which can accumulate memory across many different parameter
configurations in long-running optimization sweeps.

---

## 5. Summary: Prioritized Action Items

### High Priority (physical correctness + major perf)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | Batch FFTs (2.1) | ~5-10x fewer kernel launches per step | Medium |
| 2 | Python `if/else` for compile-time flags (2.4) | ~2x smaller compiled HLO | Low |
| 3 | Remove dead-code Legendre transforms in explicit delta (1.2) | 3 fewer `fwd_leg` per step | Low |
| 4 | Eliminate `jnp.tile` (2.3) | Memory reduction, cleaner | Low |
| 5 | Remove dead-state `lax.cond` (2.5) | Halve compiled graph size | Low |

### Medium Priority (significant perf + code quality)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 6 | Batch Legendre transforms (2.2) | Fewer kernel launches | Medium |
| 7 | Refactor parameter lists to use dataclasses (3.1) | Maintainability | Medium |
| 8 | Precompute `1j * mJarray` (4.1) | Eliminate repeated work | Low |
| 9 | Remove redundant `jnp.real()` (2.6) | Minor perf | Low |
| 10 | FFT `norm` argument (2.7) | Eliminate intermediate allocs | Low |

### Low Priority (polish)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 11 | Rename `modEuler_tdiff.py` (3.2) | PEP 8 compliance | Low |
| 12 | Simplify pytree registration (3.3) | Maintainability | Low |
| 13 | Group State fields (3.4) | Readability | Medium |
| 14 | Remove redundant Q masking in Rfun (1.3) | Minor perf | Low |
| 15 | Remove `run_model_scan_final` wrapper (4.5) | API simplification | Low |
