# Remaining Updates

Last checked: 2026-03-03 (PST)

Already-fixed items were removed. This file now lists only unresolved work.

## Current Verified Status

1. Correctness checks pass locally on CPU:
   - `JAX_PLATFORMS=cpu pytest -q` -> `20 passed`
   - `JAX_PLATFORMS=cpu pytest -q -m smoke` -> `5 passed, 15 deselected`
   - `JAX_PLATFORMS=cpu pytest -q -m parity` -> `4 passed, 16 deselected`
   - `JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=0 JAX_ENABLE_X64=0 pytest -q -m parity` -> expected `4 failed` (x64 gate enforced)
2. Lint recheck is complete in `swamp_compare`:
   - `ruff check src tests testing notebooks/nss.py --select F401,F841` -> pass
   - `ruff check src tests testing notebooks/nss.py` -> pass
   - `python -m vulture src tests testing notebooks/nss.py` -> pass (with explicit `pyproject.toml` ignore list for known callback/API-hook symbols)
3. CPU benchmark harness runs:
   - `testing/benchmark_scan.py --backend cpu --M 42 --dt 30 --tmax 30 ...`
   - result snapshot: `compile=1.282s`, `runtime_median=0.0483s`, `per_step_median=1.72ms`
4. Previously-open efficiency/code issues are now fixed:
   - `donate_state=True` works (no donation crash) and benchmarks run.
   - `remat_step=True` with `test=1` works (no `TracerBoolConversionError`).
   - NSS loop host-sync pressure was reduced by removing per-step non-logging `device_get` stop checks.
5. CPU efficiency sweep rerun (sequential, `tmax=120`) for retrieval-relevant knobs:
   - baseline (`diagnostics=false, donate=false, remat=false`): `per_step_median=1.425ms`
   - `donate=true`: `1.427ms`
   - `remat=true`: `1.432ms`
   - Interpretation: on this CPU, `donate`/`remat` are now stable but do not provide a speedup for this workload.
6. Retrieval guidance from measured results:
   - Keep `diagnostics=false` in inference/retrieval loops.
   - Keep `donate_state=false` and `remat_step=false` by default unless you hit memory pressure on target GPU hardware.
   - Re-tune these knobs again on the actual GPU machine; CPU-optimal settings are not guaranteed to transfer.

## Still Needed Before Claiming "Maximally Efficient" (GPU + NSS Focus)

1. GPU backend is still unavailable on this machine.
   - `testing/benchmark_scan.py --backend gpu ...` fails with:
     `RuntimeError: Requested backend 'gpu' is unavailable.`
   - Cannot yet claim GPU runtime efficiency or backend parity for retrieval workloads without supported hardware.
