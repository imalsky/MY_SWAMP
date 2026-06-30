# SWAMPE vs SWAMPE-JAX — speed benchmark

Combined record of the CPU and GPU timing for the paper's "Speed (CPU and GPU)" section.
All three rows use the **identical** forced hot-Jupiter configuration and float64 precision;
the only thing that differs is the implementation/device.

## Configuration (identical for all rows)

`M=42`, `dt=120 s`, `Phibar=3e5`, `omega=3.2e-5`, `a=8.2e7`, `DPhieq=1e6`,
`taurad=10 h (36000 s)`, `taudrag=6 h (21600 s)`, `K6=1.24e33`, forced mode,
modified-Euler scheme, hyperdiffusion + modal filter on. Steps counted as `tmax-2`.

## Common metric: time **per step** (run-length independent)

| Implementation        | Device                 | per-step (ms) | speedup vs SWAMPE | 10-day total | 100-day total |
|-----------------------|------------------------|--------------:|------------------:|-------------:|--------------:|
| SWAMPE (NumPy)        | CPU (1 core)           |       121.0   |            1.0×   |      871 s   |   ~2.42 h (extrap.) |
| SWAMPE-JAX (my_swamp) | CPU (1 core)           |         3.99  |           30.3×   |     28.7 s   |   ~4.8 min (extrap.) |
| SWAMPE-JAX (my_swamp) | GPU (NVIDIA A100-40GB) |        0.2335 |          518×     |   1.68 s (derived) | 16.81 s |

- **my_swamp GPU vs my_swamp CPU** (same code, cross-device): **17.1×**.
- 10-day totals: a 10-day run is 7,199 steps (`tmax=7201`). 100-day is 71,999 steps.
- "derived"/"extrap." values are `per-step × steps` (the per-step rate is what is measured;
  the loop is fixed-cost so this is exact).

## Provenance

- **CPU** (both rows): `figures/long_run_parity_outputs/forced_default_100d/summary.json` —
  a 10-day forced run on a single CPU core. `runtime_seconds = {swampe: 871.211, my_swamp: 28.734}`.
  Same file's parity: `Phi` max-abs `6.3e-8` (rel `3.7e-10`); `eta`/`delta` ~`1e-12`.
- **GPU**: Colab `NVIDIA A100-SXM4-40GB`, 100-day run, float64, **averaged over 5 timed runs**:
  avg total `16.814 s` (std `0.035 s`), avg per-step `0.2335 ms`. JIT compile excluded
  (warmup run first); `block_until_ready()` before stopping the clock; terminal state only
  (no history, no disk I/O); result finite. Per-run totals: 16.871, 16.763, 16.806, 16.826, 16.805 s.

## Fairness notes

- Install excluded; JIT compile excluded (warmup); device sync via `block_until_ready()`.
- No disk I/O (SWAMPE `saveflag=False`; my_swamp returns terminal state only).
- CPU and GPU are necessarily different hardware (SWAMPE is CPU-only NumPy); the cleanest
  same-machine, same-code number is the CPU→GPU `17.1×`. The `518×` is my_swamp-GPU vs the
  reference SWAMPE-CPU rate.

## Batched throughput (`jax.vmap`, GPU)

`jax.vmap` runs many trajectories (here a `DPhieq` sweep) in one compiled call. The efficient
batch size is set by **throughput**, not memory: at `M=42` each step is tiny, so the A100 stays
underutilized until enough trajectories are stacked to saturate it; past that, batch time scales
with `N` and throughput plateaus. Sweep on the A100 (1-day = 719-step runs, float64), doubling
`N` until throughput plateaus:

| N  | batch time (s) | throughput (traj/s) | ms/traj | ms/step/traj |
|---:|---------------:|--------------------:|--------:|-------------:|
| 1  | 0.125 |  8.0 | 124.85 | 0.1736 |
| 2  | 0.133 | 15.1 |  66.39 | 0.0923 |
| 4  | 0.152 | 26.3 |  38.05 | 0.0529 |
| 8  | 0.175 | 45.6 |  21.93 | 0.0305 |
| 16 | 0.239 | 67.0 |  14.92 | 0.0208 |
| 32 | 0.414 | 77.4 |  12.93 | 0.0180 |
| 64 | 0.789 | 81.1 |  12.33 | 0.0171 |

- **Efficient knee: N = 32** (77.4 traj/s, within 10% of peak); **peak: N = 64** (81.1 traj/s).
  Beyond that, doubling `N` just scales batch time linearly — no efficiency gain.
- At N=32, batching is **~9.7× more efficient per trajectory** than running trajectories one at
  a time, at only **163 MB** peak GPU memory (compute-bound, not memory-bound).
- Per-member final `Phi` tracks the swept `DPhieq` (94k → 384k mean), confirming the batch
  members are genuinely distinct runs; all outputs finite.
- Provenance: Colab `NVIDIA A100-SXM4-40GB`, float64, 1-day runs, JIT compile excluded
  (warmup), `block_until_ready()`. Notebook: `swampe_gpu_vmap_test.ipynb`.
