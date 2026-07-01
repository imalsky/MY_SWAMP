# Benchmark provenance data

Raw, committed outputs backing every quantitative claim in `paper/paper.tex`'s
"Numerical parity" and "Speed (CPU and GPU)" subsections, plus the script/notebook
that generated each one. This directory exists because earlier drafts of the paper
quoted benchmark numbers whose source data lived only in gitignored directories
(`figures/`) or an uncommitted notebook -- a reviewer (and a careful read of the
repo) could not actually verify them. Everything here is small (JSON/Markdown
text, no `.npz`/`.png`), and `paper/` is not covered by any `.gitignore` rule for
these extensions (only `figures/` and the bulky regenerable outputs are excluded),
so these files are tracked by git with no special-casing needed.

**2026-06-30 reorganization**: the generator scripts (`compare_long_run_parity.py`,
`make_sensitivity_figure.py`, `benchmark_gradient.py`, `swampe_gpu_vmap_test.py`/
`.ipynb`) used to live in the top-level `tests/` and `scripts/` directories and were
invoked via `../tests/...` / `../scripts/...` from `paper/Makefile`. They were moved
into `paper/scripts/` so that `paper/` is fully self-contained -- a reviewer (or a
future Claude session) can find the paper, its figures, its raw data, and the exact
code that produced all three in one place, with nothing to chase outside this
directory except the installed `my_swamp` package itself and the sibling reference
repo `../../SWAMPE`. General-purpose package tooling that is *not* paper-specific
(`scripts/benchmark_scan.py`, a generic perf microbenchmark; `scripts/
generate_reference_parity_fixtures.py`, which feeds the pytest suite's fixtures, not
the paper) deliberately stayed where it was -- moving those would have made the
*package's* validation tooling harder to find for someone not working on the paper.

## Window convention (read this first)

The paper deliberately uses **two different integration windows**, and that is
intentional, not an inconsistency:

- **100-day window** -- used by `Numerical parity` (Figure 1) and `Differentiability`
  (Figure 2). Long enough to show stable long-horizon parity and a settled
  forward-mode sensitivity field. Source: `paper/scripts/compare_long_run_parity.py
  --days 100` and `paper/scripts/make_sensitivity_figure.py --days 100`.
- **10-day window** -- used by `Speed (CPU and GPU)`, consistently for the CPU
  comparison *and* the GPU single-trajectory *and* the GPU batched (`vmap`) numbers.
  Earlier drafts mixed a 10-day CPU claim with the 100-day parity claim presented in
  the same subsection-grouped narrative, and a since-deleted/unreproducible 10-day
  output directory -- both flagged in review. The fix was not to "make everything
  100-day" or vice versa; it was to make the *Speed* subsection internally consistent
  at one window (10 days), and keep that window's source files committed here so it
  never silently drifts again.

**Do not extrapolate one window's numbers into the other's.** If a number for the
other window is needed, regenerate it explicitly with `--days <N>` and add a new
file here -- don't scale an existing number by the day ratio (the earlier 10-day
vs. 100-day mismatch happened in part because per-step cost is *not* perfectly
window-invariant: JIT-compile overhead is a larger fraction of a short run's total
wall-clock than a long run's).

## Files

| File | What it is | How it was generated |
|---|---|---|
| `cpu_parity_100day_summary.json` | Copy of the 100-day SWAMPE-vs-`my_swamp` field-error + runtime summary backing Figure 1 and the parity claim. `runtime_seconds = {swampe: 8654.59, my_swamp: 240.71}`. | `JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 python paper/scripts/compare_long_run_parity.py --days 100` (also run via `paper/Makefile`'s `figures` target). Original output: `paper/figures/long_run_parity_outputs/forced_default_100d/summary.json` (gitignored; this is a tracked copy). Machine: Apple M3 Pro, single CPU core, on a quiet system (load average confirmed before starting). |
| `cpu_speed_10day_summary.json` | Copy of the 10-day SWAMPE-vs-`my_swamp` runtime summary backing the CPU half of the Speed claim. `runtime_seconds = {swampe: 817.353, my_swamp: 25.220}` (CPU speedup ~32.4x). | `JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 python paper/scripts/compare_long_run_parity.py --days 10 --out-dir paper/figures/long_run_parity_outputs/forced_default_10d`. Same machine/conditions as above. The `--out-dir` override exists specifically so a 10-day run can never again silently overwrite the 100-day run's output directory (they used to share one path). |
| `gpu_vmap_sweep_10day.json` | The GPU single-trajectory (`N=1`) and batched-throughput (`jax.vmap`, `N` up to 64) sweep backing the GPU half of the Speed claim, all at the 10-day window. | `paper/scripts/swampe_gpu_vmap_test.ipynb`, run by the package author on a Google Colab `NVIDIA A100-SXM4-40GB` GPU runtime, `Run all`. Pasted back verbatim from the notebook's `===== RESULTS =====` block; see `_provenance` in the JSON. A CLI-equivalent, `paper/scripts/swampe_gpu_vmap_test.py`, exists for non-Colab GPU machines and produces the same schema (plus its own JSON dump) but was not the source of the numbers actually quoted in the paper -- the notebook was. |

## Reproducing

```bash
# Parity (100-day) -- also regenerates Figure 1's PNG:
cd paper && make figures

# Speed, CPU half (10-day):
cd paper && make speed
# equivalent, run from the repo root:
JAX_PLATFORMS=cpu SWAMPE_JAX_ENABLE_X64=1 python paper/scripts/compare_long_run_parity.py \
    --days 10 --out-dir paper/figures/long_run_parity_outputs/forced_default_10d

# Speed, GPU half (10-day) -- needs a GPU:
python paper/scripts/swampe_gpu_vmap_test.py --sweep-days 10
# or open paper/scripts/swampe_gpu_vmap_test.ipynb in Colab (GPU runtime) and Run all.
```

Both CPU runs require the sibling reference repo `../SWAMPE` (relative to the repo
root `MY_SWAMP/`; not shipped in this package -- see `README.md` SS2). Run them on
an otherwise-idle machine -- timing fidelity is a wall-clock measurement, and a
loaded machine will inflate both sides unevenly. (This bit the author once during
this very revision: an earlier rerun was contaminated by leftover load from
unrelated parallel work and had to be discarded and restarted on a confirmed-idle
system; a second attempt was accidentally launched twice concurrently and also had
to be discarded.)

## When you add a new benchmark number to the paper

1. Decide which window it belongs to (100-day parity/differentiability, or 10-day
   speed) -- don't invent a third window without a strong reason, and if you do,
   document it here the same way.
2. Generate it with the commands above (or the GPU notebook), save the raw
   JSON/log here, and reference the file in this table.
3. Update `paper/paper.tex` to match, exactly, with no rounding beyond what the
   paper already does elsewhere. **Substitute the numbers only -- do not rewrite,
   expand, or add caveats to the surrounding prose.** If a caveat or methodology
   detail genuinely needs recording, put it in this file or `speed_benchmark.md`,
   not in `paper.tex` itself, unless the author explicitly asks for prose changes.
   (On 2026-06-30 a round of "fully correct the paper" edits added several
   paragraphs -- a checkpointing/AD discussion, a retrieval-pipeline paragraph,
   new citations -- that the author then asked to be reverted because only the
   numbers were supposed to change. Don't repeat that.)
4. If a script changed in a way that affects past numbers (e.g. a different
   `DPhieq` sweep range, a different warmup protocol), say so in this README --
   future readers (including a future Claude session working on this repo) should
   be able to tell *why* two numbers in git history differ without re-deriving it
   from the diff alone.
5. Before running a CPU benchmark, check `uptime` / `ps aux` for unrelated heavy
   processes (including other agent sessions) and wait for a quiet system. Never
   launch a second instance of the same benchmark while one is already running --
   check `ps aux | grep compare_long_run_parity` first.
