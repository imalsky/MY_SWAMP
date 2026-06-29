# SWAMPE-JAX JOSS paper

The paper is maintained as **LaTeX** (`paper.tex` + `paper.bib`). This `.tex`
source is the canonical, authoritative version of the paper — there is no
`paper.md`. `paper.tex` reproduces the JOSS PDF layout (logo header, left
metadata sidebar, CC-BY footer) and builds locally with `make`.

## Build

```bash
make            # pdflatex -> bibtex -> pdflatex x2  ->  paper.pdf
make clean      # remove LaTeX build artifacts
```

Requires a LaTeX install with `pdflatex` + `bibtex` (uses `natbib` + `plainnat`).

## Figures (reproducible; not committed)

The figure **scripts** are committed and the figure **PNGs are regenerated**, not
stored in git (the repository does not track large binaries or data). Only the
small `joss-logo.png` asset is committed, because it is required for the build and
is not regenerable.

| Figure | Script | Command |
|---|---|---|
| Fig. 1 — SWAMPE vs SWAMPE-JAX parity (`parity_comparison.png`) | `../testing/compare_long_run_parity.py` | `make figures` (or see below) |
| Fig. 2 — AD sensitivity maps (`temperature_sensitivity_perhour_100d.png`) | `../testing/make_sensitivity_figure.py` | `python ../testing/make_sensitivity_figure.py` |

Regenerate both before building (slow: Fig. 1 runs the reference NumPy SWAMPE and
the JAX model; Fig. 2 runs a 100-day differentiated integration):

```bash
make figures    # regenerates parity_comparison.png and temperature_sensitivity_perhour_100d.png
make            # build the PDF
```

`make_sensitivity_figure.py` also writes a companion `*.npz` of the underlying
fields and the AD-vs-finite-difference validation (R^2 per panel); that data file
is regenerable and is not committed.

## Submission note

JOSS's own submission pipeline ingests a Markdown `paper.md` and builds the PDF
with its `inara`/pandoc container. This project deliberately keeps the paper in
LaTeX only; use `paper.tex` as the canonical source (and for an arXiv posting). A
`paper.md` can be produced from this content on request if a JOSS upload requires
it.
