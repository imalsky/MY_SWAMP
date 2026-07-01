# Build the SWAMPE-JAX paper in the Journal of Open Source Software (JOSS) format.
# The paper is LaTeX-only; paper.tex is the canonical source (there is no paper.md).
paper.pdf: paper.tex paper.bib parity_comparison.png temperature_sensitivity_perhour_100d.png joss-logo.png
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

# Regenerate the figures (slow; figure PNGs ARE committed to git -- Fig. 1
# needs the external reference SWAMPE to regenerate, so the rendered PNGs are
# checked in even though the data behind them is not; see .gitignore).
# Fig. 1 runs reference NumPy SWAMPE + the JAX model; Fig. 2 runs a 100-day
# forward-mode-differentiated integration. All generator scripts live in
# paper/scripts/ -- this directory is self-contained.
figures:
	python scripts/compare_long_run_parity.py --days 100
	cp figures/long_run_parity_outputs/forced_default_100d/field_comparison.png parity_comparison.png
	python scripts/make_sensitivity_figure.py --days 100
	cp figures/long_run_parity_outputs/forced_default_100d/summary.json benchmark_data/cpu_parity_100day_summary.json

# Regenerate the CPU half of the "Speed (CPU and GPU)" section's data (slow,
# ~15 min; dominated by the reference NumPy SWAMPE run). Separate --out-dir
# from the 100-day figures target above so a 10-day run can never silently
# overwrite the 100-day parity output. The GPU half is not reproducible from
# this Makefile -- see paper/benchmark_data/README.md (Colab notebook or
# scripts/swampe_gpu_vmap_test.py on a local GPU).
speed:
	python scripts/compare_long_run_parity.py --days 10 --out-dir figures/long_run_parity_outputs/forced_default_10d
	cp figures/long_run_parity_outputs/forced_default_10d/summary.json benchmark_data/cpu_speed_10day_summary.json

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out \
	      paper.fdb_latexmk paper.fls paper.synctex.gz paper.xdv

.PHONY: clean figures speed
