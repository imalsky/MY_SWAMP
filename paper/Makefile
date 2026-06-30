# Build the SWAMPE-JAX paper in the Journal of Open Source Software (JOSS) format.
# The paper is LaTeX-only; paper.tex is the canonical source (there is no paper.md).
paper.pdf: paper.tex paper.bib parity_comparison.png temperature_sensitivity_perhour_100d.png joss-logo.png
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

# Regenerate the figures (slow; figure PNGs are not committed to git).
# Fig. 1 runs reference NumPy SWAMPE + the JAX model; Fig. 2 runs a 100-day
# forward-mode-differentiated integration.
figures:
	python ../tests/compare_long_run_parity.py --days 100
	cp ../figures/long_run_parity_outputs/forced_default_100d/field_comparison.png parity_comparison.png
	python ../scripts/make_sensitivity_figure.py --days 100

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out \
	      paper.fdb_latexmk paper.fls paper.synctex.gz paper.xdv

.PHONY: clean figures
