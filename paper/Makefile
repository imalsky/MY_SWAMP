# Build the SWAMPE-JAX paper in the Journal of Open Source Software (JOSS) format.
paper.pdf: paper.tex paper.bib parity_comparison.png joss-logo.png
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out \
	      paper.fdb_latexmk paper.fls paper.synctex.gz paper.xdv

.PHONY: clean
