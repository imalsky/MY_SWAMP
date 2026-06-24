# JOSS submission notes for `SWAMPE-JAX`

Working document (not part of the published paper). It captures (1) what JOSS
requires, (2) how `SWAMPE-JAX` currently measures up, (3) the internal peer-review
synthesis used to revise the paper, and (4) the actions only the author can
complete before clicking submit.

**Format note.** The paper is provided as `paper.tex`, which reproduces the JOSS
PDF layout exactly (logo, left metadata sidebar, CC-BY footer) and builds locally
with `make`. JOSS's own submission system, however, ingests a Markdown `paper.md`
and generates the PDF itself with its `inara`/pandoc pipeline — so the file you
actually upload to JOSS is `paper.md`, which can be produced from this content on
request. Use `paper.tex` as a faithful local preview (or for an arXiv posting).

---

## 1. What JOSS is (and is not)

JOSS publishes a **short** paper (≈250–1000 words) *about* research software. The
paper is deliberately not a full methods paper — the software is the object of
review, the paper is a pointer to it. Submission is at
<https://joss.theoj.org/papers/new>; the editor opens a "pre-review" issue, a
reviewer is assigned, and review happens openly on GitHub against a checklist.

Key references:
- Submission process: <https://joss.readthedocs.io/en/latest/submitting.html>
- Review criteria: <https://joss.readthedocs.io/en/latest/review_criteria.html>

---

## 2. Requirements checklist (software + paper)

### Software

| Requirement | Status for `MY_SWAMP` | Notes |
|---|---|---|
| Open-source, OSI-approved license, in repo as a file | ✅ | `LICENCE.txt`, BSD-3-Clause; `license = "BSD-3-Clause"` in `pyproject.toml`. |
| Public VCS repo, browsable, issues open without registration | ⚠️ confirm | `pyproject.toml` points to `github.com/imalsky/MY_SWAMP`. Confirm it is public and the issue tracker is on. |
| ≥ ~3 months of public version history (not a fresh dump) | ⚠️ confirm | The repo has a real commit/release history (`v0.1.x`); confirm public history depth before submitting. |
| Automated install per language norms | ✅ | `pip install -e .`; deps pinned in `pyproject.toml`. |
| Functionality a reviewer can run | ✅ | `my-swamp` CLI entry point + Python API. |
| Automated tests | ✅ | 36 pytest tests (smoke/parity markers), CPU x64, ~21 s. |
| Community guidelines (contribute / report / support) | ✅ / ⚠️ | `CONTRIBUTING.md` exists; make sure it covers issue-reporting + support, not just style. |
| API documentation | ✅ | Docstrings throughout + a comprehensive `readme.md`. |
| "Substantial scholarly effort" (not a thin wrapper) | ✅ (see §3) | The differentiable JAX rewrite is the scholarly contribution over the already-published SWAMPE; this is the one thing a JOSS editor will scrutinize most. |

### Paper (`paper.tex`, JOSS format)

| Requirement | Status | Notes |
|---|---|---|
| Title, authors, ORCIDs, affiliations | ✅ | Three authors with ORCIDs: Malsky (JPL/Caltech), Vissapragada (Carnegie Observatories), Landgren (CIRES, CU Boulder). Confirm affiliations + co-author consent (§4). DOI / editor / reviewer / date fields are placeholders the JOSS bot fills at review time. |
| Summary intelligible to non-specialists | ✅ | |
| Statement of Need | ✅ | |
| State of the field / comparison to related software | ✅ | SWAMPE, Dedalus, NeuralGCM, ExoJAX, jax-cfd. |
| References with DOIs where available | ✅ | 23 entries; all cited keys resolve, no orphans. `jax:2018` is a software `@misc` (URL, no DOI — standard); `Langton:2008` is a thesis. |
| ≤ ~1000 words | ✅ | ~950 words of body prose. |
| A figure (optional but recommended) | ✅ | `parity_comparison.png` (SWAMPE vs MY_SWAMP parity). |

---

## 3. Internal peer-review synthesis (ARS reviewer panel)

`paper.md` was run through a simulated JOSS review panel (Editor-in-Chief +
software/AD reviewer + domain reviewer). Consolidated decision: **Minor
revisions**, conditional on the author-only items in §4. The substantive issues
the panel raised have been addressed in the current draft:

**Resolved in this revision:**
1. *Figure/claim mismatch (most serious).* The draft originally captioned the
   figure as a "forced hot-Jupiter, hundred-day" run; the committed figure is in
   fact a **10-day Williamson Test-2** comparison. The caption and the parity
   bullet were rewritten to describe exactly what is shown, and the two parity
   statements (short-horizon absolute-tolerance pytest suite vs. the long-run
   relative-$L_2$ comparison) were separated.
2. *Overstated forward-mode AD.* "forward-mode … checked against finite
   differences" was corrected — only reverse-mode is FD-validated; forward-mode
   is supported and smoke-tested.
3. *"machine precision"* softened to "high precision" (Φ/η/δ agree to a relative
   ~$10^{-12}$–$10^{-13}$, not literal machine epsilon).
4. *"weeks of CPU time per simulation"* recalibrated to "hundreds to thousands of
   core-hours … months to explore a grid," matching the cited 3-D GCM sources.
5. *Williamson "test cases"* qualified to **cases 1 and 2** (of the 7).
6. *Citation hygiene:* one-to-one mapping of day–night / superrotation /
   variability citations; "variability" softened (Liu & Showman 2013); `Gelb:2001`
   now cited at the hyperdiffusion mention (was an orphan).
7. *State of the field strengthened:* added **ExoJAX** (differentiable
   radiative transfer — the complementary exoplanet precedent) and **NeuralGCM**
   (differentiable Earth GCM); positioned `MY_SWAMP` between them.
8. *Concrete use case:* added a phase-curve / eclipse-map inference sentence with
   the eclipse-mapping lineage (Knutson 2007; Rauscher 2018).

**Deferred to the author (judgment / data calls):** see §4.

---

## 4. Author-only actions before submitting

1. **(Done — confirm affiliations.)** Three authors with ORCIDs are set: Isaac
   Malsky (JPL/Caltech, 0000-0003-0217-3880), Shreyas Vissapragada (Carnegie
   Observatories, 0000-0003-4015-9975), and Ekaterina Landgren (CIRES, CU Boulder,
   0000-0001-6029-5216). **Verify each affiliation is current** — Landgren's was
   inferred from a 2025 search (CIRES/Boulder; the 2022 SWAMPE paper had Cornell),
   and Vissapragada's is from your VULCAN-JAX paper.
2. **Authorship scope / consent.** The paper now lists Malsky, Vissapragada, and
   Landgren (the original SWAMPE lead author). Alice Nadeau, the other original
   SWAMPE author, is thanked in the Acknowledgements rather than listed — confirm
   that is intended. JOSS requires every listed co-author to have agreed to be
   included.
3. **(Decided: SWAMPE-JAX.)** The paper uses the display name `SWAMPE-JAX`
   (installed and imported as `my_swamp`), matching the README branding. For full
   consistency you may optionally rename the PyPI distribution from `my-swamp` to
   `swampe-jax`, but it is not required — a display name that differs from the
   import name is common (e.g., scikit-learn / sklearn).
4. **(Recommended) Upgrade the figure / add a forced-regime result.** A genuine
   forced 100-day hot-Jupiter parity run (`python testing/compare_long_run_parity.py
   --days 100 --test 0`) would let the figure show the science regime
   (Perez-Becker 2013) rather than the Test-2 validation case. Regenerate
   `parity_comparison.png` from its `field_comparison.png` and tighten the caption
   if you do.
5. **(Done — extend if desired.)** The editor's strongest suggestion — a single
   number demonstrating the new capability — has been added to the paper: at
   $M=42$, a reverse-mode gradient with respect to four parameters costs ~2.3x a
   forward integration (measured forward 0.70 s, gradient 1.59 s on CPU). You may
   optionally add a `vmap` ensemble-throughput or GPU number from
   `testing/benchmark_scan.py`.
6. **(Done.)** An AI-usage disclosure is included in the Acknowledgements ("The
   author used Claude (Anthropic) to assist with code development, code cleanup,
   and manuscript preparation"), matching JOSS's current review criteria. Adjust
   the wording if you want it more specific.
7. **Tag a release + archive for a DOI.** After review, make a tagged release and
   deposit it (Zenodo/figshare) to mint a DOI; report version + DOI on the review
   issue.

---

## 5. Build

The JOSS-format PDF builds locally from `paper.tex` (no pandoc needed):

```bash
cd paper && make          # pdflatex -> bibtex -> pdflatex x2  ->  paper.pdf
```

Requires a LaTeX install with `pdflatex` + `bibtex`. The build uses `natbib` +
`plainnat` and the bundled `joss-logo.png`. To generate the *official* JOSS PDF
at submission time, JOSS runs its `openjournals/inara` Docker image on a Markdown
`paper.md`; this `paper.tex` reproduces that layout for local preview.
