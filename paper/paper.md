---
title: 'MY_SWAMP: A differentiable JAX implementation of the SWAMPE spectral shallow-water model'
tags:
  - Python
  - JAX
  - geophysical fluid dynamics
  - shallow-water equations
  - spectral methods
authors:
  - name: 'TODO: First Author'
    orcid: 'TODO: 0000-0000-0000-0000'
    affiliation: 1
  - name: 'TODO: Coauthor'
    orcid: 'TODO: 0000-0000-0000-0000'
    affiliation: 1
affiliations:
  - name: 'TODO: Institution, Department, City, Country'
    index: 1
date: '2026-02-08'
bibliography: paper.bib
---

# Summary

MY_SWAMP is a Python package implementing a global spectral shallow-water model on the sphere, with the numerical core written in JAX. The package is designed for researchers who want an idealized, inspectable dynamical system that is fast enough for repeated experimentation and can also be differentiated end-to-end. The model advances vorticity, divergence, and a geopotential-like height field using spherical spectral transforms (FFT in longitude and associated Legendre transforms in latitude), and supports the standard “test case” style configurations used in shallow-water modeling.

MY_SWAMP was written to recreate the behavior of a reference implementation (“SWAMPE”) while providing modern capabilities enabled by JAX, including accelerator execution and composable program transformations (e.g., automatic differentiation, JIT compilation, and vectorization) [@jax2018github]. In addition to forward simulation, the repository includes verification tests for spectral-transform conventions and notebooks that compare MY_SWAMP results against the NumPy reference implementation and demonstrate a small differentiable inverse problem built around the model.

# Statement of need

Spectral shallow-water models are widely used as idealized testbeds for atmospheric and oceanic dynamics because they capture key rotating-fluid phenomena (e.g., global wave propagation and balanced flow) while remaining computationally tractable. They are frequently used to validate numerical methods, to build intuition via controlled experiments, and to prototype inference and control workflows before moving to more complex general circulation models.

While many existing shallow-water solvers are suitable for forward simulation, it is often difficult to obtain reliable gradients of simulation outputs with respect to physical parameters (e.g., forcing strength, damping timescales, hyperdiffusion) or initial conditions. These gradients are central to modern workflows such as variational data assimilation, gradient-based calibration, sensitivity analysis, and some forms of uncertainty quantification. MY_SWAMP addresses this gap by providing a shallow-water solver in JAX, so that the full time integration can be embedded into differentiable pipelines without rewriting the numerical core.

The target audience includes researchers and students who (i) need a lightweight and reproducible shallow-water spectral model for numerical experiments, (ii) want to reproduce or extend prior results obtained with SWAMPE, and/or (iii) want to use a physically motivated dynamical system inside optimization or inference code that benefits from JAX transformations.

# State of the field

A wide range of shallow-water solvers exist, spanning finite-difference, finite-volume, and spectral formulations, and ranging from small educational codes to components of comprehensive atmospheric models. General-purpose PDE frameworks can also be used to implement shallow-water equations, but may require additional effort to match specific spectral conventions, time-stepping details, or legacy experiment setups.

MY_SWAMP is intentionally narrow in scope: it is not a general atmospheric model, and it is not a generic PDE framework. Instead, it focuses on (a) reproducing a particular spectral shallow-water formulation used by SWAMPE, and (b) enabling differentiable simulation and accelerator execution through JAX. This “parity + differentiability” focus is the primary justification for a new implementation rather than extending an existing forward-only SWAMPE codebase.

TODO (required for final submission): add a concrete comparison to at least one or two commonly used shallow-water model implementations (or frameworks) in your research community, with citations, and explain why those alternatives do not meet the combined needs of SWAMPE parity and end-to-end differentiability.

# Software design

MY_SWAMP follows a “static precompute + pure scan loop” architecture. Resolution-dependent spectral machinery (Gaussian quadrature nodes/weights, associated Legendre basis, and time-stepping coefficients) is constructed once and stored in a static structure. The forward simulation itself is implemented as a side-effect-free function that advances the state using `jax.lax.scan`, returning time histories as arrays. Plotting, file I/O, and continuation logic are kept outside the differentiable core so they do not interfere with JAX transformations or introduce hidden side effects.

The codebase preserves the original SWAMPE module layout to ease comparison and adoption, but replaces time-critical numerical kernels with JAX equivalents. Key implementation choices include: (i) explicit separation between “build” steps (basis and coefficient construction) and “run” steps (time integration), (ii) careful handling of data types to support both 32-bit and 64-bit modes, with 64-bit enabled by default for closer numerical parity, and (iii) an API that exposes a differentiable core while retaining a compatibility wrapper that mirrors SWAMPE’s historical call signatures.

Verification is treated as a first-class requirement because spectral models are sensitive to transform conventions and normalization choices. The automated test suite includes numerical identity checks for the Legendre basis construction and round-trip consistency checks for the spectral transform stack, as well as smoke tests that run a minimal end-to-end integration and assert basic invariants (finite fields, expected array shapes, and sensible diagnostics). These tests are designed to catch regressions that would invalidate parity claims or silently change the model’s numerical interpretation.

# Research impact statement

MY_SWAMP enables workflows that are difficult to implement with many forward-only shallow-water codes. The repository includes reproducible notebooks that (a) compare MY_SWAMP output fields against the NumPy SWAMPE reference implementation for matched parameters, and (b) demonstrate a small differentiable inverse problem in which physical timescales are inferred from a synthetic final-state snapshot. Together, these materials provide concrete evidence of near-term research significance: they show both parity validation and the use of the model as a differentiable component inside an inference loop.

TODO (required for final submission): if available, add citations to published or in-review manuscripts that use MY_SWAMP, or other evidence of adoption (e.g., external users, issues/PRs from other groups). If the notebooks serve as the primary evidence, ensure they are clearly documented and reproducible, and consider adding a short benchmark or runtime note to contextualize performance on CPU/GPU.

# AI usage disclosure

Portions of the repository were drafted with the assistance of generative AI tools. Specifically, AI assistance was used for scaffolding parts of the test suite, CI configuration, and initial paper drafting. All AI-assisted outputs were reviewed and edited by the human authors, and correctness was verified by running the automated tests and by cross-checking numerical behavior against the SWAMPE reference implementation.

TODO: update this section to accurately reflect the tools/models used (including versions), where they were used (code, documentation, paper text), and the verification steps performed by the authors.

# Acknowledgements

TODO: acknowledge funding sources and contributors who are not listed as paper authors.

# References
