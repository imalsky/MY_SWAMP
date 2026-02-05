# SWAMPE-JAX

This is a JAX reimplementation of the SWAMPE 2D shallow-water model.

Goals:
- Preserve the original SWAMPE module layout and call signatures.
- Move the numerical time-stepping core to JAX so it can be JIT-compiled and differentiated.

Notes:
- Plotting/saving are handled in a compatibility wrapper (`SWAMPE.model.run_model`) and are not
  part of the differentiable core (`SWAMPE.model.run_model_scan`).
- The associated Legendre basis can be built using SciPy (for bitwise parity with the original)
  or a pure-Python recurrence. If SciPy is unavailable, the code falls back automatically.

