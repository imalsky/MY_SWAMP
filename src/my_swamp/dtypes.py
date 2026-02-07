"""my_swamp.dtypes

Centralized dtype choices for the SWAMPE JAX port.

Goals
-----
- Default to float32 for GPU throughput (JAX default).
- Allow opting into float64 for numerical parity with the reference NumPy SWAMPE
  by enabling JAX 64-bit mode (``jax_enable_x64=True``).

How to enable float64
---------------------
Recommended: set the environment variable before importing JAX / this package::

    export SWAMPE_JAX_ENABLE_X64=1

You may also set::

    from jax import config
    config.update("jax_enable_x64", True)

but note that, per JAX conventions, config flags should be set before any JAX
computations/compilations.

Design note
-----------
We do *not* freeze the dtype at import time. Instead, we query the current
``jax_enable_x64`` flag at the call site. This makes the behavior robust to
import ordering in interactive sessions.
"""

from __future__ import annotations

from typing import Any

try:
    import jax
    import jax.numpy as jnp

    def x64_enabled() -> bool:
        """Return True if JAX 64-bit mode is enabled."""
        return bool(jax.config.read("jax_enable_x64"))

    def float_dtype():
        """Return the package float dtype (float32 or float64)."""
        return jnp.float64 if x64_enabled() else jnp.float32

    def complex_dtype():
        """Return the package complex dtype (complex64 or complex128)."""
        return jnp.complex128 if x64_enabled() else jnp.complex64

    INT_DTYPE = jnp.int32

    def as_real(x: Any):
        """Convert to a JAX scalar/array with the package float dtype."""
        return jnp.asarray(x, dtype=float_dtype())

    def as_complex(x: Any):
        """Convert to a JAX scalar/array with the package complex dtype."""
        return jnp.asarray(x, dtype=complex_dtype())

except Exception:  # pragma: no cover
    # Allow import in environments where JAX isn't installed (docs, packaging).
    import numpy as np  # type: ignore

    def x64_enabled() -> bool:
        return True

    def float_dtype():
        return np.float64

    def complex_dtype():
        return np.complex128

    INT_DTYPE = np.int32

    def as_real(x: Any):
        return np.asarray(x, dtype=float_dtype())

    def as_complex(x: Any):
        return np.asarray(x, dtype=complex_dtype())
