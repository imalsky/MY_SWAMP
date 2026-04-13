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

from typing import Any, Union

try:
    import jax
    import jax.numpy as jnp

    def x64_enabled() -> bool:
        """Return True if JAX 64-bit mode is enabled."""
        return bool(jax.config.read("jax_enable_x64"))

    def float_dtype() -> Any:
        """Return ``jnp.float64`` when x64 is enabled, else ``jnp.float32``."""
        return jnp.float64 if x64_enabled() else jnp.float32

    def complex_dtype() -> Any:
        """Return ``jnp.complex128`` when x64 is enabled, else ``jnp.complex64``."""
        return jnp.complex128 if x64_enabled() else jnp.complex64

    #: A scalar that may be a Python float or a JAX array/tracer (for autodiff).
    Scalar = Union[float, jax.Array]

except Exception:  # pragma: no cover
    # Allow import in environments where JAX isn't installed (docs, packaging).
    import numpy as np

    def x64_enabled() -> bool:
        """Return True (fallback assumes float64 when JAX is unavailable)."""
        return True

    def float_dtype() -> Any:
        """Return ``np.float64`` (fallback when JAX is unavailable)."""
        return np.float64

    def complex_dtype() -> Any:
        """Return ``np.complex128`` (fallback when JAX is unavailable)."""
        return np.complex128

    Scalar = Union[float, Any]
