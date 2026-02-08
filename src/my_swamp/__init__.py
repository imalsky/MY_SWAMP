"""SWAMPE (JAX port)

This package mirrors the original NumPy SWAMPE module layout, but implements
the numerical core in JAX so it can be JIT-compiled, GPU-accelerated, and
differentiated.

Numerical parity
----------------
The reference SWAMPE implementation uses NumPy/SciPy and therefore runs in
double precision by default. For closest numerical parity, this package enables
JAX 64-bit mode by default at import time.

You can override this behavior by setting the environment variable
``SWAMPE_JAX_ENABLE_X64`` before importing this package:

  - ``SWAMPE_JAX_ENABLE_X64=1``  -> enable float64/complex128 (default)
  - ``SWAMPE_JAX_ENABLE_X64=0``  -> disable and use float32/complex64
"""

from __future__ import annotations

import os as _os

from ._version import __version__


# -----------------------------------------------------------------------------
# JAX precision configuration
# -----------------------------------------------------------------------------
# JAX config flags must be set before creating arrays / compiling.
#
# To match NumPy SWAMPE as closely as possible, we default to enabling
# float64/complex128 unless the user explicitly opts out.
try:
    from jax import config as _config

    _env_x64 = _os.getenv("SWAMPE_JAX_ENABLE_X64")
    if _env_x64 is None:
        _config.update("jax_enable_x64", True)
    else:
        _enable_x64 = _env_x64.strip().lower() in {"1", "true", "yes", "y", "on"}
        _config.update("jax_enable_x64", bool(_enable_x64))
except Exception:
    # Allow importing in environments where JAX isn't available (docs/packaging).
    pass


# -----------------------------------------------------------------------------
# Public submodules
# -----------------------------------------------------------------------------
from . import continuation
from . import spectral_transform
from . import initial_conditions
from . import filters
from . import forcing
from . import explicit_tdiff
from . import modEuler_tdiff
from . import time_stepping
from . import model
from . import main_function

from .model import run_model, run_model_gpu
from .main_function import main


__all__ = [
    "continuation",
    "spectral_transform",
    "initial_conditions",
    "filters",
    "forcing",
    "explicit_tdiff",
    "modEuler_tdiff",
    "time_stepping",
    "model",
    "main_function",
    "run_model",
    "run_model_gpu",
    "main",
]


def __getattr__(name: str):
    """Lazy import for heavy optional plotting dependencies."""

    if name == "plotting":
        from . import plotting as _plotting

        globals()["plotting"] = _plotting
        return _plotting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
