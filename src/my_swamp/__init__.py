from __future__ import annotations

from ._version import __version__
"""SWAMPE (JAX rewrite)

This package mirrors the original SWAMPE module layout, but implements the
time-stepping core in JAX so it can be JIT-compiled and differentiated.

Precision
---------
The reference SWAMPE implementation is double precision. For numerical parity,
this package enables JAX 64-bit mode by default. To disable, set:

    SWAMPE_JAX_ENABLE_X64=0

before importing SWAMPE.
"""
import os as _os

# Enable 64-bit for numerical parity with the reference numpy implementation.
# Must run at import-time (before creating arrays / compiling).
try:
    from jax import config as _config

    if _os.getenv("SWAMPE_JAX_ENABLE_X64", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
        _config.update("jax_enable_x64", True)
except Exception:
    # Allow static inspection/packaging environments where JAX isn't importable.
    pass

from . import continuation
from . import spectral_transform
from . import initial_conditions
from . import filters
from . import forcing
from . import explicit_tdiff
from . import modEuler_tdiff
from . import plotting
from . import time_stepping
from . import model
from . import main_function

from .model import run_model
from .main_function import main

__all__ = [
    "continuation",
    "spectral_transform",
    "initial_conditions",
    "filters",
    "forcing",
    "explicit_tdiff",
    "modEuler_tdiff",
    "plotting",
    "time_stepping",
    "model",
    "main_function",
    "run_model",
    "main",
]
