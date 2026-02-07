"""SWAMPE (JAX rewrite)

This package mirrors the original SWAMPE module layout, but implements the
time-stepping core in JAX so it can be JIT-compiled and differentiated.

Precision
---------
JAX defaults to float32, which is typically the right choice for GPU throughput.
If you need double precision for numerical parity with the reference numpy
SWAMPE implementation, set:

    SWAMPE_JAX_ENABLE_X64=1

*before* importing `my_swamp`.
"""

from __future__ import annotations

import os as _os

from ._version import __version__

# Configure JAX precision at import time (before creating arrays / compiling).
# We respect any user-provided JAX configuration. If the environment variable
# SWAMPE_JAX_ENABLE_X64 is set, we apply it as an explicit override.
try:
    from jax import config as _config

    _env_x64 = _os.getenv("SWAMPE_JAX_ENABLE_X64")
    if _env_x64 is not None:
        _enable_x64 = _env_x64.strip().lower() in {"1", "true", "yes", "y", "on"}
        _config.update("jax_enable_x64", bool(_enable_x64))
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
    "plotting",
    "time_stepping",
    "model",
    "main_function",
    "run_model",
    "run_model_gpu",
    "main",
]


def __getattr__(name: str):
    """Lazy imports for heavy optional modules.

    `my_swamp.plotting` pulls in matplotlib/imageio, which is expensive and
    unnecessary for pure simulation workloads. Import on demand.
    """

    if name == "plotting":
        from . import plotting as _plotting

        # Cache on the module so subsequent accesses are cheap.
        globals()["plotting"] = _plotting
        return _plotting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
