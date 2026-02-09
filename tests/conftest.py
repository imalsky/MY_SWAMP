from __future__ import annotations

import os


# Ensure tests run on CPU in CI-like environments even if accelerators are present.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# -----------------------------------------------------------------------------
# Precision control
# -----------------------------------------------------------------------------
# The reference SWAMPE implementation runs in float64 by default. For numerical
# parity, we default the test suite to 64-bit mode unless the user explicitly
# opts out.
#
# Users may select precision by exporting either variable before running pytest:
#   - SWAMPE_JAX_ENABLE_X64=0/1 (package-specific convenience)
#   - JAX_ENABLE_X64=0/1        (canonical JAX environment variable)
#
# We mirror the chosen value into the other variable so that:
#   (a) JAX reads the desired mode at import time
#   (b) my_swamp's import-time config logic does not override the user's choice
if "SWAMPE_JAX_ENABLE_X64" in os.environ and "JAX_ENABLE_X64" not in os.environ:
    os.environ["JAX_ENABLE_X64"] = os.environ["SWAMPE_JAX_ENABLE_X64"]
elif "JAX_ENABLE_X64" in os.environ and "SWAMPE_JAX_ENABLE_X64" not in os.environ:
    os.environ["SWAMPE_JAX_ENABLE_X64"] = os.environ["JAX_ENABLE_X64"]
else:
    os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_ENABLE_X64", "1")


# Avoid aggressive preallocation in constrained CI runners.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
