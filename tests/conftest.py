from __future__ import annotations

import os


# Ensure tests run on CPU in CI-like environments even if accelerators are present.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Default to 32-bit mode for CI speed; users can override locally.
# This must be set before importing `my_swamp` (it configures jax_enable_x64 at import time).
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "0")

# Avoid aggressive preallocation in constrained CI runners.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
