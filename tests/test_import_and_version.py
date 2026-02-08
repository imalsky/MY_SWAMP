from __future__ import annotations

import re


def test_import_and_version() -> None:
    # JAX is a required runtime dependency for the numerical core.
    import jax  # noqa: F401  # pylint: disable=unused-import
    import my_swamp  # noqa: F401  # pylint: disable=unused-import

    assert hasattr(my_swamp, "__version__")
    assert re.match(r"^\d+\.\d+\.\d+$", my_swamp.__version__), "Version should look like x.y.z"
