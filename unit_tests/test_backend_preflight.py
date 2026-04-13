from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_backend_preflight_cpu() -> None:
    """Verify that CPU backend preflight reports visible devices."""
    from my_swamp.backend_preflight import preflight_backend

    info = preflight_backend("cpu")
    assert info.device_count >= 1
    assert "cpu" in info.available_backends


@pytest.mark.smoke
def test_backend_preflight_invalid_backend() -> None:
    """Verify that unsupported backend requests raise a runtime error."""
    from my_swamp.backend_preflight import preflight_backend

    with pytest.raises(RuntimeError):
        preflight_backend("not-a-backend")
