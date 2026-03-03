from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_backend_preflight_cpu() -> None:
    from my_swamp.backend_preflight import preflight_backend

    info = preflight_backend("cpu")
    assert info.device_count >= 1
    assert "cpu" in info.available_backends


@pytest.mark.smoke
def test_backend_preflight_invalid_backend() -> None:
    from my_swamp.backend_preflight import preflight_backend

    with pytest.raises(RuntimeError):
        preflight_backend("not-a-backend")
