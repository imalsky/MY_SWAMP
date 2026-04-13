"""Backend preflight helpers for tests and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax


_VALID_BACKENDS = ("cpu", "gpu", "tpu")


@dataclass(frozen=True)
class BackendInfo:
    """Summary of the JAX backend state observed during preflight checks."""

    requested_backend: Optional[str]
    default_backend: str
    available_backends: Tuple[str, ...]
    device_names: Tuple[str, ...]
    device_platforms: Tuple[str, ...]

    @property
    def device_count(self) -> int:
        """Return the number of visible devices in the selected backend view."""
        return len(self.device_names)


def _available_backends() -> Tuple[str, ...]:
    """Return the subset of known JAX backends that currently expose devices."""

    out: list[str] = []
    for backend in _VALID_BACKENDS:
        try:
            if len(jax.devices(backend)) > 0:
                out.append(backend)
        except RuntimeError:
            continue
    return tuple(out)


def preflight_backend(requested_backend: Optional[str] = None, *, require_gpu: bool = False) -> BackendInfo:
    """Validate backend/device availability and return a summary.
    
    Raises
    ------
    RuntimeError
        If a requested backend is unavailable, no devices are visible, or
        ``require_gpu=True`` without a visible GPU backend.
    
    Parameters
    ----------
    requested_backend : Optional[str]
        Explicit backend to validate. ``None`` checks the default JAX device
        view.
    require_gpu : bool
        If ``True``, raise an error unless a GPU backend is visible.
    
    Returns
    -------
    BackendInfo
        Summary of the requested/default backend, visible devices, and device
        platforms.
    """
    requested = None if requested_backend is None else str(requested_backend).strip().lower()
    if requested in {"", "none"}:
        requested = None

    if requested is not None and requested not in _VALID_BACKENDS:
        raise RuntimeError(f"Unsupported backend {requested!r}. Expected one of {_VALID_BACKENDS}.")

    available = _available_backends()
    if requested is not None and requested not in available:
        raise RuntimeError(f"Requested backend {requested!r} is unavailable. Available backends: {available}.")

    if require_gpu and "gpu" not in available:
        raise RuntimeError(
            f"GPU backend requested but unavailable. Available backends: {available}. "
            "Check your JAX GPU installation and visible devices."
        )

    try:
        devices = jax.devices(requested) if requested is not None else jax.devices()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to initialize backend {requested!r}. "
            "Check JAX_PLATFORMS/JAX_PLATFORM_NAME and backend driver installation."
        ) from exc

    if len(devices) == 0:
        raise RuntimeError(
            f"No JAX devices visible for backend {requested!r}. "
            "Check runtime configuration and accelerator visibility."
        )

    return BackendInfo(
        requested_backend=requested,
        default_backend=str(jax.default_backend()),
        available_backends=available,
        device_names=tuple(str(d) for d in devices),
        device_platforms=tuple(str(d.platform) for d in devices),
    )


def backend_info_lines(info: BackendInfo) -> Tuple[str, ...]:
    """Format backend summary for logs/CLI output.
    
    Parameters
    ----------
    info : BackendInfo
        Backend summary returned by :func:`preflight_backend`.
    
    Returns
    -------
    Tuple[str, ...]
        Log-friendly key/value lines describing the resolved backend state.
    """
    return (
        f"requested_backend={info.requested_backend}",
        f"default_backend={info.default_backend}",
        f"available_backends={','.join(info.available_backends)}",
        f"device_count={info.device_count}",
        f"device_platforms={','.join(info.device_platforms)}",
        f"devices={'; '.join(info.device_names)}",
    )
