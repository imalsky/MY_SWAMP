"""my_swamp.autodiff_utils

Small utilities for forward-mode autodiff in JAX.

This module intentionally stays lightweight and does not depend on any
my_swamp-specific model internals.

Forward-mode is often the right choice for SWAMPE-style inference/optimization
when you only differentiate with respect to a handful of scalar parameters.

The helpers below are designed for the common pattern:

  - theta: a 1-D parameter vector
  - loss_fn(theta): scalar loss (shape ())
  - compute grad(theta) via forward-mode (JVPs)

"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp


def fwd_grad(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta: jnp.ndarray,
    *,
    chunk: Optional[int] = None,
) -> jnp.ndarray:
    """Forward-mode gradient of a scalar loss with respect to a 1-D parameter vector.

    Parameters
    ----------
    loss_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Callable that maps a one-dimensional parameter vector to a scalar JAX
        loss value.
    theta : jnp.ndarray
        One-dimensional parameter vector of shape ``(p,)``.
    chunk : Optional[int]
        Optional batch size for tangent directions. ``None`` uses
        ``jax.jacfwd`` on the full vector, while a positive integer uses
        chunked JVP batches to reduce peak memory.

    Returns
    -------
    jnp.ndarray
        Gradient vector with the same shape and dtype family as ``theta``.

    Notes
    -----
    - ``jax.jacfwd`` pushes all tangent directions at once, which is typically
      fine for ~5 parameters. For larger vectors, batching via JVP can be more
      memory friendly.
    - This helper is pure-JAX and can be wrapped in ``jax.jit`` if you evaluate
      it repeatedly.
    """

    theta = jnp.asarray(theta)
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D, got shape {theta.shape}.")

    p = int(theta.shape[0])
    if p == 0:
        return jnp.zeros((0,), dtype=theta.dtype)

    if chunk is None or int(chunk) >= p:
        g = jax.jacfwd(loss_fn)(theta)
        return jnp.asarray(g)

    chunk_i = int(chunk)
    if chunk_i <= 0:
        raise ValueError(f"chunk must be a positive int or None, got {chunk!r}.")

    eye = jnp.eye(p, dtype=theta.dtype)

    def one_dir(v: jnp.ndarray) -> jnp.ndarray:
        """Compute a single directional derivative ``d(loss)/dv`` via forward-mode JVP."""
        _, dl = jax.jvp(loss_fn, (theta,), (v,))
        return dl

    parts = []
    for i in range(0, p, chunk_i):
        vs = eye[i : i + chunk_i]
        parts.append(jax.vmap(one_dir)(vs))

    return jnp.concatenate(parts, axis=0)
