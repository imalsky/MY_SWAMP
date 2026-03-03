"""Utilities for specializing JAX branches when predicates are static."""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

import jax
import jax.numpy as jnp
import numpy as np


T = TypeVar("T")


def static_bool(pred: Any) -> Optional[bool]:
    """Return a Python bool when `pred` is statically known, else None."""
    if isinstance(pred, (bool, np.bool_, int, np.integer)):
        return bool(pred)

    if isinstance(pred, jax.core.Tracer):
        return None

    # Non-traced 0-D JAX arrays can be materialized safely.
    if isinstance(pred, jax.Array) and pred.shape == ():
        return bool(np.asarray(pred))

    return None


def cond(pred: Any, true_fun: Callable[[Any], T], false_fun: Callable[[Any], T], operand: Any) -> T:
    """Like `jax.lax.cond`, but uses a Python branch when possible."""
    b = static_bool(pred)
    if b is not None:
        return true_fun(operand) if b else false_fun(operand)
    return jax.lax.cond(jnp.asarray(pred), true_fun, false_fun, operand)


def select(pred: Any, on_true: T, on_false: T) -> T:
    """Like `jax.lax.select`, but uses a Python branch when possible."""
    b = static_bool(pred)
    if b is not None:
        return on_true if b else on_false
    return jax.lax.select(jnp.asarray(pred), on_true, on_false)


def maybe_apply(pred: Any, fn: Callable[[T], T], value: T) -> T:
    """Apply fn(value) if pred is true; preserve JAX-traceability if dynamic."""
    b = static_bool(pred)
    if b is True:
        return fn(value)
    if b is False:
        return value
    return jax.lax.cond(jnp.asarray(pred), fn, lambda x: x, value)
