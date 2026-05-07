# ruff: noqa: E741
"""Autodifferentiation smoke tests for the JAX core.

These tests verify that the differentiable scan core produces finite gradients
with the expected shape/dtype across the scalar parameters that appear in the
inner loop, and finite-difference cross-checks one parameter to catch silent AD
breakage (e.g., an accidental `float(tracer)` coercion in a contributor's PR).

The tests use a deliberately tiny configuration (M=42, very small tmax) so the
suite stays fast on CPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# Small, fast configuration. M=42 is the smallest supported truncation; we keep
# tmax small (5 timesteps after the 2-level init) so the scan compiles fast.
_BASE_KWARGS = dict(
    M=42,
    dt=600.0,
    tmax=7,
    Phibar=3.0e5,
    omega=7.292e-5,
    a=6.37122e6,
    test=None,
    g=9.8,
    forcflag=True,
    taurad=86400.0,
    taudrag=86400.0,
    DPhieq=4.0e6,
    diffflag=True,
    modalflag=True,
    alpha=0.01,
    expflag=False,
    K6=1.24e33,
    diagnostics=False,
    jit_scan=True,
)


def _x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def _make_loss(param_name: str):
    """Build a scalar loss(theta) that runs the model with theta substituted into ``param_name``."""
    from my_swamp.model import run_model_scan_final

    def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        kwargs = dict(_BASE_KWARGS)
        kwargs[param_name] = theta
        sim = run_model_scan_final(**kwargs)
        Phi_final = sim["last_state"].Phi_curr
        # Mean-squared geopotential: smooth, scalar, depends on every step.
        return jnp.mean(Phi_final * Phi_final)

    return loss_fn


@pytest.mark.smoke
def test_grad_returns_finite_for_each_scalar_parameter():
    """jax.grad over each documented differentiable scalar returns a finite number."""
    if not _x64_enabled():
        pytest.skip("Autodiff parity tests are gated on x64 mode.")

    test_cases = [
        ("DPhieq", 4.0e6),
        ("taurad", 86400.0),
        ("taudrag", 86400.0),
        ("K6", 1.24e33),
        ("Phibar", 3.0e5),
        ("omega", 7.292e-5),
        ("a", 6.37122e6),
        ("dt", 600.0),
        ("alpha", 0.01),
    ]
    for name, value in test_cases:
        loss = _make_loss(name)
        g = jax.grad(loss)(jnp.asarray(value))
        g_np = np.asarray(g)
        assert g_np.shape == (), f"grad wrt {name} should be scalar, got shape {g_np.shape}"
        assert np.isfinite(g_np), f"grad wrt {name} is not finite: {g_np}"


@pytest.mark.smoke
def test_jvp_returns_finite_for_DPhieq():
    """jax.jvp returns finite primal+tangent for at least one scalar parameter."""
    if not _x64_enabled():
        pytest.skip("Autodiff parity tests are gated on x64 mode.")

    loss = _make_loss("DPhieq")
    primal = jnp.asarray(4.0e6)
    primal_out, tangent_out = jax.jvp(loss, (primal,), (jnp.asarray(1.0),))
    assert np.isfinite(np.asarray(primal_out))
    assert np.isfinite(np.asarray(tangent_out))


@pytest.mark.parity
def test_grad_matches_finite_difference_for_DPhieq():
    """Cross-check jax.grad against a centered finite difference for one parameter.

    This is the test that catches the most insidious AD bugs: any new
    ``float(tracer)`` coercion or NumPy-on-tracer call inside the scan body
    would silently zero out the gradient with respect to ``DPhieq``.
    """
    if not _x64_enabled():
        pytest.skip("FD cross-check requires x64 for stable finite-difference accuracy.")

    loss = _make_loss("DPhieq")
    theta0 = 4.0e6
    eps = 1.0  # absolute step; DPhieq is large so 1.0 is well within the linear regime.

    g_ad = float(jax.grad(loss)(jnp.asarray(theta0)))
    f_plus = float(loss(jnp.asarray(theta0 + eps)))
    f_minus = float(loss(jnp.asarray(theta0 - eps)))
    g_fd = (f_plus - f_minus) / (2.0 * eps)

    # The model is smooth in DPhieq; a centered FD with this step size should
    # match jax.grad to a few parts in 1e6 in float64.
    assert g_ad != 0.0, "AD gradient is zero — possible silent breakage of differentiability."
    rel_err = abs(g_ad - g_fd) / max(abs(g_fd), 1.0e-30)
    assert rel_err < 1.0e-4, (
        f"AD gradient {g_ad} disagrees with FD gradient {g_fd} by relative error {rel_err}; "
        "investigate for accidental tracer-to-Python coercions in the scan body."
    )


@pytest.mark.smoke
def test_grad_wrt_initial_phi_field():
    """jax.grad with respect to a full (J, I) initial geopotential field."""
    if not _x64_enabled():
        pytest.skip("Autodiff parity tests are gated on x64 mode.")

    from my_swamp.initial_conditions import (
        spectral_params,
        state_var_init,
        test1_init,
        velocity_init,
    )
    from my_swamp.model import run_model_scan_final

    M = 42
    N, I, J, dt_default, lambdas, mus, _w = spectral_params(M)
    a = 6.37122e6
    omega = 7.292e-5
    Phibar = 3.0e5
    a1 = 0.05

    SU0, sina, cosa, etaamp, _Phiamp = test1_init(a, omega, a1)
    eta0, _, delta0, _, Phi0, _ = state_var_init(I, J, mus, lambdas, test=None, etaamp=etaamp)
    U0, V0 = velocity_init(I, J, SU0, cosa, sina, mus, lambdas, test=None)

    def loss_fn(Phi0_init: jnp.ndarray) -> jnp.ndarray:
        sim = run_model_scan_final(
            M=M,
            dt=dt_default,
            tmax=5,
            Phibar=Phibar,
            omega=omega,
            a=a,
            test=None,
            forcflag=True,
            diffflag=True,
            modalflag=True,
            expflag=False,
            eta0_init=eta0,
            delta0_init=delta0,
            Phi0_init=Phi0_init,
            U0_init=U0,
            V0_init=V0,
            diagnostics=False,
            jit_scan=True,
        )
        return jnp.mean(sim["last_state"].Phi_curr)

    g = jax.grad(loss_fn)(Phi0)
    g_np = np.asarray(g)
    assert g_np.shape == (J, I), f"expected gradient shape {(J, I)}, got {g_np.shape}"
    assert np.all(np.isfinite(g_np)), "gradient field contains NaN/Inf"
    # Sanity: not identically zero.
    assert float(np.max(np.abs(g_np))) > 0.0, "gradient field is identically zero"
