# ruff: noqa: E741
"""vmap-based ensemble smoke test.

Runs `run_model_scan_final` over a stack of scalar parameter values via
`jax.vmap` and verifies each ensemble member is finite, distinct, and matches
the corresponding direct (non-vmapped) call. This locks in the property that
ensemble forward simulation works without any code changes — it's a smoke test
to catch a future regression that would block ensemble use.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


_BASE_KWARGS = dict(
    M=42,
    dt=600.0,
    tmax=5,
    Phibar=3.0e5,
    omega=7.292e-5,
    a=6.37122e6,
    test=None,
    g=9.8,
    forcflag=True,
    taurad=86400.0,
    taudrag=86400.0,
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


@pytest.mark.smoke
def test_vmap_over_DPhieq_returns_finite_distinct_terminal_phi() -> None:
    """vmap run_model_scan_final over a stack of DPhieq values."""
    if not _x64_enabled():
        pytest.skip("Ensemble parity check requires x64.")

    from my_swamp.model import run_model_scan_final

    DPhieq_stack = jnp.asarray([2.0e6, 4.0e6, 6.0e6])

    def one_run(DPhieq: jnp.ndarray) -> jnp.ndarray:
        sim = run_model_scan_final(**_BASE_KWARGS, DPhieq=DPhieq)
        return sim["last_state"].Phi_curr

    Phi_stack = jax.vmap(one_run)(DPhieq_stack)
    Phi_np = np.asarray(Phi_stack)

    # Shape: (ensemble_size, J, I).
    from my_swamp.initial_conditions import spectral_params
    _N, I, J, _dt, _l, _m, _w = spectral_params(_BASE_KWARGS["M"])
    assert Phi_np.shape == (DPhieq_stack.shape[0], J, I)
    assert np.all(np.isfinite(Phi_np)), "vmap output contains NaN/Inf"

    # Members must differ from each other (different DPhieq must produce
    # different terminal Phi).
    member_diffs = np.array(
        [
            float(np.max(np.abs(Phi_np[i] - Phi_np[j])))
            for i in range(Phi_np.shape[0])
            for j in range(i + 1, Phi_np.shape[0])
        ]
    )
    assert float(np.min(member_diffs)) > 0.0, (
        "vmap ensemble members are bitwise identical — DPhieq is not entering "
        "the inner loop, which suggests broken parameter plumbing."
    )

    # Cross-check: each vmapped member should agree with a direct (unvmapped)
    # call on the same DPhieq. This catches the case where vmap changes the
    # numerics due to a broken broadcasting rule somewhere.
    for i, dpe in enumerate(np.asarray(DPhieq_stack)):
        direct = run_model_scan_final(**_BASE_KWARGS, DPhieq=float(dpe))
        direct_phi = np.asarray(direct["last_state"].Phi_curr)
        np.testing.assert_allclose(
            Phi_np[i],
            direct_phi,
            rtol=0.0,
            atol=1.0e-10,
            err_msg=f"vmap member {i} (DPhieq={dpe}) disagrees with direct call",
        )
