from __future__ import annotations

import numpy as np


def test_spectral_params_shapes() -> None:
    import jax  # noqa: F401  # pylint: disable=unused-import
    from my_swamp import initial_conditions as ic

    N, I, J, dt, lambdas, mus, w = ic.spectral_params(42)

    assert N == 42
    assert I == 128
    assert J == 64
    assert float(dt) > 0.0

    # Basic shape checks (these are the main failure modes when basis construction changes).
    assert np.asarray(lambdas).shape == (I,)
    assert np.asarray(mus).shape == (J,)
    assert np.asarray(w).shape == (J,)
