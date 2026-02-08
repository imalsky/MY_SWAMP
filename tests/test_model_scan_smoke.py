from __future__ import annotations

import numpy as np


def test_run_model_scan_smoke() -> None:
    import jax  # noqa: F401  # pylint: disable=unused-import
    from my_swamp.model import run_model_scan

    # Minimal run: one scan step (t starts at 2 by SWAMPE convention).
    res = run_model_scan(
        M=42,
        dt=30.0,
        tmax=3,
        Phibar=3.0e3,
        omega=7.2921159e-5,
        a=6.37122e6,
        test=1,
        g=9.8,
        forcflag=False,
        diffflag=False,
        modalflag=True,
        expflag=False,
        jit_scan=False,
    )

    assert set(res.keys()) >= {"static", "t_seq", "outs", "last_state", "starttime"}

    static = res["static"]
    t_seq = np.asarray(res["t_seq"])
    outs = res["outs"]

    assert t_seq.ndim == 1
    assert t_seq.size == 1
    assert int(t_seq[0]) == 2  # default start time when not using continuation

    # Core field outputs (time, lat, lon)
    for key in ("eta", "delta", "Phi", "U", "V"):
        arr = np.asarray(outs[key])
        assert arr.shape == (t_seq.size, static.J, static.I)
        assert np.isfinite(arr).all(), f"{key} contains NaN/Inf"

    # Scalar diagnostics (time,)
    for key in ("rms", "spin_min", "phi_min", "phi_max"):
        arr = np.asarray(outs[key])
        assert arr.shape == (t_seq.size,)
        assert np.isfinite(arr).all(), f"{key} contains NaN/Inf"
