# ruff: noqa: E741
from __future__ import annotations

import numpy as np


def _small_spectral_setup():
    from my_swamp import spectral_transform as st
    from my_swamp import time_stepping as ts

    M = 2
    N = 2
    J = 4
    I = 8

    mus, w = st.gauss_legendre(J)
    Pmn, Hmn = st.PmnHmn(J, M, N, mus)
    Pmnw = st.weighted_legendre_basis(Pmn, w)
    Hmnw = st.weighted_legendre_basis(Hmn, w)

    tstepcoeff1 = ts.tstepcoeff(J, M, 30.0, mus, 6.37122e6)
    tstepcoeff2 = ts.tstepcoeff2(J, M, 30.0, 6.37122e6)
    mJarray = ts.mJarray(J, M)
    narray = ts.narray(M, N)
    sigma = np.ones((M + 1, N + 1))
    sigmaPhi = np.ones((M + 1, N + 1))

    rng = np.random.default_rng(7)
    shape = (J, M + 1)
    arrays = {name: rng.standard_normal(shape) for name in ("etam0", "etam1", "deltam0", "deltam1", "Phim0", "Phim1", "Am", "Bm", "Cm", "Dm", "Em", "Fm", "Gm", "Um", "Vm")}
    arrays["PhiFm"] = rng.standard_normal(shape)
    return {
        "M": M,
        "N": N,
        "J": J,
        "I": I,
        "Pmn": Pmn,
        "Hmn": Hmn,
        "Pmnw": Pmnw,
        "Hmnw": Hmnw,
        "tstepcoeff1": tstepcoeff1,
        "tstepcoeff2": tstepcoeff2,
        "mJarray": mJarray,
        "narray": narray,
        "sigma": sigma,
        "sigmaPhi": sigmaPhi,
        "arrays": arrays,
    }


def test_dayside_mask_is_strict_inequality() -> None:
    from my_swamp.forcing import Phieqfun
    import jax.numpy as jnp

    lambdas = jnp.asarray([-jnp.pi / 2, 0.0, jnp.pi / 2])
    mus = jnp.asarray([0.0])
    out = Phieqfun(Phibar=10.0, DPhieq=4.0, lambdas=lambdas, mus=mus, I=3, J=1, g=9.8)

    assert float(out[0, 0]) == 10.0
    assert float(out[0, 2]) == 10.0
    assert float(out[0, 1]) > 10.0


def test_rfun_q_clamp_and_no_drag_branch() -> None:
    from my_swamp.forcing import Rfun
    import jax.numpy as jnp

    U = jnp.asarray([[2.0, -3.0]])
    V = jnp.asarray([[1.0, 5.0]])
    Q = jnp.asarray([[-2.0, 4.0]])
    Phi = jnp.asarray([[10.0, 10.0]])
    Phibar = 2.0

    F, G = Rfun(U, V, Q, Phi, Phibar, taudrag=-1.0)
    expected_ru = jnp.asarray([[0.0, -U[0, 1] * Q[0, 1] / (Phi[0, 1] + Phibar)]])
    expected_rv = jnp.asarray([[0.0, -V[0, 1] * Q[0, 1] / (Phi[0, 1] + Phibar)]])

    np.testing.assert_allclose(np.asarray(F), np.asarray(expected_ru), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(G), np.asarray(expected_rv), rtol=0.0, atol=1e-12)


def test_explicit_delta_is_carry_only_without_forcing_or_diffusion() -> None:
    from my_swamp import explicit_tdiff as exp
    from my_swamp import spectral_transform as st

    s = _small_spectral_setup()
    a = s["arrays"]

    got, _, _ = exp.delta_timestep(
        a["etam0"],
        a["etam1"],
        a["deltam0"],
        a["deltam1"],
        a["Phim0"],
        a["Phim1"],
        s["I"],
        s["J"],
        s["M"],
        s["N"],
        a["Am"],
        a["Bm"],
        a["Cm"],
        a["Dm"],
        a["Em"],
        a["Fm"],
        a["Gm"],
        a["Um"],
        a["Vm"],
        s["Pmn"],
        s["Hmn"],
        s["Pmnw"],
        s["Hmnw"],
        s["tstepcoeff1"],
        s["tstepcoeff2"],
        s["mJarray"],
        s["narray"],
        a["PhiFm"],
        30.0,
        6.37122e6,
        3.0e3,
        86400.0,
        86400.0,
        False,
        False,
        s["sigma"],
        s["sigmaPhi"],
        None,
        2,
    )
    expected = st.fwd_leg_w(a["deltam0"], s["Pmnw"])
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=0.0, atol=1e-12)


def test_modeuler_delta_uses_forced_terms_even_when_forcflag_false() -> None:
    from my_swamp import modEuler_tdiff as mod
    from my_swamp import spectral_transform as st
    import jax.numpy as jnp

    s = _small_spectral_setup()
    a = s["arrays"]

    got, _, _ = mod.delta_timestep(
        a["etam0"],
        a["etam1"],
        a["deltam0"],
        a["deltam1"],
        a["Phim0"],
        a["Phim1"],
        s["I"],
        s["J"],
        s["M"],
        s["N"],
        a["Am"],
        a["Bm"],
        a["Cm"],
        a["Dm"],
        a["Em"],
        a["Fm"],
        a["Gm"],
        a["Um"],
        a["Vm"],
        s["Pmn"],
        s["Hmn"],
        s["Pmnw"],
        s["Hmnw"],
        s["tstepcoeff1"],
        s["tstepcoeff2"],
        s["mJarray"],
        s["narray"],
        a["PhiFm"],
        30.0,
        6.37122e6,
        3.0e3,
        86400.0,
        86400.0,
        False,
        False,
        s["sigma"],
        s["sigmaPhi"],
        None,
        2,
    )

    t1 = s["tstepcoeff1"] / 4.0
    t2 = s["tstepcoeff2"] / 4.0
    B_force = a["Bm"] + a["Fm"]
    A_force = a["Am"] - a["Gm"]

    deltacomp1, deltacomp2, deltacomp4, deltacomp5, Phicomp2 = st.fwd_leg_w_batch(
        jnp.stack(
            (
                a["deltam1"],
                t1 * (1j) * s["mJarray"] * B_force,
                t2 * a["Phim1"],
                t2 * a["Em"],
                t1 * (1j) * s["mJarray"] * a["Cm"],
            ),
            axis=0,
        ),
        s["Pmnw"],
    )
    deltacomp3, Phicomp3 = st.fwd_leg_w_batch(jnp.stack((t1 * A_force, t1 * a["Dm"]), axis=0), s["Hmnw"])
    expected = (
        deltacomp1
        + deltacomp2
        + deltacomp3
        + s["narray"] * deltacomp4
        + s["narray"] * deltacomp5
        + (s["narray"] * (Phicomp2 + Phicomp3)) / (2.0 * (6.37122e6**2))
        - 3.0e3 * (s["narray"] * deltacomp1) / (6.37122e6**2)
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=0.0, atol=1e-10)


def test_modeuler_eta_forced_unforced_asymmetry() -> None:
    from my_swamp import modEuler_tdiff as mod
    from my_swamp import spectral_transform as st

    s = _small_spectral_setup()
    a = s["arrays"]

    got_forced, _, _ = mod.eta_timestep(
        a["etam0"],
        a["etam1"],
        a["deltam0"],
        a["deltam1"],
        a["Phim0"],
        a["Phim1"],
        s["I"],
        s["J"],
        s["M"],
        s["N"],
        a["Am"],
        a["Bm"],
        a["Cm"],
        a["Dm"],
        a["Em"],
        a["Fm"],
        a["Gm"],
        a["Um"],
        a["Vm"],
        s["Pmn"],
        s["Hmn"],
        s["Pmnw"],
        s["Hmnw"],
        s["tstepcoeff1"],
        s["tstepcoeff2"],
        s["mJarray"],
        s["narray"],
        a["PhiFm"],
        30.0,
        6.37122e6,
        3.0e3,
        86400.0,
        86400.0,
        True,
        False,
        s["sigma"],
        s["sigmaPhi"],
        None,
        2,
    )
    got_unforced, _, _ = mod.eta_timestep(
        a["etam0"],
        a["etam1"],
        a["deltam0"],
        a["deltam1"],
        a["Phim0"],
        a["Phim1"],
        s["I"],
        s["J"],
        s["M"],
        s["N"],
        a["Am"],
        a["Bm"],
        a["Cm"],
        a["Dm"],
        a["Em"],
        a["Fm"],
        a["Gm"],
        a["Um"],
        a["Vm"],
        s["Pmn"],
        s["Hmn"],
        s["Pmnw"],
        s["Hmnw"],
        s["tstepcoeff1"],
        s["tstepcoeff2"],
        s["mJarray"],
        s["narray"],
        a["PhiFm"],
        30.0,
        6.37122e6,
        3.0e3,
        86400.0,
        86400.0,
        False,
        False,
        s["sigma"],
        s["sigmaPhi"],
        None,
        2,
    )

    expected_forced = st.fwd_leg_w(a["etam1"], s["Pmnw"]) - st.fwd_leg_w(
        s["tstepcoeff1"] * (1j) * s["mJarray"] * (a["Am"] - a["Gm"]), s["Pmnw"]
    ) + st.fwd_leg_w(s["tstepcoeff1"] * (a["Bm"] + a["Fm"]), s["Hmnw"])
    expected_unforced = st.fwd_leg_w(a["etam1"], s["Pmnw"]) - st.fwd_leg_w(
        (s["tstepcoeff1"] / 2.0) * (1j) * s["mJarray"] * a["Am"], s["Pmnw"]
    ) + st.fwd_leg_w((s["tstepcoeff1"] / 2.0) * a["Bm"], s["Hmnw"])

    np.testing.assert_allclose(np.asarray(got_forced), np.asarray(expected_forced), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got_unforced), np.asarray(expected_unforced), rtol=0.0, atol=1e-12)
    assert not np.allclose(np.asarray(got_forced), np.asarray(got_unforced))
