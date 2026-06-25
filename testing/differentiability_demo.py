#!/usr/bin/env python3
"""Differentiability demo: phase-curve observables versus the forcing timescales.

This is a worked example of the capability that distinguishes ``my_swamp`` from
the reference NumPy ``SWAMPE``: because the entire shallow-water integration is
differentiable, we obtain not only how an observable physical quantity depends on
the model inputs, but also its exact sensitivity (gradient) to those inputs at a
cost comparable to a single forward run.

Two intuitive observables of a hot Jupiter's thermal phase curve are swept
against the timescales that control heat redistribution (Perez-Becker & Showman
2013):

* the day--night geopotential contrast (the shallow-water proxy for the
  day--night temperature contrast, which sets the phase-curve amplitude) versus
  the radiative timescale ``taurad``; and
* the hot-spot offset (the longitude of the wave-1 maximum of the geopotential,
  i.e. the eastward shift of the hottest point) versus the drag timescale
  ``taudrag``.

For each, the slope ``d(observable)/d(tau)`` from a single reverse-mode gradient
(``jax.grad``) is overlaid as a tangent; the tangents lie along the curves,
showing that automatic differentiation returns the exact local sensitivity.

    python testing/differentiability_demo.py --days 12 --npts 10
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("SWAMPE_JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from my_swamp.initial_conditions import spectral_params  # noqa: E402
from my_swamp.model import run_model_scan_final  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Forced hot-Jupiter regime (matches the parity figure).
M = 42
DT = 120.0
DPHIEQ = 1.0e6
TAURAD_FIX = 10.0 * 3600.0  # held fixed while sweeping taudrag
TAUDRAG_FIX = 6.0 * 3600.0  # held fixed while sweeping taurad
DAY = 86400.0
RAD2DEG = 180.0 / np.pi


def _base_kwargs(tmax: int) -> dict:
    return dict(
        M=M,
        dt=DT,
        tmax=tmax,
        Phibar=3.0e5,
        omega=3.2e-5,
        a=8.2e7,
        g=9.8,
        test=None,
        forcflag=True,
        diffflag=True,
        modalflag=True,
        alpha=0.01,
        expflag=False,
        K6=1.24e33,
        DPhieq=DPHIEQ,
        diagnostics=False,
        jit_scan=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=12.0, help="Integration horizon (>= a few x max tau).")
    parser.add_argument("--npts", type=int, default=10, help="Sweep points per timescale.")
    parser.add_argument("--tau-min-days", type=float, default=0.1)
    parser.add_argument("--tau-max-days", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=ROOT / "testing" / "differentiability_demo.png")
    args = parser.parse_args()

    tmax = int(round(args.days * DAY / DT)) + 1
    base = _base_kwargs(tmax)
    print(f"config: M={M} dt={DT} days={args.days} tmax={tmax} npts={args.npts}")

    # Latitude quadrature weights and longitudes for the spatial reductions.
    _, I, J, _, lambdas, mus, w = spectral_params(M)
    lambdas_np = np.asarray(lambdas)
    w_lat = jnp.asarray(np.asarray(w)[:, None])  # (J, 1)
    lam = jnp.asarray(lambdas_np)[None, :]  # (1, I)
    day_mask = jnp.asarray((np.abs(lambdas_np) < np.pi / 2.0).astype(np.float64))[None, :]
    night_mask = 1.0 - day_mask

    def contrast(phi: jnp.ndarray) -> jnp.ndarray:
        """Area-weighted dayside-mean minus nightside-mean geopotential."""
        day = jnp.sum(phi * w_lat * day_mask) / jnp.sum(w_lat * day_mask)
        night = jnp.sum(phi * w_lat * night_mask) / jnp.sum(w_lat * night_mask)
        return day - night

    def offset_deg(phi: jnp.ndarray) -> jnp.ndarray:
        """Hot-spot offset: longitude (deg) of the wave-1 maximum of Phi."""
        sx = jnp.sum(phi * jnp.cos(lam) * w_lat)
        sy = jnp.sum(phi * jnp.sin(lam) * w_lat)
        return jnp.arctan2(sy, sx) * RAD2DEG

    # Equilibrium contrast (normalization): contrast of the radiative-equilibrium
    # perturbation DPhieq*cos(lam)*sqrt(1-mu^2) on the dayside.
    mu = jnp.asarray(np.asarray(mus)[:, None])
    phi_eq_pert = DPHIEQ * jnp.cos(lam) * jnp.sqrt(1.0 - mu**2) * day_mask
    c_eq = float(contrast(phi_eq_pert))
    print(f"equilibrium contrast C_eq = {c_eq:.4e} m^2/s^2")

    def contrast_of_taurad(taurad: jnp.ndarray) -> jnp.ndarray:
        out = run_model_scan_final(taurad=taurad, taudrag=TAUDRAG_FIX, **base)
        return contrast(out["last_state"].Phi_curr) / c_eq

    def offset_of_taudrag(taudrag: jnp.ndarray) -> jnp.ndarray:
        out = run_model_scan_final(taurad=TAURAD_FIX, taudrag=taudrag, **base)
        return offset_deg(out["last_state"].Phi_curr)

    def contrast_of_taudrag(taudrag: jnp.ndarray) -> jnp.ndarray:
        out = run_model_scan_final(taurad=TAURAD_FIX, taudrag=taudrag, **base)
        return contrast(out["last_state"].Phi_curr) / c_eq

    # Warm the geometry cache eagerly so build_static is not traced under jit.
    _ = contrast_of_taurad(jnp.asarray(TAURAD_FIX))

    val_rad = jax.jit(contrast_of_taurad)
    val_off = jax.jit(offset_of_taudrag)
    val_cdrag = jax.jit(contrast_of_taudrag)
    grad_rad = jax.jit(jax.grad(contrast_of_taurad))
    grad_off = jax.jit(jax.grad(offset_of_taudrag))

    taus = np.logspace(np.log10(args.tau_min_days), np.log10(args.tau_max_days), args.npts) * DAY
    taus_d = taus / DAY

    c_rad = np.array([float(val_rad(jnp.asarray(t))) for t in taus])
    o_drag = np.array([float(val_off(jnp.asarray(t))) for t in taus])
    c_drag = np.array([float(val_cdrag(jnp.asarray(t))) for t in taus])
    print("taurad sweep  contrast/C_eq:", np.round(c_rad, 3))
    print("taudrag sweep hot-spot deg :", np.round(o_drag, 2))
    print("taudrag sweep contrast/C_eq:", np.round(c_drag, 3))

    # Automatic-differentiation tangents at three points per curve.
    tan_idx = [args.npts // 5, args.npts // 2, (4 * args.npts) // 5]
    g_rad = {i: float(grad_rad(jnp.asarray(taus[i]))) for i in tan_idx}
    g_off = {i: float(grad_off(jnp.asarray(taus[i]))) for i in tan_idx}

    # --- figure (science.mplstyle, enlarged fonts) ---
    style_path = Path(__file__).resolve().parent / "science.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "lines.linewidth": 2.5,
            "font.size": 16,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    panels = [
        (axes[0], c_rad, g_rad, r"Radiative timescale $\tau_{\rm rad}$ [days]",
         r"Day--night contrast / equilibrium", "Day--night temperature contrast", "C0"),
        (axes[1], o_drag, g_off, r"Drag timescale $\tau_{\rm drag}$ [days]",
         r"Hot-spot offset [deg]", "Hot-spot offset", "C1"),
    ]
    for ax, yvals, gvals, xlabel, ylabel, title, color in panels:
        ax.plot(taus_d, yvals, color=color, marker="o", markersize=6, zorder=2)
        for j, i in enumerate(tan_idx):
            t0, y0, s = taus[i], yvals[i], gvals[i]  # s = d(obs)/d(tau) [obs-units / s]
            tt = np.linspace(t0 * 0.6, t0 * 1.55, 20)
            ax.plot(
                tt / DAY, y0 + s * (tt - t0),
                color="k", lw=2.2, zorder=3,
                label="AD gradient (tangent)" if j == 0 else None,
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", frameon=True)

    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
