#!/usr/bin/env python3
"""Generate the JOSS paper's differentiability figure (Figure 2).

This is the reproducible source for ``paper/temperature_sensitivity_perhour_100d.png``
and its companion data file ``temperature_sensitivity_perhour_100d.npz``.

What it shows
-------------
A forced hot-Jupiter integration in the same regime as the parity figure
(``compare_long_run_parity.py``; M=42, tau_rad=10 h, tau_drag=6 h), run to 100
days, and the *spatial sensitivity* of the temperature field to the two forcing
timescales, obtained by automatic differentiation through the full time
integration:

    panel 1  T(lon, lat)                 -- the temperature field
    panel 2  dT/dtau_rad  [K per hour]   -- forward-mode AD, one pass
    panel 3  dT/dtau_drag [K per hour]   -- forward-mode AD, one pass

Temperature proxy
-----------------
The shallow-water model is prognostic in geopotential ``Phi`` (units m^2 s^-2),
not temperature. We map geopotential to a representative temperature with the
dry-static-energy proxy

    T = (Phibar + Phi) / R_d ,

with ``R_d`` the specific gas constant of an H2/He atmosphere (default
3779 J kg^-1 K^-1). This is an illustrative proxy, stated in the caption; it is a
single global linear map, so dT/dtau = (1/R_d) dPhi/dtau and the per-hour
conversion is exact.

Validation
----------
Each AD sensitivity field is cross-checked against a finite-difference response:
for a fractional step ``pert`` we compare the AD linear prediction
``dT/dtau * (pert*tau)`` to the actual change ``T(tau*(1+pert)) - T(tau)``, and
report the coefficient of determination R^2 across all grid points -- for *both*
the radiative and the drag panels.

Usage
-----
    python scripts/make_sensitivity_figure.py                # 100 days (paper)
    python scripts/make_sensitivity_figure.py --days 5       # fast smoke test
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
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from my_swamp.initial_conditions import spectral_params  # noqa: E402
from my_swamp.model import run_model_scan_final  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Forced hot-Jupiter regime -- identical to the parity figure (Figure 1).
M = 42
DT = 120.0
DPHIEQ = 1.0e6
PHIBAR = 3.0e5
OMEGA = 3.2e-5
RADIUS = 8.2e7
TAURAD = 10.0 * 3600.0
TAUDRAG = 6.0 * 3600.0
K6 = 1.24e33
ALPHA = 0.01
DAY = 86400.0
HOUR = 3600.0

# Temperature proxy: T = (Phibar + Phi) / R_d for an H2/He atmosphere.
R_D = 3779.0  # J kg^-1 K^-1 (specific gas constant; ~solar-composition H2/He)


def _base_kwargs(tmax: int) -> dict:
    """Forced hot-Jupiter solver settings (matches Figure 1)."""
    return dict(
        M=M,
        dt=DT,
        tmax=tmax,
        Phibar=PHIBAR,
        omega=OMEGA,
        a=RADIUS,
        g=9.8,
        test=None,
        forcflag=True,
        diffflag=True,
        modalflag=True,
        alpha=ALPHA,
        expflag=False,
        K6=K6,
        DPhieq=DPHIEQ,
        diagnostics=False,
        jit_scan=True,
    )


def _r2(actual: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination between two fields (flattened)."""
    a = np.asarray(actual).ravel()
    p = np.asarray(pred).ravel()
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=100.0, help="Integration horizon in days.")
    parser.add_argument("--pert", type=float, default=0.2, help="Fractional step for the FD cross-check.")
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "temperature_sensitivity_perhour_100d.png")
    parser.add_argument("--npz", type=Path, default=ROOT / "data" / "temperature_sensitivity_perhour_100d.npz",
                        help="Companion data file (regenerable; default: data/).")
    args = parser.parse_args()

    tmax = int(round(args.days * DAY / DT)) + 1
    base = _base_kwargs(tmax)
    npz_path = args.npz if args.npz is not None else args.out.with_suffix(".npz")
    print(f"config: M={M} dt={DT} days={args.days} tmax={tmax} R_d={R_D} pert={args.pert}")

    # Geometry: longitudes (radians, -pi..pi) and latitudes (from Gauss nodes mu).
    _, _, _, _, lambdas, mus, _ = spectral_params(M)
    lon_deg = np.asarray(lambdas) * 180.0 / np.pi
    lat_deg = np.degrees(np.arcsin(np.asarray(mus)))

    def temp_of_taurad(taurad):
        """Terminal temperature field as a function of the radiative timescale."""
        out = run_model_scan_final(taurad=taurad, taudrag=TAUDRAG, **base)
        return (PHIBAR + out["last_state"].Phi_curr) / R_D

    def temp_of_taudrag(taudrag):
        """Terminal temperature field as a function of the drag timescale."""
        out = run_model_scan_final(taurad=TAURAD, taudrag=taudrag, **base)
        return (PHIBAR + out["last_state"].Phi_curr) / R_D

    # Warm the geometry cache so build_static is not traced under jvp.
    _ = temp_of_taurad(jnp.asarray(TAURAD)).block_until_ready()

    # Forward-mode AD: one pass returns the field (primal) AND its full-field
    # sensitivity to the scalar timescale (tangent). Convert to K per hour.
    t0 = time.perf_counter()
    T_field, dT_drad = jax.jvp(temp_of_taurad, (jnp.asarray(TAURAD),), (jnp.asarray(1.0),))
    _, dT_ddrag = jax.jvp(temp_of_taudrag, (jnp.asarray(TAUDRAG),), (jnp.asarray(1.0),))
    T_field = np.asarray(T_field.block_until_ready())
    sens_rad = np.asarray(dT_drad.block_until_ready()) * HOUR   # K per hour
    sens_drg = np.asarray(dT_ddrag.block_until_ready()) * HOUR  # K per hour
    print(f"AD sensitivities computed in {time.perf_counter() - t0:.1f} s")

    # Finite-difference cross-check of BOTH panels (linear prediction vs actual).
    d_rad = args.pert * TAURAD
    d_drg = args.pert * TAUDRAG
    T_rad_p = np.asarray(temp_of_taurad(jnp.asarray(TAURAD + d_rad)).block_until_ready())
    T_drg_p = np.asarray(temp_of_taudrag(jnp.asarray(TAUDRAG + d_drg)).block_until_ready())
    dT_rad_actual = T_rad_p - T_field
    dT_drg_actual = T_drg_p - T_field
    dT_rad_pred = (sens_rad / HOUR) * d_rad  # back to per-second * step
    dT_drg_pred = (sens_drg / HOUR) * d_drg
    r2_rad = _r2(dT_rad_actual, dT_rad_pred)
    r2_drg = _r2(dT_drg_actual, dT_drg_pred)
    print(f"FD cross-check: r2_rad={r2_rad:.5f}  r2_drag={r2_drg:.5f}")
    print(f"T range: {T_field.min():.1f} - {T_field.max():.1f} K")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        lon=lon_deg, lat=lat_deg,
        T=T_field, sens_rad=sens_rad, sens_drg=sens_drg,
        r2_rad=np.asarray(r2_rad), r2_drag=np.asarray(r2_drg),
        taurad=np.asarray(TAURAD), taudrag=np.asarray(TAUDRAG),
        days=np.asarray(args.days), dt=np.asarray(DT), pert=np.asarray(args.pert),
        R_d=np.asarray(R_D),
    )
    print(f"wrote {npz_path}")

    # --- figure -------------------------------------------------------------
    style_path = Path(__file__).resolve().parent / "science.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
    # Larger type throughout (titles, axis labels, ticks) for a print-legible figure.
    plt.rcParams.update(
        {"axes.titlesize": 21, "axes.labelsize": 20, "xtick.labelsize": 16,
         "ytick.labelsize": 16, "font.size": 16}
    )
    # Tight colorbars: a thin bar the same height as its panel, with minimal pad.
    CBAR_FRACTION = 0.046
    CBAR_PAD = 0.012
    CBAR_LABELSIZE = 18
    CBAR_TICKSIZE = 14

    def _cbar(im, ax, label):
        cb = fig.colorbar(im, ax=ax, fraction=CBAR_FRACTION, pad=CBAR_PAD)
        cb.set_label(label, fontsize=CBAR_LABELSIZE)
        cb.ax.tick_params(labelsize=CBAR_TICKSIZE)
        return cb

    extent = [float(lon_deg.min()), float(lon_deg.max()), float(lat_deg.min()), float(lat_deg.max())]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.0), constrained_layout=True)

    # Panel 1: temperature field.
    im0 = axes[0].imshow(T_field, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    axes[0].set_title("Temperature field after %d days" % round(args.days))
    _cbar(im0, axes[0], "T [K]")

    # Panels 2 & 3: AD sensitivities, diverging map centered on zero + contours.
    for ax, sens, sym, fix in (
        (axes[1], sens_rad, r"\tau_{\rm rad}", r"\tau_{\rm rad}=10\,$h"),
        (axes[2], sens_drg, r"\tau_{\rm drag}", r"\tau_{\rm drag}=6\,$h"),
    ):
        lim = float(np.max(np.abs(sens)))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        im = ax.imshow(sens, origin="lower", aspect="auto", extent=extent, cmap="PuOr_r", norm=norm)
        ax.contour(sens, levels=6, colors="0.3", linewidths=0.5,
                   extent=extent, origin="lower")
        ax.set_title(r"Sensitivity to %s timescale" % ("radiative" if "rad" in fix else "drag")
                     + "\n" + r"$\partial T/\partial %s$ at $%s" % (sym, fix))
        _cbar(im, ax, "[K per hour]")

    for ax in axes:
        ax.plot(0.0, 0.0, marker="*", color="yellow", markersize=18,
                markeredgecolor="k", markeredgewidth=1.0, zorder=5)
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        ax.set_xticks([-180, -90, 0, 90, 180])

    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
