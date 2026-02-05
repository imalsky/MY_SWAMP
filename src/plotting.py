# -*- coding: utf-8 -*-
"""Plotting utilities for SWAMPE (JAX port).

This file is API-compatible with the original SWAMPE plotting module, with two
practical adjustments for batch/HPC usage:

1) All inputs are accepted as either NumPy arrays or JAX arrays; values are
   converted to NumPy on entry.
2) Each plotting function returns a matplotlib Figure object. (The original
   code often returned the result of plt.plot, which is not a Figure.)

A `show` keyword is added (default False) to avoid accidental interactive popups
during non-interactive runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
if os.getenv("DISPLAY", "") == "":
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import imageio


def _to_numpy(x) -> np.ndarray:
    """Convert a NumPy/JAX array-like to NumPy without importing JAX."""
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def mean_zonal_wind_plot(
    plotdata,
    mus,
    timestamp,
    units: str = "hours",
    customtitle: Optional[str] = None,
    customxlabel: Optional[str] = None,
    savemyfig: bool = False,
    filename: Optional[str] = None,
    custompath: Optional[str] = None,
    color: Optional[str] = None,
    show: bool = False,
):
    """Generates a plot of mean zonal winds (averaged across all longitudes)."""
    plotdata = _to_numpy(plotdata)
    mus = _to_numpy(mus)

    zonal_mean = np.mean(plotdata, axis=1)
    Y = np.arcsin(mus) * 180 / np.pi

    fig, ax = plt.subplots()

    if color is not None:
        ax.plot(zonal_mean, Y, color=color)
    else:
        ax.plot(zonal_mean, Y)

    ax.set_xlabel("mean U, m/s" if customxlabel is None else customxlabel)
    ax.set_ylabel("latitude")
    ax.ticklabel_format(axis="both", style="sci")

    ax.set_title(f"Mean zonal winds at {timestamp} {units}" if customtitle is None else customtitle)

    if savemyfig:
        if filename is None:
            raise ValueError("filename must be provided when savemyfig=True")
        outdir = Path(custompath) if custompath is not None else Path("plots")
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / filename, bbox_inches="tight", dpi=800)

    if show:
        plt.show()

    return fig


def fmt(x, pos):
    """Scientific-notation formatter for axis and colorbar labels."""
    a, b = "{:.2e}".format(x).split("e")
    return r"${} \times 10^{{{}}}$".format(a, int(b))


def quiver_geopot_plot(
    U,
    V,
    Phi,
    lambdas,
    mus,
    timestamp,
    sparseness: int = 4,
    minlevel=None,
    maxlevel=None,
    units: str = "hours",
    customtitle: Optional[str] = None,
    savemyfig: bool = False,
    filename: Optional[str] = None,
    custompath: Optional[str] = None,
    axlabels: bool = False,
    colormap=None,
    show: bool = False,
):
    """Quiver plot of winds over geopotential contours."""
    U = _to_numpy(U)
    V = _to_numpy(V)
    Phi = _to_numpy(Phi)
    lambdas = _to_numpy(lambdas)
    mus = _to_numpy(mus)

    # convert to degrees
    X = lambdas * 180 / np.pi
    Y = np.arcsin(mus) * 180 / np.pi

    if minlevel is None:
        minlevel = float(np.min(Phi))
    if maxlevel is None:
        maxlevel = float(np.max(Phi))

    levels = np.linspace(minlevel, maxlevel, num=11, endpoint=True)

    cmap = cm.nipy_spectral if colormap is None else colormap

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Phi, levels=levels, cmap=cmap)

    ax.quiver(
        X[::sparseness],
        Y[::sparseness],
        U[::sparseness, ::sparseness],
        V[::sparseness, ::sparseness],
        pivot="mid",
        scale=600,
    )

    if axlabels:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")

    ax.set_title(f"Geopotential and winds at {timestamp} {units}" if customtitle is None else customtitle)

    cbar = fig.colorbar(contour, ax=ax, format=ticker.FuncFormatter(fmt))
    cbar.set_label("geopotential")

    if savemyfig:
        if filename is None:
            raise ValueError("filename must be provided when savemyfig=True")
        outdir = Path(custompath) if custompath is not None else Path("plots")
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / filename, bbox_inches="tight", dpi=800)

    if show:
        plt.show()

    return fig


def spinup_plot(
    plotdata,
    dt,
    units: str = "hours",
    customtitle: Optional[str] = None,
    customxlabel: Optional[str] = None,
    customylabel: Optional[str] = None,
    savemyfig: bool = False,
    filename: Optional[str] = None,
    custompath: Optional[str] = None,
    color=None,
    legendflag: bool = True,
    customlegend=None,
    show: bool = False,
):
    """Plot RMS winds and minimal winds over time (spinup diagnostic)."""
    plotdata = _to_numpy(plotdata)

    tmax = int(np.shape(plotdata)[0])

    if units == "hours":
        tlim = dt * tmax / 3600
    elif units == "minutes":
        tlim = dt * tmax / 60
    elif units == "seconds":
        tlim = dt * tmax
    else:
        raise ValueError("Cannot parse units. Acceptable units are: hours, minutes, seconds.")

    t = np.linspace(0, tlim, num=tmax, endpoint=True)

    fig, ax = plt.subplots()

    if color is not None:
        ax.plot(t, plotdata[:, 0], color=color[0])
        ax.plot(t, plotdata[:, 1], color=color[1])
    else:
        ax.plot(t, plotdata[:, 0])
        ax.plot(t, plotdata[:, 1])

    ax.set_xlabel(f"time ({units})" if customxlabel is None else customxlabel)
    ax.set_ylabel("velocity (m/s)" if customylabel is None else customylabel)
    ax.ticklabel_format(axis="both", style="sci")

    if legendflag:
        legend = ["min wind", "rms wind"] if customlegend is None else customlegend
        ax.legend(legend)

    ax.set_title("Spinup" if customtitle is None else customtitle)

    if savemyfig:
        if filename is None:
            raise ValueError("filename must be provided when savemyfig=True")
        outdir = Path(custompath) if custompath is not None else Path("plots")
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / filename, bbox_inches="tight", dpi=800)

    if show:
        plt.show()

    return fig


def gif_helper(fig, dpi: int = 200):
    """Convert a figure canvas to an RGB image array for GIF generation."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def write_quiver_gif(
    lambdas,
    mus,
    Phidata,
    Udata,
    Vdata,
    timestamps,
    filename: str,
    frms: int = 5,
    sparseness: int = 4,
    dpi: int = 200,
    minlevel=None,
    maxlevel=None,
    units: str = "hours",
    customtitle: Optional[str] = None,
    custompath: Optional[str] = None,
    axlabels: bool = False,
    colormap=None,
):
    """Write a GIF generated from a series of geopotential quiver plots."""
    lambdas = _to_numpy(lambdas)
    mus = _to_numpy(mus)
    Phidata = _to_numpy(Phidata)
    Udata = _to_numpy(Udata)
    Vdata = _to_numpy(Vdata)
    timestamps = _to_numpy(timestamps)

    if minlevel is None:
        minlevel = float(np.min(Phidata))
    if maxlevel is None:
        maxlevel = float(np.max(Phidata))

    outpath = Path(custompath) if custompath is not None else Path(".")
    outpath.mkdir(parents=True, exist_ok=True)

    images = []
    for i in range(Phidata.shape[0]):
        fig = quiver_geopot_plot(
            Udata[i, :, :],
            Vdata[i, :, :],
            Phidata[i, :, :],
            lambdas,
            mus,
            timestamps[i],
            sparseness=sparseness,
            minlevel=minlevel,
            maxlevel=maxlevel,
            units=units,
            customtitle=customtitle,
            custompath=None,
            axlabels=axlabels,
            colormap=colormap,
            show=False,
        )
        images.append(gif_helper(fig, dpi=dpi))
        plt.close(fig)

    imageio.mimsave(str(outpath / filename), images, fps=frms)
