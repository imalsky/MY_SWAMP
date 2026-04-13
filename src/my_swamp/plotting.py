# -*- coding: utf-8 -*-
"""
This module contains functions related to generating figures and gifs with SWAMPE.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import os
import imageio


def _to_numpy(x: Any) -> np.ndarray:
    """Convert JAX or array-like input to a NumPy array."""
    # Accept NumPy arrays, JAX arrays, or array-likes.
    try:
        import jax.numpy as jnp  # type: ignore
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def mean_zonal_wind_plot(
    plotdata: np.ndarray,
    mus: np.ndarray,
    timestamp: float,
    units: str = 'hours',
    customtitle: Optional[str] = None,
    customxlabel: Optional[str] = None,
    savemyfig: bool = False,
    filename: Optional[str] = None,
    custompath: Optional[str] = None,
    color: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Generates a plot of mean zonal winds (averaged across all longitudes). 
    
    :param plotdata: Wind data, ususally U, of size (J,I).
    :type plotdata: array of float
    :param mus: Array of Gaussian latitudes of length J.
    :type mus: array of float
    :param timestamp: time of snapshot
    :type timestamp: float
    :param units: units of timestamp, defaults to 'hours'
    :type units: str, optional
    :param customtitle: option to change the title, defaults to None
    :type customtitle: string, optional
    :param customxlabel: option to change the label of the x-axis, defaults to None
    :type customxlabel: str, optional
    :param savemyfig: option to save the figure, defaults to False
    :type savemyfig: bool, optional
    :param filename: file name for saving the figure, defaults to None
    :type filename: str, optional
    :param custompath: path for saving the figure, defaults to None
    :type custompath: str, optional
    :param color: option to change the color of the plot, defaults to None
    :type color: str, optional
    :return: figure
    :rtype: matplotlib figure
    """

    #average data
    zonal_mean=np.mean(plotdata,axis=1)
    
    #convert latitudes to degrees
    Y = np.arcsin(mus)*180/np.pi #generate latitudes in degrees
    
    plt.figure()

    if color is not None:
        plt.plot(zonal_mean, Y,color=color)
    else:
        plt.plot(zonal_mean, Y)
    
    if customxlabel is None:    
        plt.xlabel('mean U, m/s')
    else:
        plt.xlabel(customxlabel)
        
    plt.ticklabel_format(axis='both', style='sci')
    plt.ylabel('latitude')
    plt.ticklabel_format(axis='both', style='sci')
    
    if customtitle is None:
        plt.title('Mean zonal winds at '+ str(timestamp) + ' '+units)
    else:
        plt.title(customtitle)
        
    if savemyfig:
        if custompath is None:
            path = 'plots/'
            isExist = os.path.exists(path)
            if not isExist:
                os.mkdir('plots/')
            plt.savefig(path+filename, bbox_inches='tight', dpi=800)
        else:
            plt.savefig(custompath+filename, bbox_inches='tight', dpi=800)
    
    plt.show()
    return plt.gcf()

# Formatter used by the geopotential colorbar and axis labels.
def fmt(x: float, pos: Any) -> str:
    """Generates the format for scientific notation in axis and colorbar labels.
    
    Parameters
    ----------
    x : Any
        Tick value to format.
    pos : Any
        Tick position passed by Matplotlib and ignored by the formatter.
    
    Returns
    -------
    Any
        MathText string representing ``x`` in scientific notation.
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def quiver_geopot_plot(
    U: np.ndarray,
    V: np.ndarray,
    Phi: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    timestamp: float,
    sparseness: int = 4,
    minlevel: Optional[float] = None,
    maxlevel: Optional[float] = None,
    units: str = 'hours',
    customtitle: Optional[str] = None,
    savemyfig: bool = False,
    filename: Optional[str] = None,
    custompath: Optional[str] = None,
    axlabels: bool = False,
    colormap: Any = None,
) -> matplotlib.figure.Figure:
    """Generates a quiver plot with the geopotential field and overlayed wind vectors.
    
    :param U: lat-lon zonal wind component, (J,I)
    :type U: array of float
    :param V: lat-lon meridional wind component, (J,I)
    :type V: array of float
    :param Phi: lat-lon geopotential field, (J,I)
    :type Phi: array of float
    :param lambdas: Uniformly spaced longitudes of length I.
    :type lambdas: array of float
    :param mus: Gaussian latitudes of length J.
    :type mus: array of float
    :param timestamp: time of snapshot
    :type timestamp: float
    :param sparseness: spacing of overlayed wind vector field, defaults to 4
    :type sparseness: int, optional
    :param minlevel: colorbar minimum for geopotential plotting, defaults to minimum of Phi
    :type minlevel: float, optional
    :param maxlevel: colorbar maximum for geopotential plotting, defaults to maximum of Phi
    :type maxlevel: float, optional
    :param units: units of timestamp, defaults to 'hours'
    :type units: str, optional
    :param customtitle: option to change the title, defaults to None
    :type customtitle: string, optional
    :param savemyfig: option to save the figure, defaults to False
    :type savemyfig: bool, optional
    :param filename: file name for saving the figure, defaults to None
    :type filename: str, optional
    :param custompath: path for saving the figure, defaults to None
    :type custompath: str, optional
    :param axlabels: option to display axis labels for latitude and longitude, defaults to False
    :type axlabels: bool, optional
    :param colormap: option to change the colormap, defaults to nipy.spectral
    :type colormap: matplotlib colormap, optional
    :return: figure
    :rtype: matplotlib figure
    """
    # set up lat-lon grid
    X = lambdas*180/np.pi
    Y = np.arcsin(mus)*180/np.pi
    fig, ax=plt.subplots()
    # Plot the geopotential surface.
    ax.contourf(X, Y, (Phi))

    if minlevel is None:
        minlevel=np.min(Phi)
    if maxlevel is None:
        maxlevel=np.max(Phi)
        
    if colormap is None:
        colormap=cm.nipy_spectral
    #set the colorbar limits
    levels =np.linspace(minlevel, maxlevel)
    CS = plt.contourf(X, Y, (Phi), levels=levels, cmap=colormap, extend='both') 
    plt.colorbar(CS, format=ticker.FuncFormatter(fmt))

    # set sparce grid to display wind vector field    
    Xsparse=X[0::sparseness]
    Ysparse=Y[0::sparseness]

    Usparse=U[0::sparseness,0::sparseness]
    Vsparse=V[0::sparseness,0::sparseness]
    
    #to prevent aliasing if saving to pdf
    for c in ax.collections:
        c.set_edgecolor("face")   

    plt.quiver(Xsparse,Ysparse,Usparse,Vsparse)
    
    if axlabels:
        ax.set_ylabel('latitude')
        ax.set_xlabel('longitude')
    
    if customtitle is None:
        ax.set_title('Geopotential at '+ str(timestamp) + ' '+units)
    else:
        ax.set_title(customtitle)
    
    if savemyfig:
        if custompath is None:
            path = 'plots/'
            isExist = os.path.exists(path)
            if not isExist:
                os.mkdir('plots/')
            plt.savefig(path+filename, bbox_inches='tight', dpi=800)
        else:
            plt.savefig(custompath+filename, bbox_inches='tight', dpi=800)
        
    return fig
    

def spinup_plot(
    plotdata: np.ndarray,
    dt: float,
    units: str = 'hours',
    customtitle: Optional[str] = None,
    customxlabel: Optional[str] = None,
    customylabel: Optional[str] = None,
    savemyfig: bool = False,
    filename: Optional[str] = None,
    custompath: Optional[str] = None,
    color: Optional[List[str]] = None,
    legendflag: bool = True,
    customlegend: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """Generates a plot of RMS winds and minimal winds over time. Can be useful for monitoring spinup.
    
    :param plotdata: Spinup winds, of size (2,tmax).
    :type plotdata: array of float
    :param dt: timestep length, in seconds
    :type dt: float
    :param units: units of timestamp, defaults to 'hours'
    :type units: str, optional
    :param customtitle: option to change the title, defaults to None
    :type customtitle: string, optional
    :param customxlabel: option to change the label of the x-axis, defaults to None
    :type customxlabel: str, optional
    :param customylabel: option to change the label of the y-axis, defaults to None
    :type customylabel: str, optional
    :param savemyfig: option to save the figure, defaults to False
    :type savemyfig: bool, optional
    :param filename: file name for saving the figure, defaults to None
    :type filename: str, optional
    :param custompath: path for saving the figure, defaults to None
    :type custompath: str, optional
    :param color: option to specify array of two colors ["color1", "color2"], defaults to None
    :type color: array of string, optional
    :param legendflag: option to display legend, defaults to True
    :type legendflag: bool, optional
    :param customlegend: option to customize the legend, defaults to None
    :type customlegend: array of string, optional
    :return: figure
    :rtype: matplotlib figure
    """
    tmax=np.shape(plotdata)[0]
    
    if units=='hours':
        tlim=dt*tmax/3600
    elif units=='minutes':
        tlim=dt*tmax/60
    elif units=='seconds':
        tlim=dt*tmax
    else:
        print('Cannot parse units. Acceptable units are: hours, minutes, seconds.')
    
    t = np.linspace(0, tlim, num=tmax, endpoint=True)
    plt.figure()
    if color is not None:
        plt.plot(t, plotdata[:,0],color=color[0])
        plt.plot(t, plotdata[:,1],color=color[1])
    else:
        plt.plot(t, plotdata[:,0])
        plt.plot(t, plotdata[:,1])
    
    
    if legendflag:
        if customlegend is None:
            plt.legend(['min winds','RMS winds'],loc="center right")
        else:
            plt.legend(customlegend,loc="center right")
    
    if customxlabel is None:
        plt.xlabel('time, '+units)
    else:
        plt.xlabel(customxlabel)
    
    if customylabel is None:
        plt.ylabel('Winds, m/s')
    else:
        plt.ylabel(customylabel)
        
    plt.ticklabel_format(axis='both', style='sci')
    
    if customtitle is None:
        plt.title('RMS winds during spin-up')
    else:
        plt.title(customtitle)
    
    if savemyfig:
        if custompath is None:
            path = 'plots/'
            isExist = os.path.exists(path)
            if not isExist:
                os.mkdir('plots/')
            plt.savefig(path+filename, bbox_inches='tight', dpi=800)
        else:
            plt.savefig(custompath+filename, bbox_inches='tight', dpi=800)
    
    plt.show()
    return plt.gcf()



def gif_helper(fig: matplotlib.figure.Figure, dpi: int = 200) -> np.ndarray:
    """Converts the figure to image format for gif generation.
    
    :param fig: figure
    :type fig: matplotlib figure
    :param dpi: resolution, defaults to 200
    :type dpi: int, optional
    :return: image
    :rtype: numerical image
    """
    fig.set_dpi(dpi)
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

def write_quiver_gif(
    lambdas: np.ndarray,
    mus: np.ndarray,
    Phidata: np.ndarray,
    Udata: np.ndarray,
    Vdata: np.ndarray,
    timestamps: np.ndarray,
    filename: str,
    frms: int = 5,
    sparseness: int = 4,
    dpi: int = 200,
    minlevel: Optional[float] = None,
    maxlevel: Optional[float] = None,
    units: str = 'hours',
    customtitle: Optional[str] = None,
    custompath: Optional[str] = None,
    axlabels: bool = False,
    colormap: Any = None,
) -> None:
    """Writes a gif generated from a series of geopotential quiver plots.
    
    :param lambdas: Uniformly spaced longitudes of length I.
    :type lambdas: array of float
    :param mus: Gaussian latitudes of length J.
    :type mus: array of float
    :param Phidata: array of geopotential snapshots (num_snapshots,J,I)
    :type Phidata: array of float
    :param Udata: array of zonal wind component snapshots (num_snapshots,J,I)
    :type Udata: array of float
    :param Vdata: array of meridional wind component snapshots (num_snapshots,J,I)
    :type Vdata: array of float
    :param timestamps: array of timestamps for the snapshots, length corresponding to (num_snapshots)
    :type timestamps: array of float
    :param filename: name of the gif file, should end in ".gif"
    :type filename: str
    :param frms: frames per second, defaults to 5
    :type frms: int, optional
    :param sparseness: spacing of the wind vector field, defaults to 4
    :type sparseness: int, optional
    :param dpi: resolution, defaults to 200
    :type dpi: int, optional
    :param minlevel: colorbar minimum for geopotential plotting, defaults to minimum of Phi
    :type minlevel: float, optional
    :param maxlevel: colorbar maximum for geopotential plotting, defaults to maximum of Phi
    :type maxlevel: float, optional
    :param units: units of timestamp, defaults to 'hours'
    :type units: str, optional
    :param customtitle: option to change the title, defaults to None
    :type customtitle: string, optional
    :param custompath: path for saving the figure, defaults to None
    :type custompath: str, optional
    :param axlabels: option to display axis labels for latitude and longitude, defaults to False
    :type axlabels: bool, optional
    :param colormap: option to change the colormap, defaults to nipy.spectral
    :type colormap: matplotlib colormap, optional
    
    """
    Udata = _to_numpy(Udata)
    Vdata = _to_numpy(Vdata)
    Phidata = _to_numpy(Phidata)
    lambdas = _to_numpy(lambdas)
    mus = _to_numpy(mus)
    if minlevel is None:
        minlevel=np.min(Phidata)
    if maxlevel is None:
        maxlevel=np.max(Phidata)
    imageio.mimsave('./'+filename, [gif_helper(quiver_geopot_plot(Udata[i,:,:],Vdata[i,:,:],Phidata[i,:,:],lambdas,mus,timestamps[i],sparseness=sparseness,minlevel=minlevel,maxlevel=maxlevel,units=units,customtitle=customtitle,custompath=None,axlabels=axlabels,colormap=colormap),dpi=200) for i in range(len(timestamps))], fps=frms)
