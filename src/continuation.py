"""
This module contains the functions needed to save and read SWAMPE data, as well as for continuation.

This mirrors the SWAMPE numpy continuation.py interface, but uses pathlib and
does minimal type conversion so it can work with either numpy arrays or
JAX DeviceArrays (which are converted to numpy when pickled).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np


def _as_path(custompath: Optional[str]) -> Path:
    if custompath is None:
        return Path("data")
    return Path(custompath)


def write_pickle(filename: str, data: Any, custompath: Optional[str] = None) -> None:
    """
    Writes a pickle file from the data.
    
    Parameters
    ----------
    filename : str
        name of the pickle file to be saved
    data : Any
        a Python array of data to be saved
    custompath : str, optional
        path to the custom directory, defaults to None. 
        If None, files will be saved in the parent_directory/data/.
    """
    base = _as_path(custompath)
    base.mkdir(parents=True, exist_ok=True)
    
    path = base / filename
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filename: str, custompath: Optional[str] = None) -> Any:
    """
    Loads a pickle file.
    
    Parameters
    ----------
    filename : str
        name of the pickle file to be read
    custompath : str, optional
        path to the custom directory, defaults to None

    Returns
    -------
    Any
        data from the pickle file
    """
    base = _as_path(custompath)
    path = base / filename
    with path.open("rb") as f:
        return pickle.load(f)


def compute_timestamp(units: str, t: int, dt: float) -> str:
    """
    Computes timestamp in appropriate units to append to the saved data files.

    Parameters
    ----------
    units : str
        Units of timestamps on the savefile: 'hours', 'minutes', or 'seconds'
    t : int
        number of current timestep
    dt : float
        timestep length, in seconds

    Returns
    -------
    str
        timestamp in desired units
    """
    if units == 'hours':
        timestamp = str(int(dt * t / 3600))
    elif units == 'minutes':
        timestamp = str(int(dt * t / 60))
    elif units == 'seconds':
        timestamp = str(int(dt * t))
    elif units == 'steps':
        timestamp = str(int(t))
    elif units == 'days':
        timestamp = str(int(dt * t / (3600 * 24)))
    else:
        raise ValueError('Cannot parse units. Acceptable units are: hours, minutes, seconds, steps, days.')
    
    return timestamp


def compute_t_from_timestamp(units: str, timestamp: int, dt: float) -> int:
    """
    Computes the current timestep t based on timestamp, units, and timestep size.

    Parameters
    ----------
    units : str
        Units of timestamps on the savefile: 'hours', 'minutes', or 'seconds'
    timestamp : int
        Timestamp in specified units
    dt : float
        timestep length, in seconds

    Returns
    -------
    int
        number of timestep to continue the simulation
    """
    if units == 'hours':
        t = int(timestamp * 3600 / dt)
    elif units == 'minutes':
        t = int(timestamp * 60 / dt)
    elif units == 'seconds':
        t = int(timestamp / dt)
    elif units == 'steps':
        t = int(timestamp)
    elif units == 'days':
        t = int(timestamp * 3600 * 24 / dt)
    else:
        raise ValueError('Cannot parse units. Acceptable units are: hours, minutes, seconds, steps, days.')

    return t


def save_data(
    timestamp: Union[int, str],
    etadata: Any,
    deltadata: Any,
    Phidata: Any,
    U: Any,
    V: Any,
    spinupdata: Any,
    geopotdata: Any,
    custompath: Optional[str] = None
) -> None:
    """
    Saves the data for plotting and continuation purposes.
    
    Parameters
    ----------
    timestamp : str or int
        timestamp to be used for naming saved files
    etadata : array (J, I)
        data for absolute vorticity eta
    deltadata : array (J, I)
        data for divergence delta
    Phidata : array (J, I)
        data for geopotential Phi
    U : array (J, I)
        data for zonal winds U
    V : array (J, I)
        data for meridional winds V
    spinupdata : array
        time series array of minimum length of wind vector and RMS winds
    geopotdata : array
        time series array of minimum and maximum of the geopotential Phi
    custompath : str, optional
        path to the custom directory, defaults to None. 
        If None, files will be saved in the parent_directory/data/
    """
    ts = str(timestamp)
    
    write_pickle('eta-' + ts, np.asarray(etadata), custompath=custompath)
    write_pickle('delta-' + ts, np.asarray(deltadata), custompath=custompath)
    write_pickle('Phi-' + ts, np.asarray(Phidata), custompath=custompath)
    write_pickle('U-' + ts, np.asarray(U), custompath=custompath)
    write_pickle('V-' + ts, np.asarray(V), custompath=custompath)
    
    write_pickle('spinup-winds', np.asarray(spinupdata), custompath=custompath)
    write_pickle('spinup-geopot', np.asarray(geopotdata), custompath=custompath)


def load_data(
    timestamp: Union[int, str],
    custompath: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the data necessary for continuation based on timestamp.
    
    Parameters
    ----------
    timestamp : str or int
        timestamp used for naming saved files
    custompath : str, optional
        path to the custom directory, defaults to None. 
        If None, files will be loaded from the parent_directory/data/

    Returns
    -------
    eta : array (J, I)
        absolute vorticity
    delta : array (J, I)
        divergence
    Phi : array (J, I)
        geopotential
    U : array (J, I)
        zonal winds
    V : array (J, I)
        meridional winds
    """
    ts = str(timestamp)
    
    eta = read_pickle('eta-' + ts, custompath=custompath)
    delta = read_pickle('delta-' + ts, custompath=custompath)
    Phi = read_pickle('Phi-' + ts, custompath=custompath)
    U = read_pickle('U-' + ts, custompath=custompath)
    V = read_pickle('V-' + ts, custompath=custompath)
    
    return np.asarray(eta), np.asarray(delta), np.asarray(Phi), np.asarray(U), np.asarray(V)
