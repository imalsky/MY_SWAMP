# -*- coding: utf-8 -*-
"""
This module contains the functions that perform the explicit timestepping scheme
as described in Hack and Jakob (1992).

This is a corrected JAX version that matches the original numpy implementation exactly.
"""
from __future__ import annotations

import jax.numpy as jnp

from . import spectral_transform as st
from . import filters


def phi_timestep(
    etam0, etam1, deltam0, deltam1, Phim0, Phim1,
    I, J, M, N, Am, Bm, Cm, Dm, Em, Fm, Gm, Um, Vm,
    Pmn, Hmn, w, tstepcoeff1, tstepcoeff2, mJarray, narray,
    PhiFm, dt, a, Phibar, taurad, taudrag, forcflag, diffflag,
    sigma, sigmaPhi, test, t
):
    """
    Explicit time-stepping for geopotential Phi.
    """
    # Component 1: Forward Legendre of Phim0
    Phicomp1 = st.fwd_leg(Phim0, J, M, N, Pmn, w)
    
    # Component 2: tstepcoeff1 * (1j)*mJarray * Cm, then forward Legendre
    Phicomp2prep = tstepcoeff1 * (1j) * mJarray * Cm
    Phicomp2 = st.fwd_leg(Phicomp2prep, J, M, N, Pmn, w)
    
    # Component 3: tstepcoeff1 * Dm, then forward Legendre with Hmn
    Phicomp3prep = tstepcoeff1 * Dm
    Phicomp3 = st.fwd_leg(Phicomp3prep, J, M, N, Hmn, w)
    
    # Component 4: 2*dt*Phibar times forward Legendre of deltam1
    Phicomp4 = 2 * dt * Phibar * st.fwd_leg(deltam1, J, M, N, Pmn, w)
    
    # Combine
    Phimntstep = Phicomp1 - Phicomp2 + Phicomp3 - Phicomp4
    
    # Forcing
    if forcflag:
        Phiforcing = st.fwd_leg(2 * dt * PhiFm, J, M, N, Pmn, w)
        Phimntstep = Phimntstep + Phiforcing
    
    # Diffusion filter
    if diffflag:
        Phimntstep = filters.diffusion(Phimntstep, sigmaPhi)
    
    # Transform back to physical space
    newPhimtstep = st.invrs_leg(Phimntstep, I, J, M, N, Pmn)
    newPhitstep = st.invrs_fft(newPhimtstep, I)
    
    return Phimntstep, newPhitstep


def delta_timestep(
    etam0, etam1, deltam0, deltam1, Phim0, Phim1,
    I, J, M, N, Am, Bm, Cm, Dm, Em, Fm, Gm, Um, Vm,
    Pmn, Hmn, w, tstepcoeff1, tstepcoeff2, mJarray, narray,
    PhiFm, dt, a, Phibar, taurad, taudrag, forcflag, diffflag,
    sigma, sigmaPhi, test, t
):
    """
    Explicit time-stepping for divergence delta.
    CORRECTED to match original numpy implementation.
    """
    # Component 1: Forward Legendre of deltam0
    deltacomp1 = st.fwd_leg(deltam0, J, M, N, Pmn, w)
    
    # Component 2: tstepcoeff1 * (1j)*mJarray * Bm
    deltacomp2prep = tstepcoeff1 * (1j) * mJarray * Bm
    deltacomp2 = st.fwd_leg(deltacomp2prep, J, M, N, Pmn, w)
    
    # Component 3: tstepcoeff1 * Am with Hmn
    deltacomp3prep = tstepcoeff1 * Am
    deltacomp3 = st.fwd_leg(deltacomp3prep, J, M, N, Hmn, w)
    
    # Component 4: tstepcoeff2 * (Phim1 + Em)
    deltacomp4prep = tstepcoeff2 * (Phim1 + Em)
    deltacomp4 = st.fwd_leg(deltacomp4prep, J, M, N, Pmn, w)
    deltacomp4 = narray * deltacomp4
    
    # NOTE: Original code has this commented line showing full formula:
    # deltamntstep = deltacomp1 + deltacomp2 + deltacomp3 + deltacomp4
    # But the actual implementation only uses:
    deltamntstep = deltacomp1  # + deltacomp2 + deltacomp4 (commented in original)
    
    # Forcing terms
    if forcflag:
        # deltaf1
        deltaf1prep = tstepcoeff1 * (1j) * mJarray * Um / taudrag
        deltaf1 = st.fwd_leg(deltaf1prep, J, M, N, Pmn, w)
        
        # deltaf2
        deltaf2prep = tstepcoeff1 * Vm / taudrag
        deltaf2 = st.fwd_leg(deltaf2prep, J, M, N, Hmn, w)
        
        # deltaf3
        deltaf3prep = tstepcoeff1 * (1j) * mJarray * Fm
        deltaf3 = st.fwd_leg(deltaf3prep, J, M, N, Pmn, w)
        
        # deltaf4
        deltaf4prep = tstepcoeff1 * Gm
        deltaf4 = st.fwd_leg(deltaf4prep, J, M, N, Hmn, w)
        
        deltaforcing = -deltaf1 + deltaf2 + deltaf3 - deltaf4
        deltamntstep = deltamntstep + deltaforcing
    
    # Diffusion filter
    if diffflag:
        deltamntstep = filters.diffusion(deltamntstep, sigma)
    
    # Transform back
    newdeltamtstep = st.invrs_leg(deltamntstep, I, J, M, N, Pmn)
    newdeltatstep = st.invrs_fft(newdeltamtstep, I)
    
    return deltamntstep, newdeltatstep


def eta_timestep(
    etam0, etam1, deltam0, deltam1, Phim0, Phim1,
    I, J, M, N, Am, Bm, Cm, Dm, Em, Fm, Gm, Um, Vm,
    Pmn, Hmn, w, tstepcoeff1, tstepcoeff2, mJarray, narray,
    PhiFm, dt, a, Phibar, taurad, taudrag, forcflag, diffflag,
    sigma, sigmaPhi, test, t
):
    """
    Explicit time-stepping for absolute vorticity eta.
    CORRECTED to match original numpy implementation.
    """
    # Component 1: Forward Legendre of etam0
    etacomp1 = st.fwd_leg(etam0, J, M, N, Pmn, w)
    
    # Component 2: tstepcoeff1 * (1j)*mJarray * Am
    etacomp2prep = tstepcoeff1 * (1j) * mJarray * Am
    etacomp2 = st.fwd_leg(etacomp2prep, J, M, N, Pmn, w)
    
    # Component 3: tstepcoeff1 * Bm with Hmn
    etacomp3prep = tstepcoeff1 * Bm
    etacomp3 = st.fwd_leg(etacomp3prep, J, M, N, Hmn, w)
    
    etamntstep = etacomp1 - etacomp2 + etacomp3
    
    # Forcing terms
    if forcflag:
        # etaf1
        etaf1prep = tstepcoeff1 * (1j) * mJarray * Vm / taudrag
        etaf1 = st.fwd_leg(etaf1prep, J, M, N, Pmn, w)
        
        # etaf2
        etaf2prep = tstepcoeff1 * Um / taudrag
        etaf2 = st.fwd_leg(etaf2prep, J, M, N, Hmn, w)
        
        # etaf3
        etaf3prep = tstepcoeff1 * (1j) * mJarray * Gm
        etaf3 = st.fwd_leg(etaf3prep, J, M, N, Pmn, w)
        
        # etaf4
        etaf4prep = tstepcoeff1 * Fm
        etaf4 = st.fwd_leg(etaf4prep, J, M, N, Hmn, w)
        
        etaforcing = -etaf1 + etaf2 + etaf3 + etaf4
        etamntstep = etamntstep + etaforcing
    
    # Diffusion filter
    if diffflag:
        etamntstep = filters.diffusion(etamntstep, sigma)
    
    # Transform back
    newetamtstep = st.invrs_leg(etamntstep, I, J, M, N, Pmn)
    newetatstep = st.invrs_fft(newetamtstep, I)
    
    return etamntstep, newetatstep
