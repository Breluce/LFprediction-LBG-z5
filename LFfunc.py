#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:33:31 2022

@author: bflucero
"""

import numpy as np
import astropy.units as u
from scipy import integrate
from astropy.cosmology import Planck18
import scipy.stats as st
from scipy.interpolate import interp1d

# %% Define main functions


def m_to_M(m_app, z, *args):
    """
    Convert an AB apparent magnitude to an absolute magnitude.

    Parameters
    ----------
    m_app : apparent AB magnitude
    z : redshift

    Returns
    -------
    M_abs : the absolute AB magnitude

    """
    D_l = Planck18.luminosity_distance(z).to(u.pc)
    M_abs = m_app - 5*np.log10(D_l/(10*u.pc))

    return(M_abs)


def ABmag_to_fnu(m_ab):
    """
    Convert AB magnitude to F_nu flux.

    Parameters
    ----------
    m_ab : AB magnitude

    Returns
    -------
    fnu : F_nu flux in Jansky's

    """
    fnu = np.power(10, -((m_ab + 48.6)/2.5))
    return(fnu*u.Jy)


def UV_to_obs(M_uv, z):
    """
    Convert a UV rest-frame magnitude to an observed AB magnitude (no k-corr).

    Parameters
    ----------
    M_uv : Rest frame UV magnitude
    z : redshift

    Returns
    -------
    m_ab : observed AB magnitude

    """
    D_l = Planck18.luminosity_distance(z).to(u.pc)
    m_ab = M_uv + 5*np.log10(D_l/(10*u.pc)) - 2.5*np.log10(1+z)

    print('to kcorrect: must SUBTRACT (muv - mi) term')

    return(m_ab)


def obs_to_UV(m_ab, z, *args):
    """
    Convert an observed AB magnitude to UV rest-frame magnitude (no k-corr).

    Parameters
    ----------
    m_ab : observed AB magnitude
    z : redshift

    Returns
    -------
    M_uv : Rest-frame UV magnitude

    """
    D_l = Planck18.luminosity_distance(z).to(u.pc)
    M_uv = m_ab + 2.5*np.log10(1+z) - 5*np.log10(D_l/(10*u.pc))

    print('to kcorrect: must ADD (muv - mi) term')

    return(M_uv)


def Schechter(M, phi, M_char, alpha):
    """
    Schechther power law model of LF as a function of magnitudes.

    Parameters
    ----------
    M: range of rest-frame magnitudes
    phi: normalization factor
    M_char: characteristic magnitude value
    alpha: faint-end slope value

    Returns
    -------
    ϕ(M): Schechter form of the Luminosity Function

    """
    x = np.power(10, -0.4*(M - M_char))
    return ((np.log(10)/2.5) * phi * np.power(x, alpha + 1) * np.exp(-x))


def DPL(M, phi_dpl, M_dpl, alpha_dpl, beta):
    """
    Double Power Law model of LF as a function of magnitudes.

    Parameters
    ----------
    M : range of rest-frame magnitudes
    phi_dpl : normalization factor
    M_dpl : characteristic magnitude value
    alpha_dpl : faint-end slope of the DPL
    beta : bright-end slope of the DPL

    Returns
    -------
    ϕ_DPL(M): Double power law form of the Luminosity Function

    """
    y = np.power(10, -0.4*(M - M_dpl))
    return (0.4*np.log(10) * phi_dpl * (np.power(y, -(alpha_dpl+1))
                                        + np.power(y, -(beta+1)))**(-1))

# TODO: Complete the lensed schechter function
# def Schech_lens(M, phi, M_char, alpha, mean_mu_mult, tau, mags):
#     """
    # Lensed Schechter function.

    # Schechter power law model for LF corrected for lensing magnification
    # as a function of magnitudes.

#     Parameters
#     ----------
#     M : TYPE
#         DESCRIPTION.
#     phi : TYPE
#         DESCRIPTION.
#     M_char : TYPE
#         DESCRIPTION.
#     alpha : TYPE
#         DESCRIPTION.
#     mean_mu_mult : TYPE
#         DESCRIPTION.
#     tau : TYPE
#         DESCRIPTION.
#     mags : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     mu_demag = (1 - mean_mu_mult*tau) / (1 - tau)

#     dP1_dmu = np.piecewise(u, [0<u<2, u>2], [0, lambda u: 2*((u-1)**-3)])
#     dP2_dmu = [2*((x-1)**-3) for x in u if x>0]

#     integrand_mu = (1/mu)*

    # M_demag = M/mu_demag
    # x_demag = np.power(10, -0.4*(M_demag - M_char))
    # phi_demag = ((np.log(10)/2.5) * phi * np.power(x_demag, alpha + 1)
    #               * np.exp(-x_demag))

    # M_mu = M/mu
    # x_mu= np.power(10, -0.4*(M_mu - M_char))
    # phi_mu = ((np.log(10)/2.5) * phi * np.power(x_mu, alpha + 1)
    #           * np.exp(-x_mu))


    # phi_lens = (1-tau)*(1/mu_mean_mult)*phi_demag + tau*integrate.quad(integrand_mu, 0, np.inf)


def R_pad(R_filter, lam_filter, lam_obs, z):
    """
    Create padded response curve.

    Define a padded function for the response curve that covers the observed
    wavelength range of the SED.

    Parameters
    ----------
    R_filter : response array for the given filter X
    lam_filter : range of lambda for the given filter X
    lam_obs : observed wavelength of the SED we are observing with filter X

    Returns
    -------
    lambpad : full range of SED wavelength
    Rpad : values of response array padded with zeros

    """
    ''' filter response curve '''
    R = np.array(R_filter)
    filterlam = np.array(lam_filter*u.Angstrom)

    '''get full wavelength range of SED'''
    obslam = np.array(lam_obs*u.Angstrom)
    emlam = obslam/(1+z)

    '''create an array for the full wavelength range
    for response curve (add zeros to ends)'''
    leftpad = ((np.arange(emlam.min(), (lam_filter.min() - emlam.min()), 5)))
    rightpad = ((np.arange(lam_filter.max()+5, (obslam.max()), 5)))
    rightpad = np.append(rightpad, obslam.max())
    lampad = np.concatenate((leftpad, filterlam, rightpad))
    Rpad = np.pad(R, (len(leftpad), len(rightpad)))

    return(lampad, Rpad)


def normSED(SEDflux, lam_obs):
    """
    Normalize an SED by the maximum value.

    Parameters
    ----------
    SEDflux : flux array of the SED
    lam_obs : observed wavelengths of the SED in Angstroms

    Returns
    -------
    obslam : observed wavelengths of the SED in Angstroms
    flux : normalized SED flux

    """
    flux = SEDflux/(SEDflux.max())
    obslam = np.array(lam_obs*u.Angstrom)

    return(obslam, flux)


def Rfluxlam_obsint(lam_obs, Rfunc, SEDfunc):
    """
    Integrand for determining the AB mag of filter R over observed lambda.

    Parameters
    ----------
    lam_obs : array of observed wavelength range
    Rfunc : interpolated function of lambda for the response curve
    SEDfunc : interpolated function of lambda for the SED

    Returns
    -------
    integrand : the integrand for determining AB mag over observed lambda.

    """
    return SEDfunc(lam_obs)*Rfunc(lam_obs)*lam_obs


def Rfluxlam_emint(lam_em, Rfunc, SEDfunc, z):
    """
    Integrand for determining the AB mag of filter R over emitted lambda.

    Parameters
    ----------
    lam_em : array of emitted wavelength range
    Rfunc : interpolated function of lambda for the response curve
    SEDfunc : interpolated function of lambda for the SED
    z : redshift value(s) to shift lambda by

    Returns
    -------
    integrand : the integrand for determining AB mag over emitted lambda.

    """
    return SEDfunc(lam_em*(1+z))*Rfunc(lam_em)*lam_em


def m_ab(R_filter, lam_filter, SEDflux, lam_obs, z):
    """
    Get ABmag in a filter band with response R.

    Returns a value or array for the magnitude(s) of the specified filter at z
    to be implemented in the k-correction term.

    Parameters
    ----------
    R_filter : array for the given filter transmission values (0-1)
    lam_filter : wavelength array for the given filter X in angstroms
    SEDflux : flux array of the SED
    lam_obs : wavelength array for the given SED in angstroms
    z : redshift value

    Returns
    -------
    m : value or array of AB magnitude(s) of filter with response R

    """
    '''filter response curve'''
    R = np.array(R_filter)
    filterlam = np.array(lam_filter*u.Angstrom)

    '''SED'''
    flux = SEDflux/(SEDflux.max())
    obslam = np.array(lam_obs*u.Angstrom)
    emlam = obslam/(1+z)

    '''create an array for the full wavelength range
    for response curve (add zeros to ends)'''
    leftpad = ((np.arange(emlam.min(), (lam_filter.min() - emlam.min()), 5)))
    rightpad = ((np.arange(lam_filter.max()+5, (obslam.max()), 5)))
    rightpad = np.append(rightpad, obslam.max())
    lampad = np.concatenate((leftpad, filterlam, rightpad))
    Rpad = np.pad(R, (len(leftpad), len(rightpad)))

    '''create a function for transmission curve and SED'''
    Rfunc = interp1d(lampad, Rpad)
    SEDfunc = interp1d(obslam, flux)

    int_obs, obserr = integrate.quad(Rfluxlam_obsint, obslam.min(),
                                     obslam.max(), points=[lam_filter.min(),
                                                           lam_filter.max()],
                                     args=(Rfunc, SEDfunc))

    int_em, emerr = integrate.quad(Rfluxlam_emint, emlam.min(), emlam.max(),
                                   points=[lam_filter.min(), lam_filter.max()],
                                   args=(Rfunc, SEDfunc, z))

    m = 2.5*np.log10(int_obs/int_em)

    return(m)


def k_corr(filter1, filter2, SEDdata, z):
    """
    K-correction for transformation from filter1 to filter2 at redshift z.

    Parameters
    ----------
    filter1 : 2d-array of the response values and wavelengths for filt 1
    filter2 : 2d-array of the response values and wavelengths for filt 2
    SEDdata : 2d-array of the SED and observed wavelengths
    z : value or array of redshift(s) to determine K-correction at

    Returns
    -------
    mfilt2 - mfilt1 : the k-correction term(s) between filters at redshift(s) z

    """
    R1, Rlam1 = filter1.response, filter1.wavelength
    R2, Rlam2 = filter2.response, filter2.wavelength
    flux, lam = SEDdata.flux, SEDdata.lam

    kcorr = []

    for zval in z:
        m1 = m_ab(R1, Rlam1, flux, lam, zval)
        m2 = m_ab(R2, Rlam2, flux, lam, zval)
        k = m2 - m1
        kcorr.append(k)

    return(kcorr)

# %% define integration functions


def dV_dz(z):
    """
    Differential comoving volume element at redshift z.

    Parameters
    ----------
    z : redshift value or array of values

    Returns
    -------
    DV_dz : differential comoving volume value (no unit attached)

    """
    '''Mpc3 sr-1 per unit z'''
    dV_dz = (Planck18.differential_comoving_volume(z))

    return(dV_dz.value)


def SC_count_pred(survey_area, mag_lim, schechter_params, zgrid, kgrid,
                  dz=0.01, lowerint_lim=-50, total_skyarea=41253):
    """
    Integration over the Schechter Luminosity Function.

    Parameters
    ----------
    survey_area : area of survey field in sq deg
    mag_lim : magnitude limit for photometric band redder than dropout band
             (i.e. band we observe the LBG in)
    schechter_params : parameters for the schechter form of the LF
    zgrid : redshift grid to be integrated over
    kgrid : kcorrection values corresponding to the redshift grid
    dz : spacing of redshift grid (integration step-size)
    lowerint_lim : lower limit of integration. The default is -50.
    total_skyarea : area of the entire sky in sq deg. The default is 41253.

    Returns
    -------
    n_survey : The predicted count of LBG's in the given survey_area.

    """
    nsum = 0

    for z, k in zip(zgrid, kgrid):
        '''[Mpc^3 sr^-1]'''
        dV, dV_err = integrate.quad(dV_dz, z, z+dz)
        Mz = obs_to_UV(mag_lim, z) + k

# FIXME: move integrand definition outside integration function

        def SCintegrand(M, phi, M_char, alpha):
            return dV*Schechter(M, phi, M_char, alpha)
            # [Mpc^3 sr^-1]*[mag^-1 Mpc^-3] = [sr^-1 mag^-1]

        n, n_err = integrate.quad(SCintegrand, lowerint_lim, Mz,
                                  args=schechter_params)
        # [sr^-1 mag^-1]*[mag] = [sr^-1]
        nsum = nsum + n

        survey_sr = (survey_area * u.deg**2).to(u.sr)  # [sr]
        n_survey = nsum * (survey_sr.value)  # [number of obj in survey area]

    return(n_survey)


# FIXME: add docstring info
def DPL_count_pred(survey_area, mag_lim, DPL_params, zgrid, kgrid, dz=0.01,
                   lowerint_lim=-50, total_skyarea=41253):
    """


    Parameters
    ----------
    survey_area : TYPE
        DESCRIPTION.
    mag_lim : TYPE
        DESCRIPTION.
    DPL_params : TYPE
        DESCRIPTION.
    zgrid : TYPE
        DESCRIPTION.
    kgrid : TYPE
        DESCRIPTION.
    lowerint_lim : TYPE, optional
        DESCRIPTION. The default is -50.
    total_skyarea : TYPE, optional
        DESCRIPTION. The default is 41253.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    nsum = 0

    for z, k in zip(zgrid, kgrid):
        dV, dV_err = integrate.quad(dV_dz, z, z+dz)
        Mz = obs_to_UV(mag_lim, z) + k

# FIXME: move integrand definition outside integration function

        def DPLintegrand(M, phi, M_char, alpha, beta):
            return dV*DPL(M, phi, M_char, alpha, beta)

        n, n_err = integrate.quad(DPLintegrand, lowerint_lim, Mz,
                                  args=DPL_params)
        nsum = nsum + n

        n_survey = nsum * (survey_area/total_skyarea)

    return(n_survey)


# FIXME: add docstring info
def DPL_DPL_ctpred(survey_area, mag_lim, DPL_params_agn, DPL_params_gal, zgrid,
                   kgrid, dz=0.01, lowerint_lim=-50, total_skyarea=41253):
    """


    Parameters
    ----------
    survey_area : TYPE
        DESCRIPTION.
    mag_lim : TYPE
        DESCRIPTION.
    DPL_params_agn : TYPE
        DESCRIPTION.
    DPL_params_gal : TYPE
        DESCRIPTION.
    zgrid : TYPE
        DESCRIPTION.
    kgrid : TYPE
        DESCRIPTION.
    lowerint_lim : TYPE, optional
        DESCRIPTION. The default is -50.
    total_skyarea : TYPE, optional
        DESCRIPTION. The default is 41253.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    nsum = 0

    phi_agn, Mchar_agn, alpha_agn, beta_agn = DPL_params_agn
    phi_gal, Mchar_gal, alpha_gal, beta_gal = DPL_params_gal

    for z, k in zip(zgrid, kgrid):
        dV, dV_err = integrate.quad(dV_dz, z, z+dz)
        Mz = obs_to_UV(mag_lim, z) + k

# FIXME: move integrand definition outside integration function
        def superposition_integrand(M, phi_agn, Mchar_agn, alpha_agn, beta_agn,
                                    phi_gal, Mchar_gal, alpha_gal, beta_gal):
            return dV*(DPL(M, phi_agn, Mchar_agn, alpha_agn, beta_agn)
                       + DPL(M, phi_gal, Mchar_gal, alpha_gal, beta_gal))

        n, n_err = integrate.quad(superposition_integrand, lowerint_lim, Mz,
                                  args=(*DPL_params_agn, *DPL_params_gal))
        nsum = nsum + n

        n_survey = nsum * (survey_area/total_skyarea)

    return(n_survey)


# FIXME: add docstring info
def DPL_S_ctpred(survey_area, mag_lim, DPL_params, schechter_params, zgrid,
                 kgrid, dz=0.01, lowerint_lim=-50, total_skyarea=41253):
    """


    Parameters
    ----------
    survey_area : TYPE
        DESCRIPTION.
    mag_lim : TYPE
        DESCRIPTION.
    DPL_params : TYPE
        DESCRIPTION.
    schechter_params : TYPE
        DESCRIPTION.
    zgrid : TYPE
        DESCRIPTION.
    kgrid : TYPE
        DESCRIPTION.
    lowerint_lim : TYPE, optional
        DESCRIPTION. The default is -50.
    total_skyarea : TYPE, optional
        DESCRIPTION. The default is 41253.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    nsum = 0

    phi_agn, Mchar_agn, alpha_agn, beta_agn = DPL_params
    phi_gal, Mchar_gal, alpha_gal = schechter_params

    for z, k in zip(zgrid, kgrid):
        dV, dV_err = integrate.quad(dV_dz, z, z+dz)
        Mz = obs_to_UV(mag_lim, z) + k

        def superposition_integrand(M, phi_agn, Mchar_agn, alpha_agn, beta_agn,
                                    phi_gal, Mchar_gal, alpha_gal):
            return dV*(DPL(M, phi_agn, Mchar_agn, alpha_agn, beta_agn)
                       + Schechter(M, phi_gal, Mchar_gal, alpha_gal))

        n, n_err = integrate.quad(superposition_integrand, lowerint_lim, Mz,
                                  args=(*DPL_params, *schechter_params))
        nsum = nsum + n

        n_survey = nsum * (survey_area/total_skyarea)

    return(n_survey)

# %% random draw functions


# FIXME: add docstring info
def SC_rand_draw(data, plusmin, logpm):
    """


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    plusmin : TYPE
        DESCRIPTION.
    logpm : TYPE
        DESCRIPTION.

    Returns
    -------
    phi : TYPE
        DESCRIPTION.
    M_char : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.

    """
    phi0, M_char0, alpha0 = data
    log_phi0 = np.log10(phi0)

    if isinstance(logpm, list) is False:
        phi_err = np.mean((plusmin[0][0], plusmin[0][1]))
        phi = st.norm.rvs(loc=phi0, scale=phi_err)

    if isinstance(logpm, list) is True:
        logphi_err = np.mean(logpm)
        log_phi = st.norm.rvs(loc=log_phi0, scale=logphi_err)
        phi = np.power(10, log_phi)

    M_char_err = np.mean((plusmin[1][0], plusmin[1][1]))
    alpha_err = np.mean((plusmin[2][0], plusmin[2][1]))

    M_char = st.norm.rvs(loc=M_char0, scale=M_char_err)
    alpha = st.norm.rvs(loc=alpha0, scale=alpha_err)

    # print(logpm, phi)

    return phi, M_char, alpha


# FIXME: add docstring info
def DPL_rand_draw(data, plusmin, logpm):
    """


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    plusmin : TYPE
        DESCRIPTION.
    logpm : TYPE
        DESCRIPTION.

    Returns
    -------
    phi : TYPE
        DESCRIPTION.
    M_char : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.

    """
    phi0, M_char0, alpha0, beta0 = data
    log_phi0 = np.log10(phi0)

    if isinstance(logpm, list) is False:
        phi_err = np.mean((plusmin[0][0], plusmin[0][1]))
        phi = st.norm.rvs(loc=phi0, scale=phi_err)

    if isinstance(logpm, list) is True:
        logphi_err = np.mean(logpm)
        log_phi = st.norm.rvs(loc=log_phi0, scale=logphi_err)
        phi = np.power(10, log_phi)

    M_char_err = np.mean((plusmin[1][0], plusmin[1][1]))
    alpha_err = np.mean((plusmin[2][0], plusmin[2][1]))
    beta_err = np.mean((plusmin[3][0], plusmin[3][1]))

    M_char = st.norm.rvs(loc=M_char0, scale=M_char_err)
    alpha = st.norm.rvs(loc=alpha0, scale=alpha_err)
    beta = st.norm.rvs(loc=beta0, scale=beta_err)

    return phi, M_char, alpha, beta
