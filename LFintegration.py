#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
from scipy import integrate 
import scipy.optimize as so
from astropy.cosmology import Planck18
from astropy.units import Quantity
import scipy.stats as st
from itertools import chain
from tqdm import tqdm
import sys
from speclite import filters
from scipy.interpolate import interp1d

#%% Filter response curves + SED templates
fig, ax = plt.subplots(1,1)

with_atm = filters.load_filters('decamDR1-*')
filters.plot_filters(with_atm)

R_i = with_atm[2].response
lam_i = with_atm[2].wavelength

# without_atm = speclite.filters.load_filters('decamDR1noatm-*')
# speclite.filters.plot_filters(without_atm)

z=5
seddata = pd.read_csv('/Users/bflucero/Desktop/research/SED files/galsedaNGC4670.txt', delimiter = "\s+", comment = '#', usecols = [0,1], names=['Restframe Wavelength (Angstrom)', 'Flux per unit wavelength (ergs/s/cm^2/Angstrom)'])

plt.plot(seddata['Restframe Wavelength (Angstrom)']*(1+z), seddata['Flux per unit wavelength (ergs/s/cm^2/Angstrom)']/seddata['Flux per unit wavelength (ergs/s/cm^2/Angstrom)'].max(), label = 'observed SED (z=5)', c = 'black', alpha = 0.8)
plt.plot(seddata['Restframe Wavelength (Angstrom)'], seddata['Flux per unit wavelength (ergs/s/cm^2/Angstrom)']/seddata['Flux per unit wavelength (ergs/s/cm^2/Angstrom)'].max(), label = 'rest frame SED', c = 'grey', alpha = 0.5)
plt.legend()

#%%define k-correction

SED_normf = seddata['Flux per unit wavelength (ergs/s/cm^2/Angstrom)']/seddata['Flux per unit wavelength (ergs/s/cm^2/Angstrom)'].max()
SED_lam = seddata['Restframe Wavelength (Angstrom)']

#color-correction term, i.e. m_UV - m_AB,i
def color_corr(filter_response, Rlam_range, SEDflux, lam_obs, z):
    
    #filter response curve
    R = np.array(filter_response)
    filterlam = np.array(Rlam_range*u.Angstrom)
    
    #SED
    flux = SEDflux.values*( (u.erg) / (u.s*(u.cm**2)*u.Angstrom) )
    obslam = np.array(lam_obs*u.Angstrom)
    emlam = obslam/(1+z)
    
    #create an array for the full wavelength range for response curve (add zeros to ends)
    leftpad = ((np.arange(emlam.min(), (lam_i.min() - emlam.min()), 5)))
    rightpad = ((np.arange(lam_i.max()+5, (obslam.max()), 5)))
    rightpad = np.append(rightpad, obslam.max())
    filterlam = np.concatenate((leftpad, filterlam, rightpad))
    Rpad = np.pad(R, (len(leftpad), len(rightpad)))
    
    #create a function for transmission curve and SED
    Rfunc = interp1d(filterlam, Rpad)
    SEDfunc = interp1d(obslam, flux)
    
    def Rfluxlam_obsint(lam_obs):
        return SEDfunc(lam_obs)*Rfunc(lam_obs)*lam_obs
    
    int_obs, intobserr = integrate.quad(Rfluxlam_obsint, obslam.min(), obslam.max(), points = [lam_i.min(), lam_i.max()])
    
    def Rfluxlam_emint(lam_em, *kwargs):
        return SEDfunc(lam_em*(1+z))*Rfunc(lam_em)*lam_em
    
    int_em, intemerr = integrate.quad(Rfluxlam_emint, emlam.min(), emlam.max(), points = [lam_i.min(), lam_i.max()])
    
    #C correction term
    C = 2.5*np.log10(int_obs/int_em)
    
    return(C, obslam, Rfluxlam_obsint(obslam), emlam, Rfluxlam_emint(emlam))

C_corr,x,y,x2,y2 = (color_corr(R_i, lam_i, SED_normf, SED_lam, 5))

plt.plot(x,y, label = 'RxSEDxlam at obslam')
plt.plot(x2, y2, label = 'RxSEDxlam at emlam')
# plt.plot(x, SED_normf, label = 'obs flux')
# plt.plot(x2, SED_normf, label = 'em flux')
plt.legend()

    
 #    # TODO: compare to Hoggs paper / Oke paper => is integrating over  f_lam_em w lambda_obs equivalent to int f_lam_obs w lambda_em
 #        # NO -> why... and what is the correct way
 #        # i think over lam obs is the correct way because we trying to analyze flux in a specific band not in the redshifted wavelength range
 #    # TODO: i don't understand why for photometry flam(lamobs) does NOT equal flam(lamem*(1+z))

#%% define main functions

#convert abs AB mag to UV restframe

def obs_to_UV(m_ab, z):
    D_l = Planck18.luminosity_distance(z).to(u.pc)
    M_uv = m_ab + 2.5*np.log10(1+z) - 5*np.log10(D_l/(10*u.pc))
    return(M_uv)

def ABmag_to_fnu(m_ab):
    fnu = np.power(10, -((m_ab + 48.6)/2.5))
    return(fnu*u.Jy) #Jy

def UV_to_obs(M_uv, z):
    D_l = Planck18.luminosity_distance(z).to(u.pc)
    m_ab = M_uv - 2.5*np.log10(1+z) + 5*np.log10(D_l/(10*u.pc))
    return(m_ab)

#general schechter function:
    
def Schechter(M, phi, M_char, alpha):
    x = np.power(10, -0.4*(M - M_char))
    return ((np.log(10)/2.5) * phi * np.power(x, alpha + 1) * np.exp(-x))
 
#double power law function:

def DPL(M, phi_dpl, M_dpl, alpha_dpl, beta):
    y = np.power(10, -0.4*(M - M_dpl))
    return (0.4*np.log(10) * phi_dpl * (np.power(y, -(alpha_dpl+1)) + np.power(y, -(beta+1)))**(-1))


#%% SHECHTER PARAMETERS

paramsDF = pd.DataFrame(index=('phi_star', 'phi_plusmin', 'logphi_pm', 'M_star', 'M_plusmin', 'alpha', 'alpha_plusmin', 'z', 'M_UV_range'))

#Ono(2018)
paramsDF['Ono (2018)'] = [1.06e-3, [0.13*10**-3, 0.11*10**-3], np.nan, -20.96, [0.06, 0.05], -1.60, [0.06, 0.11], 5, [-26, -14]] #[phi* (mpc-3) , M* (mag)]

#Bouwens (2021) - this includes the same dataset used to derive the 2015 parameters; just updated w more sources
paramsDF['Bouwens (2021a)'] = [0.79e-3, [0.16e-3, 0.13e-3], np.nan, -21.20, [0.11, 0.11], -1.74, [0.06, 0.06], 4.9, [-21.85, -17.60]] #[phi* (mpc-3) , M* (mag)]

########################## NOTE the phi* and M* values have different units than the rest of the params ##################################################################
#Harikane 2021
def pw10_plus(upper, val):
    main = 10**val
    upper = 10**(val + upper)
    return(upper-main, upper)

def pw10_min(lower, val):
    main = 10**val
    lower = 10**(val - lower)
    return(main-lower, lower)

paramsDF['Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)'] = [10**(-3.16), [pw10_plus(0.03, -3.16)[0], pw10_min(0.03, -3.16)[0]], [0.03, 0.03], -21.09, [0.04,0.03], -1.76, [0.04, 0.03], 5, [-29, -14]] #[phi* (mag) -NOTE: orig given in log(phi*), M* (mpc-3)]
paramsDF['Harikane (2021) Galaxy LF (Schechter)'] = [10**(-3.10), [pw10_plus(0.06, -3.10)[0], pw10_min(0.06, -3.10)[0]], [0.06, 0.06], -21.04, [0.08, 0.07], -1.76, [0.05, 0.05], 5,[-29, -14]]

#%% DOUBLE POWER LAW PARAMETERS 
DPLparams = pd.DataFrame(columns = [], index = ('phi_dpl', 'phi_dpl_pm', 'logphi_pm', 'M_dpl', 'M_dpl_pm', 'alpha_dpl', 'alpha_dpl_pm', 'beta', 'beta_pm', 'z', 'M_UV_range'))

#Ono 2018
DPLparams['Ono (2018) DPL'] = [0.36e-3, [0.05e-3, 0.05e-3], np.nan, -21.44, [0.07, 0.07], -1.88, [0.05, 0.04], -5.07, [0.17, 0.18], 5,[-26, -14]] #[phi* (mpc-3) , M* (mag)]
    
########################## NOTE the phi* and M* values have different units than the rest of the params ##################################################################
#Harikane 2021
DPLparams['Harikane (2021) AGN component (AGN_DPL + GAL_Schechter)'] = [10**(-8.71), [pw10_plus(0.89, -8.71)[0], pw10_min(0.67, -8.71)[0]], [0.89, 0.67], -27.67, [1.47, 0.88], -2.27, [0.48, 0.22], -5.92, [0.66, 1.14], 5, [-29, -14]]
DPLparams['Harikane (2021) AGN component (AGN_DPL + GAL_DPL)'] = [10**(-8.35), [pw10_plus(0.47, -8.35)[0], pw10_min(0.18, -8.35)[0]], [0.47, 0.18], -27.32, [0.76, 0.26], -1.92, [0.31,0.17], -4.77, [0.61,0.62], 5, [-29, -24]]
DPLparams['Harikane (2021) galaxy component (AGN_DPL + GAL_DPL)'] = [10**(-3.63), [pw10_plus(0.04, -3.63)[0], pw10_min(0.03, -3.63)[0]],[0.04, 0.03], -21.54, [0.04, 0.04], -2.01, [0.04,0.03], -4.91, [0.08,0.08], 5, [-22, -14]]
DPLparams['Harikane (2021) Galaxy LF (DPL)'] = [10**(-3.48), [pw10_plus(0.07, -3.48)[0], pw10_min(0.06, -3.48)[0]], [00.07, 0.06], -21.39, [0.09, 0.07], -1.94, [0.04, 0.04], -4.96, [0.21, 0.18], 5, [-22, -14]]
#[10**(-3.48), [10**0.07, 10**0.06], -21.39, [0.09, 0.07], -1.94, [0.04, 0.04], -4.96, [0.21, 0.18], 5] #[phi* (mag) -NOTE: orig given in log(phi*), M* (mpc-3)]


#%% LF Plots

fig, ax = plt.subplots(1,1)
# onomags = np.linspace(-26, -14, num = 500) #intro of ono2018
# bouwensmags = np.linspace(-21.85,-17.60) #table 4 of bouwens2021a
# harikanemags = np.linspace(-29, -14) #various sections of harikane2021
mags = np.linspace(obs_to_UV(21.5, 5.5), obs_to_UV(26.5, 5.5), num = 100)

for i in paramsDF: 
    params = [paramsDF[i].phi_star, paramsDF[i].M_star, paramsDF[i].alpha]
    
    plus = [paramsDF[i].phi_plusmin[0], paramsDF[i].M_plusmin[0], paramsDF[i].alpha_plusmin[0]]
    uplim = np.add(params, plus)
    minus = [paramsDF[i].phi_plusmin[1], paramsDF[i].M_plusmin[1], paramsDF[i].alpha_plusmin[1]]
    lowlim = np.subtract(params, minus)

    if paramsDF[i].name != 'Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)':
        plot = plt.plot(mags, (Schechter(mags, *params)), label = '{name}'.format(name = paramsDF[i].name), alpha=0.45)
        ax.fill_between(mags, (Schechter(mags, *params)), (Schechter(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(paramsDF[i].name), color = plot[0].get_color())
        ax.fill_between(mags, (Schechter(mags, *params)), (Schechter(mags, *lowlim)), alpha = 0.15)
    if paramsDF[i].name == 'Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)':
        plot = plt.plot(mags, (Schechter(mags, *params)), label = '{name}'.format(name = paramsDF[i].name), linestyle = ':', color = 'cyan', zorder=20)
        ax.fill_between(mags, (Schechter(mags, *params)), (Schechter(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(paramsDF[i].name), color = plot[0].get_color())
        ax.fill_between(mags, (Schechter(mags, *params)), (Schechter(mags, *lowlim)), alpha = 0.15)
        
#LF plots w DPL Harikane
        
for i in DPLparams: 

    params = [DPLparams[i].phi_dpl, DPLparams[i].M_dpl, DPLparams[i].alpha_dpl, DPLparams[i].beta]
    
    plus = [DPLparams[i].phi_dpl_pm[0], DPLparams[i].M_dpl_pm[0], DPLparams[i].alpha_dpl_pm[0], DPLparams[i].beta_pm[0]]
    uplim = np.add(params, plus)
    minus = [DPLparams[i].phi_dpl_pm[1], DPLparams[i].M_dpl_pm[1], DPLparams[i].alpha_dpl_pm[1], DPLparams[i].beta_pm[1]]
    lowlim = np.subtract(params, minus)
    
    if DPLparams[i].name == 'Harikane (2021) AGN component (AGN_DPL + GAL_Schechter)': 
        # mags = np.linspace(-26, -24, num = 100)
        plot2 = plt.plot(mags, DPL(mags, *params), label = '{name}'.format(name = DPLparams[i].name), color = 'cyan', linestyle = ':', zorder=20)
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(DPLparams[i].name), color = plot2[0].get_color())
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *lowlim)), alpha = 0.15, color = plot2[0].get_color())
    if DPLparams[i].name == 'Harikane (2021) AGN component (AGN_DPL + GAL_DPL)': 
        # mags = np.linspace(-26, -24, num = 100)
        plot2 = plt.plot(mags, (DPL(mags, *params)), label = '{name}'.format(name = DPLparams[i].name), color = 'brown', linestyle = ':', zorder=20)
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(DPLparams[i].name), color = plot2[0].get_color())
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *lowlim)), alpha = 0.15, color = plot2[0].get_color())
    if DPLparams[i].name == 'Harikane (2021) galaxy component (AGN_DPL + GAL_DPL)':
        # mags = np.linspace(-22, -18, num = 100)
        plot2 = plt.plot(mags, (DPL(mags, *params)), label = '{name}'.format(name = DPLparams[i].name), color = 'brown', linestyle = ':', zorder=20)
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(DPLparams[i].name), color = plot2[0].get_color())
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *lowlim)), alpha = 0.15, color = plot2[0].get_color())
    if DPLparams[i].name == 'Ono (2018) DPL':
        # mags = np.linspace(-26, -18, num = 500) #cross check with the papers    
        plot2 = plt.plot(mags, (DPL(mags, *params)), label = '{name}'.format(name = DPLparams[i].name), alpha=0.5)
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(DPLparams[i].name), color = plot2[0].get_color())
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *lowlim)), alpha = 0.15, color = plot2[0].get_color())
    else:
        plot2 = plt.plot(mags, (DPL(mags, *params)), label = '{name}'.format(name = DPLparams[i].name))
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(DPLparams[i].name), color = plot2[0].get_color())
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *lowlim)), alpha = 0.15, color = plot2[0].get_color())
        
    secx = ax.secondary_xaxis('top', functions=(lambda x: UV_to_obs(x, 5.5), lambda x: UV_to_obs(x, 5.5)))
    secx.set(xlabel='$m_{obs}$ (AB)')
    ax.annotate("", xy=(obs_to_UV(24, 5.5), 10**-5), xytext=(-24.5, 10**-5), arrowprops=dict(arrowstyle="->"), zorder=50)

#plot the superposition of DPL/DPL and DPL/Schech for AGN and galaxy LF
agn2DPL = 'Harikane (2021) AGN component (AGN_DPL + GAL_DPL)'
gal2DPL = 'Harikane (2021) galaxy component (AGN_DPL + GAL_DPL)'
agnDPL_s = 'Harikane (2021) AGN component (AGN_DPL + GAL_Schechter)'
galDPL_s = 'Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)'

spmags = np.linspace(-26, -18, num=100)
sp2DPL = DPL(spmags, DPLparams[agn2DPL].phi_dpl, DPLparams[agn2DPL].M_dpl, DPLparams[agn2DPL].alpha_dpl, DPLparams[agn2DPL].beta) + DPL(spmags, DPLparams[gal2DPL].phi_dpl, DPLparams[gal2DPL].M_dpl, DPLparams[gal2DPL].alpha_dpl, DPLparams[gal2DPL].beta)
spDPL_s = DPL(spmags, DPLparams[agnDPL_s].phi_dpl, DPLparams[agnDPL_s].M_dpl, DPLparams[agnDPL_s].alpha_dpl, DPLparams[agnDPL_s].beta) + Schechter(spmags, paramsDF[galDPL_s].phi_star, paramsDF[galDPL_s].M_star, paramsDF[galDPL_s].alpha)

plt.plot(mags, (sp2DPL), label = 'AGN DPL + GAL DPL', color = 'brown', alpha=0.45, zorder=20)#, linestyle = ':')
plt.plot(mags, (spDPL_s), label = 'AGN DPL + GAL Schechter', color = 'cyan', alpha=0.45, zorder=20)#, linestyle = ':')
plt.xlabel('$M_{UV}$') 
plt.ylabel('$\phi(M)$ (mag$^{-1}$ Mpc$^{-3}$)')
plt.yscale('log')
ax.annotate("AGN-dominated", xy=(obs_to_UV(21.8, 5.5), 10**(-3)), fontsize = 'x-small')
ax.annotate("LBG-dominated", xy=(obs_to_UV(24.8, 5.5), 10**(-3)), fontsize = 'x-small')
ax.axvline(x = obs_to_UV(24.0, 5.5), linestyle = "--", c = 'black', linewidth = 0.9, label = 'DES i-band 95% completeness \n UPPER INT LIM')
ax.axvline(x = -24, linestyle = "--", c = 'blue', linewidth = 0.9, label = 'upper lim on AGN-dominated LF \n 0% galaxy fraction for z ~ 4-7 (Harikane 2021)') #fig 5 in literature
ax.axvline(x = -22, linestyle = "--", c = 'red', linewidth = 0.9, label = 'lower lim for gal-dominated LF \n 100% galaxy fraction for z ~ 4-7 (Harikane 2021)') #fig 5 in literature
#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
#specify order of items in legend
order = [14, 15, 16, 0,4,1,3,11,2,5,13,9,7,12]
#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize = 'xx-small', loc = 4, ncol=1)
ax.legend(fontsize = 'x-small', loc = 4)

#%%
fig, ax = plt.subplots()

plt.plot(spmags, (sp2DPL), label = 'AGN DPL + GAL DPL', color = 'brown', alpha=0.45)#, linestyle = ':')
plt.plot(spmags, (spDPL_s), label = 'AGN DPL + GAL Schechter', color = 'cyan', alpha=0.45)#, linestyle = ':')
plt.xlabel('$M_{UV}$') 
plt.ylabel('$\phi(M)$ (mag$^{-1}$ Mpc$^{-3}$)')
plt.yscale('log')
ax.annotate("AGN-dominated", xy=(-25.5, 10e-4), fontsize = 'small')
ax.annotate("LBG-dominated", xy=(-21, 10e-4), fontsize = 'small')
ax.axvline(x = obs_to_UV(24.0, 5.5), linestyle = "--", c = 'black', linewidth = 0.9, label = 'DES i-band 95% completeness \n UPPER INT LIM')
ax.axvline(x = -24, linestyle = "--", c = 'blue', linewidth = 0.9, label = 'upper lim on AGN-dominated LF \n 0% galaxy fraction for z ~ 4-7 (Harikane 2021)') #fig 5 in literature
ax.axvline(x = -22, linestyle = "--", c = 'red', linewidth = 0.9, label = 'lower lim for gal-dominated LF \n 100% galaxy fraction for z ~ 4-7 (Harikane 2021)') #fig 5 in literature
plt.legend()

#%%
fig, ax = plt.subplots()

for i in paramsDF: 
    params = [paramsDF[i].phi_star, paramsDF[i].M_star, paramsDF[i].alpha]
    
    plus = [paramsDF[i].phi_plusmin[0], paramsDF[i].M_plusmin[0], paramsDF[i].alpha_plusmin[0]]
    uplim = np.add(params, plus)
    minus = [paramsDF[i].phi_plusmin[1], paramsDF[i].M_plusmin[1], paramsDF[i].alpha_plusmin[1]]
    lowlim = np.subtract(params, minus)
    mags = np.linspace(-26, -18, num = 100)

    if paramsDF[i].name == 'Harikane (2021) Galaxy LF (Schechter)':
        mags = np.linspace(-22, -18, num = 100)
        plot = plt.plot(mags, (Schechter(mags, *params)), label = '{name}'.format(name = paramsDF[i].name))
        ax.fill_between(mags, (Schechter(mags, *params)), (Schechter(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(paramsDF[i].name), color = plot[0].get_color())
        ax.fill_between(mags, (Schechter(mags, *params)), (Schechter(mags, *lowlim)), alpha = 0.15)

for i in DPLparams: 

    params = [DPLparams[i].phi_dpl, DPLparams[i].M_dpl, DPLparams[i].alpha_dpl, DPLparams[i].beta]
    
    plus = [DPLparams[i].phi_dpl_pm[0], DPLparams[i].M_dpl_pm[0], DPLparams[i].alpha_dpl_pm[0], DPLparams[i].beta_pm[0]]
    uplim = np.add(params, plus)
    minus = [DPLparams[i].phi_dpl_pm[1], DPLparams[i].M_dpl_pm[1], DPLparams[i].alpha_dpl_pm[1], DPLparams[i].beta_pm[1]]
    lowlim = np.subtract(params, minus)
    
    if DPLparams[i].name == 'Harikane (2021) Galaxy LF (DPL)':
        # mags = np.linspace(-22, -18, num = 100)
        plot2 = plt.plot(mags, (DPL(mags, *params)), label = '{name}'.format(name = DPLparams[i].name))
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(DPLparams[i].name), color = plot2[0].get_color())
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *lowlim)), alpha = 0.15, color = plot2[0].get_color())
    
    secx = ax.secondary_xaxis('top', functions=(lambda x: UV_to_obs(x, 5.5), lambda x: UV_to_obs(x, 5.5)))
    secx.set(xlabel='$m_{obs}$ (AB)')
    
plt.xlabel('$M_{UV}$') 
plt.ylabel('$\phi(M)$ (mag$^{-1}$ Mpc$^{-3}$)')
plt.yscale('log')
plt.legend()

#%%
#LF plots w DPL Ono

fig, ax = plt.subplots(1,1)

for i in DPLparams:
    if DPLparams[i].name == 'Ono (2018) DPL':
        params = [DPLparams[i].phi_dpl, DPLparams[i].M_dpl, DPLparams[i].alpha_dpl, DPLparams[i].beta]
        
        plus = [DPLparams[i].phi_dpl_pm[0], DPLparams[i].M_dpl_pm[0], DPLparams[i].alpha_dpl_pm[0], DPLparams[i].beta_pm[0]]
        uplim = np.add(params, plus)
        minus = [DPLparams[i].phi_dpl_pm[1], DPLparams[i].M_dpl_pm[1], DPLparams[i].alpha_dpl_pm[1], DPLparams[i].beta_pm[1]]
        lowlim = np.subtract(params, minus)
        
        mags = np.linspace(-25, -15, num = 500) #cross check with the papers    
        
        plot2 = plt.plot(mags, (DPL(mags, *params)), label = '{name}'.format(name = DPLparams[i].name))
        secx = ax.secondary_xaxis('top', functions=(lambda x: UV_to_obs(x, 5.5), lambda x: UV_to_obs(x, 5.5)))
        secx.set(xlabel='$m_{obs}$ (AB)')
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *uplim)), alpha = 0.15, label = 'error {}'.format(DPLparams[i].name), color = plot2[0].get_color())
        ax.fill_between(mags, (DPL(mags, *params)), (DPL(mags, *lowlim)), alpha = 0.15, color = plot2[0].get_color())

plt.xlabel('$M_{UV}$') 
plt.ylabel('$\phi(M)$ (mag$^{-1}$ Mpc$^{-3}$)')
plt.yscale('log')
ax.axvline(x = obs_to_UV(24.0, 5.5), linestyle = "--", c = 'black', linewidth = 0.9, label = 'DES i-band 95% completeness \n UPPER INT LIM')
ax.legend()

#%% define integration parameters and DES detection limits

UVrcomplimit = obs_to_UV(24.3, 4) #use for z = 4 gband droputs
UVicomplimit = obs_to_UV(24.0, 5) #use for z = 5 rband dropouts

UVr10sig = obs_to_UV(24.3, 4) #use for z = 4 gband droputs
UVi10sig = obs_to_UV(23.5, 5) #use for z = 5 rband droputs

DEStile_skyarea = Quantity(0.534, "deg2")
DES_skyarea = Quantity(5000, "deg2")
# GDS_skyarea = Quantity(0.1736, "deg2")
total_skyarea = Quantity(41253, "deg2")    

#%% define functions for luminosity dist and differential comoving vol

def d_l(z):
    return(Planck18.luminosity_distance(z)/u.Mpc) #luminosity distance (Mpc)
    
def dV_dz(z, sky_area):
    dV_dz_area = (Planck18.differential_comoving_volume(z)*sky_area).to(u.Mpc**3)
    return(dV_dz_area/(u.Mpc**3))  #differential volume element in given area (Mpc3)


#%% random draw functions

def schech_rand_draw(data, plusmin, logpm):
    
    phi0, M_char0, alpha0 = data
    log_phi0 = np.log10(phi0)
    
    if isinstance(logpm, list) == False:
        phi_err = np.mean((plusmin[0][0], plusmin[0][1]))
        phi = st.norm.rvs(loc = phi0, scale = phi_err)
        
    if isinstance(logpm, list) == True:
        logphi_err = np.mean(logpm)
        log_phi = st.norm.rvs(loc = log_phi0, scale = logphi_err)
        phi = np.power(10, log_phi)
        
    M_char_err = np.mean((plusmin[1][0], plusmin[1][1]))
    alpha_err = np.mean((plusmin[2][0], plusmin[2][1]))
    
    M_char = st.norm.rvs(loc = M_char0, scale = M_char_err)
    alpha = st.norm.rvs(loc = alpha0, scale = alpha_err)
    
    # print(logpm, phi)
    
    return phi, M_char, alpha

def DPL_rand_draw(data, plusmin, logpm):
    
    phi0, M_char0, alpha0, beta0 = data
    log_phi0 = np.log10(phi0)
    
    if isinstance(logpm, list) == False:
        phi_err = np.mean((plusmin[0][0], plusmin[0][1]))
        phi = st.norm.rvs(loc = phi0, scale = phi_err)
    if isinstance(logpm, list) == True:
        logphi_err = np.mean(logpm)
        log_phi = st.norm.rvs(loc = log_phi0, scale = logphi_err)
        phi = np.power(10, log_phi)
 
    M_char_err = np.mean((plusmin[1][0], plusmin[1][1]))
    alpha_err = np.mean((plusmin[2][0], plusmin[2][1]))
    beta_err = np.mean((plusmin[3][0], plusmin[3][1]))
    
    M_char = st.norm.rvs(loc = M_char0, scale = M_char_err)
    alpha = st.norm.rvs(loc = alpha0, scale = alpha_err)
    beta = st.norm.rvs(loc = beta0, scale = beta_err)
    
    return phi, M_char, alpha, beta
    

#%% create a dataframe with n randomized parameters for each set of schechter params

rand_Schech = pd.DataFrame(columns = paramsDF.columns)
num = 100

for i in range(len(paramsDF.columns)):
    p = (paramsDF[paramsDF.columns[i]].phi_star, paramsDF[paramsDF.columns[i]].M_star, paramsDF[paramsDF.columns[i]].alpha)
    pm = (paramsDF[paramsDF.columns[i]].phi_plusmin, paramsDF[paramsDF.columns[i]].M_plusmin, paramsDF[paramsDF.columns[i]].alpha_plusmin)
    logpm = paramsDF[paramsDF.columns[i]].logphi_pm
    
    rand = []
    
    for x in range(0,num):
        params = schech_rand_draw(p, pm, logpm)
        rand.append(params)
    
    rand_Schech[paramsDF.columns[i]] = rand

#%% create a dataframe with n randomized parameters for each set of DPL params

rand_dpl = pd.DataFrame(columns = DPLparams.columns)
num = 100

for i in range(len(DPLparams.columns)):
    p = (DPLparams[DPLparams.columns[i]].phi_dpl, DPLparams[DPLparams.columns[i]].M_dpl, DPLparams[DPLparams.columns[i]].alpha_dpl, DPLparams[DPLparams.columns[i]].beta)
    pm = (DPLparams[DPLparams.columns[i]].phi_dpl_pm, DPLparams[DPLparams.columns[i]].M_dpl_pm, DPLparams[DPLparams.columns[i]].alpha_dpl_pm, DPLparams[DPLparams.columns[i]].beta_pm)
    logpm = (DPLparams[DPLparams.columns[i]].logphi_pm)
    
    rand = []
    
    for x in range(0,num):
        params = DPL_rand_draw(p, pm, logpm)
        rand.append(params)
    
    rand_dpl[DPLparams.columns[i]] = rand
    
#%% functions for integrating schech, DPL, and superpositions
# psuedo code:

# for z in zgrid:

#     integrate dvdz from 0 to z -> dV(z)
#     get a value for iband mag lim at z in UV -> M(z) (the upper bound on inner integral)
#     integrate over dV*phi(M) from 0 to M -> n
#     sum = sum + n

zgrid = np.arange(4.0, 5.9, 0.1)

def count_pred(survey_area, mag_lim, schechter_params, zgrid, response = R_i, wavelength_ang = lam_i, lowerint_lim = -50): #survey area in degrees sq, mag lim in AB mag, lowerint limit in M_UV
    nsum = 0
    dens_sum = 0
    # n_errsum = 0

    for z in zgrid:
    
        dV,dV_err = integrate.quad(dV_dz, z, z+0.01, args=(survey_area)) #volume in Δz in sky area (Mpc**3)
        Mz = obs_to_UV(mag_lim, z)# + k_corr(mag_lim, response, wavelength_ang, z) 
    
        def integrand2(M, phi, M_char, alpha):
            return dV*Schechter(M, phi, M_char, alpha)
    
        density, d_err = integrate.quad(Schechter, lowerint_lim, Mz, args = schechter_params)
        n, n_err = integrate.quad(integrand2, lowerint_lim, Mz, args = schechter_params)
        nsum = nsum + n
        dens_sum = dens_sum + density
        
    #     n_errsum = n_errsum + n_err
    return(nsum, dens_sum)

def DPL_count_pred(survey_area, mag_lim, DPL_params, zgrid, response = R_i, wavelength_ang = lam_i, lowerint_lim = -50): #survey area in degrees sq, mag lim in AB mag
    nsum = 0
    dens_sum = 0

    for z in zgrid:
    
        dV,dV_err = integrate.quad(dV_dz, z, z+0.01, args=(survey_area)) #volume in Δz in sky area (Mpc**3)
        Mz = obs_to_UV(mag_lim, z) #+ k_corr(mag_lim, response, wavelength_ang, z) 
    
        def integrand2(M, phi, M_char, alpha, beta):
            return dV*DPL(M, phi, M_char, alpha, beta)
    
        density, d_err = integrate.quad(DPL, lowerint_lim, Mz, args = DPL_params)
        n, n_err = integrate.quad(integrand2, lowerint_lim, Mz, args = DPL_params)
        nsum = nsum + n
        dens_sum = dens_sum + density
    #     n_errsum = n_errsum + n_err

    # return(nsum, n_errsum)
    return(nsum, dens_sum)

def DPL_DPL_ctpred(survey_area, mag_lim, DPL_params_agn, DPL_params_gal, zgrid, response = R_i, wavelength_ang = lam_i, lowerint_lim = -50):
    nsum = 0
    dens_sum = 0
    
    phi_agn, Mchar_agn, alpha_agn, beta_agn = DPL_params_agn
    phi_gal, Mchar_gal, alpha_gal, beta_gal = DPL_params_gal
    
    
    for z in zgrid:
        dV,dV_err = integrate.quad(dV_dz, z, z+0.01, args=(survey_area)) #volume in Δz in sky area (Mpc**3)
        Mz = obs_to_UV(mag_lim, z) #+ k_corr(mag_lim, response, wavelength_ang, z) 
        
        def superposition_integrand(M, phi_agn, Mchar_agn, alpha_agn, beta_agn, phi_gal, Mchar_gal, alpha_gal, beta_gal):
            return dV*(DPL(M, phi_agn, Mchar_agn, alpha_agn, beta_agn) + DPL(M, phi_gal, Mchar_gal, alpha_gal, beta_gal))
        
        def sp(M, phi_agn, Mchar_agn, alpha_agn, beta_agn, phi_gal, Mchar_gal, alpha_gal, beta_gal):
            return(DPL(M, phi_agn, Mchar_agn, alpha_agn, beta_agn) + DPL(M, phi_gal, Mchar_gal, alpha_gal, beta_gal))
        
        density, d_err = integrate.quad(sp, lowerint_lim, Mz, args = (*DPL_params_agn, *DPL_params_gal))
        n, n_err = integrate.quad(superposition_integrand, lowerint_lim, Mz, args = (*DPL_params_agn, *DPL_params_gal))
        nsum = nsum + n
        dens_sum = dens_sum + density
    
    return(nsum, dens_sum)

def DPL_S_ctpred(survey_area, mag_lim, DPL_params, schechter_params, zgrid, response = R_i, wavelength_ang = lam_i, lowerint_lim = -50):    
    nsum = 0
    dens_sum = 0
    
    phi_agn, Mchar_agn, alpha_agn, beta_agn = DPL_params
    phi_gal, Mchar_gal, alpha_gal = schechter_params
    
    for z in zgrid:
        dV,dV_err = integrate.quad(dV_dz, z, z+0.01, args=(survey_area)) #volume in Δz in sky area (Mpc**3)
        Mz = obs_to_UV(mag_lim, z) #+ k_corr(mag_lim, response, wavelength_ang, z) 
        
        def superposition_integrand(M, phi_agn, Mchar_agn, alpha_agn, beta_agn, phi_gal, Mchar_gal, alpha_gal):
            return dV*(DPL(M, phi_agn, Mchar_agn, alpha_agn, beta_agn) + Schechter(M, phi_gal, Mchar_gal, alpha_gal))
        
        def sp(M, phi_agn, Mchar_agn, alpha_agn, beta_agn, phi_gal, Mchar_gal, alpha_gal):
            return(DPL(M, phi_agn, Mchar_agn, alpha_agn, beta_agn) + Schechter(M, phi_gal, Mchar_gal, alpha_gal))
        
        density, d_err = integrate.quad(sp, lowerint_lim, Mz, args = (*DPL_params, *schechter_params))
        n, n_err = integrate.quad(superposition_integrand, lowerint_lim, Mz, args = (*DPL_params, *schechter_params))
        nsum = nsum + n
        dens_sum += density
    
    return(nsum, dens_sum)
        

#%% integrate over all schechter functions

S_cts = pd.DataFrame(columns= rand_Schech.columns, index = ['N', 'N_err', 'dens', 'dens_err'])

for i in (range(len(rand_Schech.columns))):
    S_ct = []
    S_phi = []
    for x in tqdm(range(0,num)):
        parameters = rand_Schech[rand_Schech.columns[i]][x]
        n = count_pred(DES_skyarea, 24.0, parameters, zgrid)[0]
        p = count_pred(DES_skyarea, 24.0, parameters, zgrid)[1]
        S_ct.append(n)
        S_phi.append(p)
        
    Navg = np.mean(S_ct)
    Nerr = np.std(S_ct)
    Pavg = np.mean(S_phi)
    Perr = np.std(S_phi)
    
    S_cts[rand_Schech.columns[i]].N = Navg
    S_cts[rand_Schech.columns[i]].N_err = Nerr   
    S_cts[rand_Schech.columns[i]].dens = Pavg
    S_cts[rand_Schech.columns[i]].dens_err = Perr  
          
#%% integrate overe all DPL functions

DPL_cts = pd.DataFrame(columns= rand_dpl.columns, index = ['N', 'N_err', 'dens', 'dens_err'])

for i in (range(len(rand_dpl.columns))):
    DPLct = []
    DPLphi = []
    for x in tqdm(range(0,num)):
        params = rand_dpl[rand_dpl.columns[i]][x]
        n = DPL_count_pred(DES_skyarea, 24.0, params, zgrid)[0]
        p = DPL_count_pred(DES_skyarea, 24.0, params, zgrid)[1]
        DPLct.append(n)
        DPLphi.append(p)
        
    Navg = np.mean(DPLct)
    Nerr = np.std(DPLct)
    Pavg = np.mean(DPLphi)
    Perr = np.std(DPLphi)
    
    DPL_cts[rand_dpl.columns[i]].N = Navg
    DPL_cts[rand_dpl.columns[i]].N_err = Nerr 
    DPL_cts[rand_dpl.columns[i]].dens = Pavg
    DPL_cts[rand_dpl.columns[i]].dens_err = Perr 
        
#%% integrate over DPL+Schech superposition

DPL_S_cts = pd.DataFrame(columns=['(AGN_DPL + GAL_Schech)'], index = ['N', 'N_err', 'dens', 'dens_err'])
DPLS_ct = []
DPLSphi_ct = []

for x in tqdm(range(0,num)):
    agn_params = rand_dpl['Harikane (2021) AGN component (AGN_DPL + GAL_Schechter)'][x]
    gal_params = rand_Schech['Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)'][x]
    n = DPL_S_ctpred(DES_skyarea, 24.0, agn_params, gal_params, zgrid)[0]
    p = DPL_S_ctpred(DES_skyarea, 24.0, agn_params, gal_params, zgrid)[1]
    DPLS_ct.append(n)
    DPLSphi_ct.append(p)

Navg = np.mean(DPLS_ct)
Nerr = np.std(DPLS_ct)
Pavg = np.mean(DPLSphi_ct)
Perr = np.std(DPLSphi_ct)

DPL_S_cts['(AGN_DPL + GAL_Schech)'].N = Navg
DPL_S_cts['(AGN_DPL + GAL_Schech)'].N_err = Nerr 
DPL_S_cts['(AGN_DPL + GAL_Schech)'].dens = Pavg
DPL_S_cts['(AGN_DPL + GAL_Schech)'].dens_err = Perr 

#%% integrate over DPL+DPL superposition   
DPL_DPL_cts = pd.DataFrame(columns=['(AGN_DPL + GAL_DPL)'], index = ['N', 'N_err', 'dens', 'dens_err'])
DPLDPL_ct = []
DPLDPLphi_ct = []

for x in tqdm(range(0,num)):
    agn_params = rand_dpl['Harikane (2021) AGN component (AGN_DPL + GAL_DPL)'][x]
    gal_params = rand_dpl['Harikane (2021) galaxy component (AGN_DPL + GAL_DPL)'][x]
    n = DPL_DPL_ctpred(DES_skyarea, 24.0, agn_params, gal_params, zgrid)[0]
    p = DPL_DPL_ctpred(DES_skyarea, 24.0, agn_params, gal_params, zgrid)[1]
    DPLDPL_ct.append(n)
    DPLDPLphi_ct.append(p)
    
Navg = np.mean(DPLDPL_ct)
Nerr = np.std(DPLDPL_ct)
Pavg = np.mean(DPLDPLphi_ct)
Perr = np.std(DPLDPLphi_ct)

DPL_DPL_cts['(AGN_DPL + GAL_DPL)'].N = Navg
DPL_DPL_cts['(AGN_DPL + GAL_DPL)'].N_err = Nerr 
DPL_DPL_cts['(AGN_DPL + GAL_DPL)'].dens = Pavg
DPL_DPL_cts['(AGN_DPL + GAL_DPL)'].dens_err = Perr 

#%%% SURVEY INFO
#some things to consider (discrepancies):
    # areas covered by different filters in each survey field   

#combine all parameters into one table
masterparamsDF = pd.concat([paramsDF, DPLparams], axis = 0)


#read in survey information for each paper
#survey limits based on the DES i-band equivalent passband 
#counts are the number of z=5 predicted LBG objects (r-band dropouts)

onosurveys = pd.DataFrame(columns = ['survey', 'area_deg2', 'i_lim_AB', 'rbandcounts', 'sample_zrange', 'sample_zavg', 'paper'])
onosurveys.survey = ['UD-SXDS', 'UD-COSMOS', 'D-XMM-LSS', 'D-COSMOS', 'D-ELAIS-N1', 'D-DEEP2-3', 'W-XMM', 'W-GAMA09H', 'W-WIDE12H', 'W-GAMA15H', 'W-HECTOMAP', 'W-VVDS']
onosurveys.area_deg2 = [1.1, 1.3, 2.4, 6.5, 3.3, 5.5, 28.5, 12.4, 15.2, 16.6, 4.8, 5.1]
onosurveys.i_lim_AB = [26.53, 26.46, 25.88, 26.04, 25.87, 25.96, 25.71, 25.65, 25.82, 25.81, 25.82, 25.74]
onosurveys.rbandcounts = [1209, 1990, 711, 6282, 612, 1498, 6371, 5989, 5243, 6457, 1082, 1500]
onosurveys.sample_zrange = [[4.2, 5.5]]*len(onosurveys.survey)
onosurveys.sample_zavg = [4.9]*len(onosurveys.survey)
onosurveys.paper = ['ono2018']*len(onosurveys.survey)

#Note: since the HUDF9 survey does not include the i-band equivalent HST filters, i will be excluding that information from bouwens
bouwenssurveys = pd.DataFrame(columns = ['survey', 'area_deg2', 'i_lim_AB', 'rbandcounts', 'sample_zrange', 'sample_zavg', 'paper'])
bouwenssurveys.survey = ['HUDF/XDF', 'CANDELS-GS-DEEP', 'CANDELS-GS-WIDE', 'ERS', 'CANDELS-GN-DEEP', 'CANDELS-GN-WIDE', 'CANDELS-UDS', 'CANDELS-COSMOS', 'CANDELS-EGS', 'Abell2744-Par', 'MACS0416-PAR', 'MACS0717-PAR', 'MACS1149-PAR', 'AbellS1063-PAR', 'Abell370-PAR', 'HFFtotal', 'all_fields']
bouwenssurveys.area_deg2 = ([4.7, 64.5, 34.2, 40.5, 68.3, 65.4, 151.2, 151.9, 150.7, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 29.4, 1135.9]*u.arcmin**2).to(u.deg**2).value
#using detection limit of F775W for HUDF/XDF and CANDELS surveys (idx 0-8)
bouwenssurveys.i_lim_AB.iloc[0:9] = 28.55
#using 5σ detection limit of HST ACS/WFC F606W for Hubble Frontier Fields (PAR surveys idx 9-16) provided in Table 3 of Lotz et al 2016 (THE FRONTIER FIELDS: SURVEY DESIGN)
bouwenssurveys.i_lim_AB.iloc[9:16] = 28.8
bouwenssurveys.i_lim_AB.iloc[16] = np.mean(bouwenssurveys.i_lim_AB) #all fields
bouwenssurveys.rbandcounts = [153, 471, 117, 205, 634, 282, 270, 320, 381, 67, 71, 55, 76, 79, 100, 448, 3449]
bouwenssurveys.sample_zrange = [[4.1, 5.9]]*len(bouwenssurveys.survey)
bouwenssurveys.sample_zavg = [4.9]*len(bouwenssurveys.survey)
bouwenssurveys.paper = ['bouwens2021a']*len(bouwenssurveys.survey)

harikanesurveys = pd.DataFrame(columns = ['survey', 'area_deg2', 'i_lim_AB', 'rbandcounts', 'sample_zrange', 'sample_zavg', 'paper'])
harikanesurveys.survey = ['UD-SXDS', 'UD-COSMOS', 'D-XMM-LSS', 'D-COSMOS', 'D-ELAIS-N1', 'D-DEEP2-3', 'W-W02', 'W-W03', 'W-W04', 'W-W05', 'W-W06', 'W-W07', 'all_fields']
harikanesurveys.area_deg2 = [1.3, 1.3, 2.2, 4.9, 5.4, 5.1, 33.3, 66.2, 72.2, 86.6, 28.4, 0.9, 307.9]
harikanesurveys.i_lim_AB = [26.57, 26.75, 25.87, 26.32, 26.13, 25.98, 25.69, 25.76, 25.86, 25.61, 25.78, 25.79, np.nan]
harikanesurveys.loc[12, 'i_lim_AB'] = np.mean(harikanesurveys.i_lim_AB)
harikanesurveys.rbandcounts = [1517, 2760, 1237, 6439, 3947, 3808, 8034, 29758, 35440, 35105, 11166, 148, 139359]
harikanesurveys.sample_zrange = [[4.2, 5.8]]*len(harikanesurveys.survey)
harikanesurveys.sample_zavg = [4.9]*len(harikanesurveys.survey)
harikanesurveys.paper = ['harikane2021']*len(harikanesurveys.survey)

surveysDF = pd.concat([onosurveys, bouwenssurveys, harikanesurveys])

#%% ONO SCHECHTER FUNCTION CHECK
onocheck = pd.DataFrame(index = [onosurveys.survey], columns= ['N_pred', 'N_err', 'N_true'])
onocheck['N_true'] = onosurveys.rbandcounts.values

for i in range(len(onosurveys)):
    phivals = []
    Nvals = []
    for x in tqdm(range(num)):
        params = rand_Schech['Ono (2018)'][x]
        area = Quantity(onosurveys.area_deg2[i], "deg2")
        zgrid = np.arange(onosurveys.sample_zrange[i][0], onosurveys.sample_zrange[i][1], 0.1)
        
        phi = count_pred(area, onosurveys.i_lim_AB[i], params, zgrid)[1]
        N = count_pred(area, onosurveys.i_lim_AB[i], params, zgrid)[0]
        phivals.append(phi)
        Nvals.append(N)
        
    avg_dens = np.mean(phi)
    err_dens = np.std(phi)
    avg_N = np.mean(Nvals)
    err_N = np.std(Nvals)
    
    onocheck.loc[onosurveys.survey[i], 'N_pred'] = avg_N
    onocheck.loc[onosurveys.survey[i], 'N_err'] = err_N

onocheck['ratio'] = onocheck['N_pred']/onocheck['N_true']
onocheck['order of mag diff'] = np.log10(np.array(onocheck['ratio']).astype(float))
#%% ONO SCHECHTER FUNCTION CHECK BAR PLOT

fig, ax = plt.subplots()

N = len(onocheck)
ind = np.arange(0,N)
width = 0.25

ax.bar(ind, (onocheck['N_true']), width, color = 'blue', label = 'Ono 2018 truth counts')
ax.bar(ind+width, (onocheck['N_pred']), width, color = 'tan', label = 'algorithm prediction')
ax.bar(ind+width, (onocheck['N_pred']), width, yerr=onocheck.N_err, color = 'orange', align='center', alpha=0.5, ecolor='black', capsize=5)
plt.xticks(ind, onosurveys.survey, rotation = 35, size = 6.5)
# plt.yscale('log')
plt.ylabel('$\log (N)$')
plt.legend()

#%% ONO DPL FUNCTION CHECK
onoDPLcheck = pd.DataFrame(index = [onosurveys.survey], columns= ['N_pred', 'N_err', 'N_true'])
onoDPLcheck['N_true'] = onosurveys.rbandcounts.values

for i in range(len(onosurveys)):
    phivals = []
    Nvals = []
    for x in tqdm(range(num)):
        params = rand_dpl['Ono (2018) DPL'][x]
        area = Quantity(onosurveys.area_deg2[i], "deg2")
        zgrid = np.arange(onosurveys.sample_zrange[i][0], onosurveys.sample_zrange[i][1], 0.1)
        
        phi = DPL_count_pred(area, onosurveys.i_lim_AB[i], params, zgrid)[1]
        N = DPL_count_pred(area, onosurveys.i_lim_AB[i], params, zgrid)[0]
        phivals.append(phi)
        Nvals.append(N)
        
    avg_dens = np.mean(phi)
    err_dens = np.std(phi)
    avg_N = np.mean(Nvals)
    err_N = np.std(Nvals)
    
    onoDPLcheck.loc[onosurveys.survey[i], 'N_pred'] = avg_N
    onoDPLcheck.loc[onosurveys.survey[i], 'N_err'] = err_N
    
onoDPLcheck['ratio'] = onoDPLcheck['N_pred']/onoDPLcheck['N_true']
onoDPLcheck['order of mag diff'] = np.log10(np.array(onoDPLcheck['ratio']).astype(float))

#%%
fig, ax = plt.subplots()

N = len(onoDPLcheck)
ind = np.arange(0,N)
width = 0.25

ax.bar(ind, (onoDPLcheck['N_true']), width, color = 'blue', label = 'Ono 2018 truth counts')
ax.bar(ind+width, (onoDPLcheck['N_pred']), width, color = 'tan', label = 'algorithm prediction')
ax.bar(ind+width, (onoDPLcheck['N_pred']), width, yerr=onoDPLcheck.N_err, color = 'orange', align='center', alpha=0.5, ecolor='black', capsize=5)
plt.xticks(ind, onosurveys.survey, rotation = 35, size = 6.5)
plt.yscale('log')
plt.ylabel('$\log (N)$')
plt.legend()

#%% BOUWENS CHECK
 
bouwenscheck = pd.DataFrame(index = [bouwenssurveys.survey], columns = ['N_pred', 'N_err', 'N_true'])
bouwenscheck['N_true'] = bouwenssurveys.rbandcounts.values

for i in range(len(bouwenssurveys)):
    phivals = []
    Nvals = []
    for x in tqdm(range(num)):
        params = rand_Schech['Bouwens (2021a)'][x]
        area = Quantity(bouwenssurveys.area_deg2[i], "deg2")
        zgrid = np.arange(bouwenssurveys.sample_zrange[i][0], bouwenssurveys.sample_zrange[i][1], 0.1)
        
        phi = count_pred(area, bouwenssurveys.i_lim_AB[i], params, zgrid)[1]
        N = count_pred(area, bouwenssurveys.i_lim_AB[i], params, zgrid)[0]
        phivals.append(phi)
        Nvals.append(N)
        
    avg_dens = np.mean(phi)
    err_dens = np.std(phi)
    avg_N = np.mean(Nvals)
    err_N = np.std(Nvals)
    
    bouwenscheck.loc[bouwenssurveys.survey[i], 'N_pred'] = avg_N
    bouwenscheck.loc[bouwenssurveys.survey[i], 'N_err'] = err_N

bouwenscheck['ratio'] = bouwenscheck['N_pred']/bouwenscheck['N_true']
bouwenscheck['order of mag diff'] = np.log10(np.array(bouwenscheck['ratio']).astype(float))

#%%
fig, ax = plt.subplots()

bshort = bouwenssurveys[0:9].append(bouwenssurveys[-2:]) #remove individual parallel fields
N = len(bshort)
ind = np.arange(0,N)
width = 0.25

N_trueshort = bouwenscheck['N_true'][0:9].append(bouwenscheck.N_true[-2:])
N_predshort = bouwenscheck['N_pred'][0:9].append(bouwenscheck.N_pred[-2:])

ax.bar(ind, (N_trueshort), width, color = 'blue', label = 'Bouwens 2021a truth counts')
ax.bar(ind+width, (N_predshort), width, color = 'tan', label = 'algorithm prediction')
ax.bar(ind+width, (N_predshort), width, yerr=bouwenscheck.N_err[0:9].append(bouwenscheck.N_err[-2:]), color = 'orange', align='center', alpha=0.5, ecolor='black', capsize=5)
plt.xticks(ind, bshort.survey, rotation = 35, size = 6.5)
plt.yscale('log')
plt.ylabel('$\log (N)$')
plt.legend()

#%%  HARIKANE CHECK: DPL+DPL, DPL+SCHECH, GALSCHECH, GALDPL

harikane2DPL_check = pd.DataFrame(index = [harikanesurveys.survey], columns = ['N_pred', 'N_err', 'N_true'])
harikane2DPL_check['N_true'] = harikanesurveys.rbandcounts.values

harikaneDPLS_check = pd.DataFrame(index = [harikanesurveys.survey], columns = ['N_pred', 'N_err', 'N_true'])
harikaneDPLS_check['N_true'] = harikanesurveys.rbandcounts.values

harikaneGALS_check = pd.DataFrame(index = [harikanesurveys.survey], columns = ['N_pred', 'N_err', 'N_true'])
harikaneGALS_check['N_true'] = harikanesurveys.rbandcounts.values

harikaneGALDPL_check = pd.DataFrame(index = [harikanesurveys.survey], columns = ['N_pred', 'N_err', 'N_true'])
harikaneGALDPL_check['N_true'] = harikanesurveys.rbandcounts.values

#%% HARIKANE DPL+DPL CHECK
for i in range(len(harikanesurveys)):
    phivals = []
    Nvals = []
    for x in tqdm(range(num)):
        
        agn_params = rand_dpl['Harikane (2021) AGN component (AGN_DPL + GAL_DPL)'][x]
        gal_params = rand_dpl['Harikane (2021) galaxy component (AGN_DPL + GAL_DPL)'][x]
        area = Quantity(harikanesurveys.area_deg2[i], "deg2")
        zgrid = np.arange(harikanesurveys.sample_zrange[i][0], harikanesurveys.sample_zrange[i][1], 0.1)
        
        phi = DPL_DPL_ctpred(area, harikanesurveys.i_lim_AB[i], agn_params, gal_params, zgrid)[1]
        N = DPL_DPL_ctpred(area, harikanesurveys.i_lim_AB[i], agn_params, gal_params, zgrid)[0]
        phivals.append(phi)
        Nvals.append(N)
        
    avg_dens = np.mean(phi)
    err_dens = np.std(phi)
    avg_N = np.mean(Nvals)
    err_N = np.std(Nvals)
    
    harikane2DPL_check.loc[harikanesurveys.survey[i], 'N_pred'] = avg_N
    harikane2DPL_check.loc[harikanesurveys.survey[i], 'N_err'] = err_N
    
harikane2DPL_check['ratio'] = harikane2DPL_check['N_pred']/harikane2DPL_check['N_true']
harikane2DPL_check['order of mag diff'] = np.log10(np.array(harikane2DPL_check['ratio']).astype(float))

#%%

fig, ax = plt.subplots()

N = len(harikane2DPL_check)
ind = np.arange(0,N)
width = 0.25

ax.bar(ind, (harikane2DPL_check['N_true']), width, color = 'blue', label = 'Harikane 2021 truth counts')
ax.bar(ind+width, (harikane2DPL_check['N_pred']), width, color = 'tan', label = 'algorithm prediction')
ax.bar(ind+width, (harikane2DPL_check['N_pred']), width, yerr=harikane2DPL_check.N_err, color = 'orange', align='center', alpha=0.5, ecolor='black', capsize=5)
plt.xticks(ind, harikanesurveys.survey, rotation = 35, size = 6.5)
plt.yscale('log')
plt.ylabel('$\log (N)$')
plt.title('Harikane DPL + DPL integration')
plt.legend()

#%% HARIKANE DPL+S CHECK
for i in range(len(harikanesurveys)):
    phivals = []
    Nvals = []
    for x in tqdm(range(num)):
        
        agn_params = rand_dpl['Harikane (2021) AGN component (AGN_DPL + GAL_Schechter)'][x]
        gal_params = rand_Schech['Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)'][x]
        area = Quantity(harikanesurveys.area_deg2[i], "deg2")
        zgrid = np.arange(harikanesurveys.sample_zrange[i][0], harikanesurveys.sample_zrange[i][1], 0.1)
        
        phi = DPL_S_ctpred(area, harikanesurveys.i_lim_AB[i], agn_params, gal_params, zgrid)[1]
        N = DPL_S_ctpred(area, harikanesurveys.i_lim_AB[i], agn_params, gal_params, zgrid)[0]
        phivals.append(phi)
        Nvals.append(N)
        
    avg_dens = np.mean(phi)
    err_dens = np.std(phi)
    avg_N = np.mean(Nvals)
    err_N = np.std(Nvals)
    
    harikaneDPLS_check.loc[harikanesurveys.survey[i], 'N_pred'] = avg_N
    harikaneDPLS_check.loc[harikanesurveys.survey[i], 'N_err'] = err_N
    
harikaneDPLS_check['ratio'] = harikaneDPLS_check['N_pred']/harikaneDPLS_check['N_true']
harikaneDPLS_check['order of mag diff'] = np.log10(np.array(harikaneDPLS_check['ratio']).astype(float))

#%%
fig, ax = plt.subplots()

N = len(harikaneDPLS_check)
ind = np.arange(0,N)
width = 0.25

ax.bar(ind, (harikaneDPLS_check['N_true']), width, color = 'blue', label = 'Harikane 2021 truth counts')
ax.bar(ind+width, (harikaneDPLS_check['N_pred']), width, color = 'tan', label = 'algorithm prediction')
ax.bar(ind+width, (harikaneDPLS_check['N_pred']), width, yerr=harikaneDPLS_check.N_err, color = 'orange', align='center', alpha=0.5, ecolor='black', capsize=5)
plt.xticks(ind, harikanesurveys.survey, rotation = 35, size = 6.5)
# plt.yscale('log')
# plt.ylabel('$\log (N)$')
plt.title('Harikane DPL + Schechter integration')
plt.legend()

#%% HARIKANE GAL SCHECH CHECK
for i in range(len(harikanesurveys)):
    phivals = []
    Nvals = []
    for x in tqdm(range(num)):
        params = rand_Schech['Harikane (2021) Galaxy LF (Schechter)'][x]
        area = Quantity(harikanesurveys.area_deg2[i], "deg2")
        zgrid = np.arange(harikanesurveys.sample_zrange[i][0], harikanesurveys.sample_zrange[i][1], 0.1)
        
        phi = count_pred(area, harikanesurveys.i_lim_AB[i], params, zgrid)[1]
        N = count_pred(area, harikanesurveys.i_lim_AB[i], params, zgrid)[0]
        phivals.append(phi)
        Nvals.append(N)
        
    avg_dens = np.mean(phi)
    err_dens = np.std(phi)
    avg_N = np.mean(Nvals)
    err_N = np.std(Nvals)
    
    harikaneGALS_check.loc[harikanesurveys.survey[i], 'N_pred'] = avg_N
    harikaneGALS_check.loc[harikanesurveys.survey[i], 'N_err'] = err_N
    
harikaneGALS_check['ratio'] = harikaneGALS_check['N_pred']/harikaneGALS_check['N_true']
harikaneGALS_check['order of mag diff'] = np.log10(np.array(harikaneGALS_check['ratio']).astype(float))

#%%

fig, ax = plt.subplots()

N = len(harikaneGALS_check)
ind = np.arange(0,N)
width = 0.25

ax.bar(ind, (harikaneGALS_check['N_true']), width, color = 'blue', label = 'Harikane 2021 truth counts')
ax.bar(ind+width, (harikaneGALS_check['N_pred']), width, color = 'tan', label = 'algorithm prediction')
ax.bar(ind+width, (harikaneGALS_check['N_pred']), width, yerr=harikaneGALS_check.N_err, color = 'orange', align='center', alpha=0.5, ecolor='black', capsize=5)
plt.xticks(ind, harikanesurveys.survey, rotation = 35, size = 6.5)
plt.yscale('log')
plt.ylabel('$\log (N)$')
plt.title('Harikane GAL Schechter integration')
plt.legend()

#%% HARIKANE GAL DPL CHECK
for i in range(len(harikanesurveys)):
    phivals = []
    Nvals = []
    for x in tqdm(range(num)):
        params = rand_dpl['Harikane (2021) Galaxy LF (DPL)'][x]
        area = Quantity(harikanesurveys.area_deg2[i], "deg2")
        zgrid = np.arange(harikanesurveys.sample_zrange[i][0], harikanesurveys.sample_zrange[i][1], 0.1)
        
        phi = DPL_count_pred(area, harikanesurveys.i_lim_AB[i], params, zgrid)[1]
        N = DPL_count_pred(area, harikanesurveys.i_lim_AB[i], params, zgrid)[0]
        phivals.append(phi)
        Nvals.append(N)
        
    avg_dens = np.mean(phi)
    err_dens = np.std(phi)
    avg_N = np.mean(Nvals)
    err_N = np.std(Nvals)
    
    harikaneGALDPL_check.loc[harikanesurveys.survey[i], 'N_pred'] = avg_N
    harikaneGALDPL_check.loc[harikanesurveys.survey[i], 'N_err'] = err_N
    
harikaneGALDPL_check['ratio'] = harikaneGALDPL_check['N_pred']/harikaneGALDPL_check['N_true']
harikaneGALDPL_check['order of mag diff'] = np.log10(np.array(harikaneGALDPL_check['ratio']).astype(float))

#%%

fig, ax = plt.subplots()

N = len(harikaneGALS_check)
ind = np.arange(0,N)
width = 0.25

ax.bar(ind, ( harikaneGALDPL_check['N_true']), width, color = 'blue', label = 'Harikane 2021 truth counts')
ax.bar(ind+width, ( harikaneGALDPL_check['N_pred']), width, color = 'tan', label = 'algorithm prediction')
ax.bar(ind+width, ( harikaneGALDPL_check['N_pred']), width, yerr= harikaneGALDPL_check.N_err, color = 'orange', align='center', alpha=0.5, ecolor='black', capsize=5)
plt.xticks(ind, harikanesurveys.survey, rotation = 35, size = 6.5)
plt.yscale('log')
plt.ylabel('$\log (N)$')
plt.title('Harikane GAL DPL integration')
plt.legend()
