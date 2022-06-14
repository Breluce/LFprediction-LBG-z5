#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 20:45:40 2022

@author: bflucero
"""

import LFfunc as lf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from speclite import filters

# %% read in parameters

SCparams = pd.read_pickle("schechterparams.pkl")
DPLparams = pd.read_pickle("DPLparams.pkl")

# GALAXY ONLY LF'S

galLF_SC = SCparams[['Ono (2018)', 'Harikane (2021) Galaxy LF (Schechter)']]
galLF_DPL = DPLparams[['Ono (2018) DPL', 'Harikane (2021) Galaxy LF (DPL)']]

# GALAXY + AGN LF'S SUPERPOSITION FUNCTIONS

agn2DPL = 'Harikane (2021) AGN component (AGN_DPL + GAL_DPL)'
gal2DPL = 'Harikane (2021) galaxy component (AGN_DPL + GAL_DPL)'
agnDPL_SC = 'Harikane (2021) AGN component (AGN_DPL + GAL_Schechter)'
galDPL_SC = 'Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)'


# agnDPL + galDPL
def sp2DPL(mags):
    """Return the superposition of AGN DPL and GAL DPL functions."""
    sp = lf.DPL(mags, DPLparams[agn2DPL].phi_dpl, DPLparams[agn2DPL].M_dpl,
                DPLparams[agn2DPL].alpha_dpl,
                DPLparams[agn2DPL].beta) + lf.DPL(mags,
                                                  DPLparams[gal2DPL].phi_dpl,
                                                  DPLparams[gal2DPL].M_dpl,
                                                  DPLparams[gal2DPL].alpha_dpl,
                                                  DPLparams[gal2DPL].beta)
    return(sp)


# agnDPL + galSC
def spDPL_SC(mags):
    """Return the superposition of AGN DPL and GAL Schechter functions."""
    sp = lf.DPL(mags, DPLparams[agnDPL_SC].phi_dpl,
                DPLparams[agnDPL_SC].M_dpl, DPLparams[agnDPL_SC].alpha_dpl,
                DPLparams[agnDPL_SC].beta) + lf.Schechter(mags,
                                                          SCparams[galDPL_SC].phi_star,
                                                          SCparams[galDPL_SC].M_star,
                                                          SCparams[galDPL_SC].alpha)
    return(sp)

# %% Filter response curves + SED


'''read in SED file'''
SED = pd.read_csv('galsedaNGC4670.txt', delimiter='\s+', comment='#',
                  usecols=[0, 1], names=['lam', 'flux'])


'''read in UV filter and add to speclite filters'''
dataUV = pd.read_csv('f1600_at0.010.res', sep='\s+', header=None,
                     names=['obslam', 'response'])
UVfilt = filters.FilterResponse(wavelength=dataUV.obslam/1.01,
                                response=dataUV.response,
                                meta=dict(group_name='sample',
                                          band_name='UV'))

'''load UV filter'''
UV = filters.load_filters('sample-UV')
Muv0_speclite = UV[0].get_ab_magnitude(SED.flux, SED.lam)
uvz5 = UV[0].create_shifted(5)
Muv5_speclite = uvz5.get_ab_magnitude(SED.flux, SED.lam)
# filters.plot_filters(UV)

'''load DECam DR1 filters'''
DECam = filters.load_filters('decamDR1-*')
mi0_speclite = DECam[2].get_ab_magnitude(SED.flux, SED.lam)
iz5 = DECam[2].create_shifted(5)
mi5_speclite = iz5.get_ab_magnitude(SED.flux, SED.lam)
# filters.plot_filters(DECam)

'''load LSST filters'''
LSST = filters.load_filters('lsst2016-*')
# filters.plot_filters(LSST)

# %% SET REDSHIFT VALUE AND GET K-CORRECTION
z = 5.0
kcorr_z = lf.k_corr(DECam[2], UV[0], SED, z)

# onomags = np.linspace(-26, -14, num = 500) #intro of ono2018
# bouwensmags = np.linspace(-21.85,-17.60) #table 4 of bouwens2021a
# harikanemags = np.linspace(-29, -14) #various sections of harikane2021

# %% GALAXY LF PLOTS

fig, ax = plt.subplots(1, 1)
mags = np.linspace(-27, -16, num=100)
# plot galaxies

for i in galLF_SC:
    params = [galLF_SC[i].phi_star, galLF_SC[i].M_star, galLF_SC[i].alpha]

    plus = [galLF_SC[i].phi_plusmin[0], galLF_SC[i].M_plusmin[0],
            galLF_SC[i].alpha_plusmin[0]]
    uplim = np.add(params, plus)
    minus = [galLF_SC[i].phi_plusmin[1], galLF_SC[i].M_plusmin[1],
             galLF_SC[i].alpha_plusmin[1]]
    lowlim = np.subtract(params, minus)

    plot = plt.plot(mags, (lf.Schechter(mags, *params)),
                    label='{name}'.format(name=galLF_SC[i].name),
                    alpha=0.45)
    ax.fill_between(mags, (lf.Schechter(mags, *params)),
                    (lf.Schechter(mags, *uplim)), alpha=0.15,
                    label='error {}'.format(galLF_SC[i].name),
                    color=plot[0].get_color())
    ax.fill_between(mags, (lf.Schechter(mags, *params)),
                    (lf.Schechter(mags, *lowlim)), alpha=0.15)

for i in galLF_DPL:
    params = [galLF_DPL[i].phi_dpl, galLF_DPL[i].M_dpl, galLF_DPL[i].alpha_dpl,
              galLF_DPL[i].beta]

    plus = [galLF_DPL[i].phi_dpl_pm[0], galLF_DPL[i].M_dpl_pm[0],
            galLF_DPL[i].alpha_dpl_pm[0], galLF_DPL[i].beta_pm[0]]
    uplim = np.add(params, plus)
    minus = [galLF_DPL[i].phi_dpl_pm[1], galLF_DPL[i].M_dpl_pm[1],
             galLF_DPL[i].alpha_dpl_pm[1], galLF_DPL[i].beta_pm[1]]
    lowlim = np.subtract(params, minus)

    plot2 = plt.plot(mags, lf.DPL(mags, *params),
                     label='{name}'.format(name=DPLparams[i].name),
                     color='cyan', linestyle=':', zorder=20)
    ax.fill_between(mags, (lf.DPL(mags, *params)), (lf.DPL(mags, *uplim)),
                    alpha=0.15, label='error {}'.format(DPLparams[i].name),
                    color=plot2[0].get_color())
    ax.fill_between(mags, (lf.DPL(mags, *params)), (lf.DPL(mags, *lowlim)),
                    alpha=0.15, color=plot2[0].get_color())

secx = ax.secondary_xaxis('top',
                          functions=(lambda x: lf.UV_to_obs(x, z)-kcorr_z,
                                     lambda x: lf.obs_to_UV(x, z)+kcorr_z))
secx.set(xlabel='$m_{obs}$ (AB) k-corrected')
ax.axvline(x=lf.obs_to_UV(24.0, z)+kcorr_z, linestyle="--", c='black',
           linewidth=0.9, label='DES i-band maglim=24.0 \n (k-corrected)')
ax.axvline(x=lf.obs_to_UV(24.0, z), linestyle="--", c='gray',
           linewidth=0.9, label='DES i-band maglim=24.0 \n (NOT k-corrected)')
ax.annotate("", xy=(lf.obs_to_UV(24, 5.5)+kcorr_z, 10**-6),
            xytext=(-27.5, 10**-6),
            arrowprops=dict(arrowstyle="->"), zorder=50)
plt.xlabel('$M_{UV}$')
plt.ylabel('$\phi(M)_{GAL}$ [mag$^{-1}$ Mpc$^{-3}$]')
plt.yscale('log')
plt.ylim(pow(10, -12), pow(10, -2))
plt.title('Galaxy LF at z = {}'.format(z))
plt.legend()

# %% AGN+GAL LF PLOTS

fig, ax = plt.subplots(1, 1)
spmags = np.linspace(-27, -16, num=100)

plt.plot(spmags, sp2DPL(spmags), label='AGN DPL + GAL DPL', c='brown',
         alpha=0.45, zorder=20)
plt.plot(spmags, spDPL_SC(spmags), label='AGN DPL + GAL Schechter', c='cyan',
         alpha=0.45, zorder=20)

ax.annotate("AGN-dominated", xy=(-25, 10**(-3)),
            fontsize='x-small', c='blue')
ax.annotate("LBG-dominated", xy=(-21.5, 10**(-3)),
            fontsize='x-small', c='red')

secx = ax.secondary_xaxis('top',
                          functions=(lambda x: lf.UV_to_obs(x, z)-kcorr_z,
                                     lambda x: lf.obs_to_UV(x, z)+kcorr_z))
secx.set(xlabel='$m_{obs}$ (AB) k-corrected')
ax.axvline(x=lf.obs_to_UV(24.0, z)+kcorr_z, linestyle="--", c='black',
           linewidth=0.9, label='DES i-band maglim=24.0 \n (k-corrected)')
ax.axvline(x=lf.obs_to_UV(24.0, z), linestyle="--", c='gray',
           linewidth=0.9, label='DES i-band maglim=24.0 \n (NOT k-corrected)')
ax.annotate("", xy=(lf.obs_to_UV(24, z)+kcorr_z, 10**-6),
            xytext=(-26.5, 10**-6),
            arrowprops=dict(arrowstyle="->"), zorder=50)
ax.axvline(x=-24, linestyle="--", c='blue', linewidth=0.9,
           label='upper lim on AGN-dominated LF \n 0% galaxy fraction for z ~ 4-7 (Harikane 2021)') #fig 5 in literature
ax.axvline(x=-22, linestyle="--", c='red', linewidth=0.9,
           label='lower lim for gal-dominated LF \n 100% galaxy fraction for z ~ 4-7 (Harikane 2021)') #fig 5 in literature

plt.xlabel('$M_{UV}$')
plt.ylabel('$\phi(M)_{AGN+GAL}$ (mag$^{-1}$ Mpc$^{-3}$)')
plt.yscale('log')
plt.title('AGN+GAL LF at z = {}'.format(z))
plt.ylim(pow(10, -12), pow(10, -2))
plt.legend(fontsize='small', loc='lower right')
