#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:13:57 2022

@author: bflucero
"""

import LFfunc as lf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from speclite import filters

# %% Filter response curves + SED template

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

# %% plot M_uv v m_i and kcorrection over z=0-5

zgrid = np.linspace(0, 6, 1000)

k_UV_i = lf.k_corr(DECam[2], UV[0], SED, zgrid)

# %%

# kcorrection values at z
k_UV_i45 = lf.k_corr(DECam[2], UV[0], SED, 4.5)
k_UV_i5 = lf.k_corr(DECam[2], UV[0], SED, 5)
k_UV_i55 = lf.k_corr(DECam[2], UV[0], SED, 5.5)

# define range of rest frame M_uv
Muv = np.linspace(-29, -18, 50)

# corresponding values of m_i converted from M_uv
mi45 = lf.UV_to_obs(Muv, z=4.5)
mi5 = lf.UV_to_obs(Muv, z=5)
mi55 = lf.UV_to_obs(Muv, z=5.5)
# mi = lf.UV_to_obs(Muv, zgrid)

# rest frame muv at z=5
muv5 = lf.m_uv_kcorr(mi5, 5, k_UV_i5)

# %% plot magnitude relations and k-correct
fig, ax = plt.subplots(2, 2)

ax[0][0].plot(Muv, mi5, label='M_UV to m_AB')
plotz5 = ax[0][0].plot(Muv, np.subtract(mi5, k_UV_i5),
                       label='UV to m_AB \n w/ k-correction at 4.5<z<5.0')
ax[0][0].fill_between(Muv, np.subtract(mi45, k_UV_i45),
                      np.subtract(mi5, k_UV_i5),
                      alpha=0.3, color=plotz5[0].get_color())
ax[0][0].fill_between(Muv, np.subtract(mi5, k_UV_i5),
                      np.subtract(mi55, k_UV_i55),
                      alpha=0.3, color=plotz5[0].get_color())
ax[0][0].set_ylabel('$m_{i}$ (AB)')
ax[0][0].set_xlabel('$M_{UV}$')
ax[0][0].legend(fontsize='xx-small')
ax[0][0].set_title('$M_{UV}$ vs $m_{i}$ (AB)')
ax[0][0].set_box_aspect(1)

ax[0][1].plot(zgrid, k_UV_i)
ax[0][1].set_xlabel('z')
ax[0][1].set_ylabel('k-correction \n $M_{UV}$ to $m_{i}$')
ax[0][1].set_title('k correction at 0<z<6')
ax[0][1].set_box_aspect(1)

ax[1][0].plot(muv5, mi5)
ax[1][0].set_ylabel('$m_{i}$ (AB)')
ax[1][0].set_xlabel('$m_{UV}$')
ax[1][0].set_title('$m_{UV}$ vs $m_{i}$ (AB)')
ax[1][0].set_box_aspect(1)

ax[1][1].plot(muv5, Muv)
ax[1][1].set_ylabel('$M_{UV}$')
ax[1][1].set_xlabel('$m_{UV}$')
ax[1][1].set_title('$M_{UV}$ vs $m_{UV}$')
ax[1][1].set_box_aspect(1)

fig.tight_layout(pad=1.03)
