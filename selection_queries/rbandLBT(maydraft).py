#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:16:43 2022

@author: bflucero
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib.offsetbox import AnchoredText
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy import units as u
import os

# %% load DES SVA1 COSMOS overlap subset catalog

with fits.open('./sva1_cosmossubset.fits') as hdu:
    data = hdu[1].data
    wcs = WCS(hdu[0].header)
    SVA1 = Table(data)
    hdu.close()

SVA1sub = SVA1.to_pandas()
# %% load COSMOS Hyper Suprime Camera photometry + photoz

with fits.open('./COSMOS2020_HSC.fits') as hdu:
    data = hdu[1].data
    C = Table(data)
    hdu.close()

COSMOS = C.to_pandas()

# %% load zCOSMOS (specz catalog)

with fits.open('./ADP.2015-10-15T12_55_10.777.fits') as hdu:
    data = hdu[1].data
    zC = Table(data)
    hdu.close()

zCOSMOS = zC.to_pandas()
zCOSMOS = zCOSMOS.dropna(subset=['REDSHIFT'])

# %% crossmatch SVA1 and COSMOS2020

# RAmin = np.min(COSMOS.ALPHA_J2000.values)
# RAmax = np.max(COSMOS.ALPHA_J2000.values)
# DECmin = np.min(COSMOS.DELTA_J2000.values)
# DECmax = np.max(COSMOS.DELTA_J2000.values)

# # approximate number of objects overlapping the GDS region
# approx_ct = len(SVA1sub.loc[(SVA1sub.RA.values < RAmax) &
#                             (SVA1sub.RA.values > RAmin) &
#                             (SVA1sub.DEC.values < DECmax) &
#                             (SVA1sub.RA.values > DECmin)])

ra1 = np.array(SVA1sub.RA.values)
dec1 = np.array(SVA1sub.DEC.values)

ra2 = np.array(COSMOS.ALPHA_J2000.values)
dec2 = np.array(COSMOS.DELTA_J2000.values)

c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)

# specify the maximum sep between objects in order to be considered a match
maxsep = 0.25*u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraint = d2d < maxsep
SVA1_matches = c[sep_constraint]
COSMOS_matches = catalog[idx[sep_constraint]]
# to access individual values: catalog[idx].ra or catalog[idx].dec

# matched objects with all catalog columns
SVA1matched = SVA1sub[sep_constraint]
COSMOSmatched = COSMOS.iloc[idx[sep_constraint]]

# %% crossmatch SVA1matched and zCOSMOS

ra_one = np.array(SVA1matched.RA.values)
dec_one = np.array(SVA1matched.DEC.values)

ra_two = np.array(zCOSMOS.RAJ2000.values)
dec_two = np.array(zCOSMOS.DEJ2000.values)

sva1pos = SkyCoord(ra=ra_one*u.degree, dec=dec_one*u.degree)
zcosmospos = SkyCoord(ra=ra_two*u.degree, dec=dec_two*u.degree)

index, r2r, r3r = sva1pos.match_to_catalog_3d(zcosmospos)
sepcons = r2r < maxsep

# matched objects with all catalog columns
zSVA1matched = SVA1matched[sepcons]
zCOSMOSmatched = zCOSMOS.iloc[index[sepcons]]

# %% add spec z column

zSVA1matched = zSVA1matched.sort_values(by=['RA'])
zCOSMOSmatched = zCOSMOSmatched.sort_values(by=['RAJ2000'])

zSVA1matched['z_spec'] = zCOSMOSmatched.REDSHIFT.values

# %% merge spec z with full SVA1matched catalog

SVA1matched = SVA1matched.merge(zSVA1matched, how='left')

# %% plot overlap of catalogs and cross-matched objects

fig, ax = plt.subplots(1, 1)

plt.scatter(COSMOS.ALPHA_J2000, COSMOS.DELTA_J2000, label='COSMOS HSC',
            alpha=0.5)
# plt.scatter(zCOSMOS.RAJ2000, zCOSMOS.DEJ2000, label='COSMOS w/ spec_z')
plt.scatter(SVA1_matches.ra, SVA1_matches.dec, s=50, color='red',
            marker="x", label='matched objects in SVA1', alpha=0.002)
plt.legend()

# %% color selection

SVA1_cc = SVA1matched.loc[(SVA1matched['MAG_AUTO_R'] - SVA1matched['MAG_AUTO_I'] > 1.2) &
                  (SVA1matched['MAG_AUTO_R'] - SVA1matched['MAG_AUTO_I'] < 25) & #remove null vals
                  (SVA1matched['MAG_AUTO_R'] - SVA1matched['MAG_AUTO_I'] > 1.5*(SVA1matched['MAG_AUTO_I'] - SVA1matched['MAG_AUTO_Z']) + 1) &
                  (SVA1matched['MAG_AUTO_I'] - SVA1matched['MAG_AUTO_Z'] < 0.7) &
                  (SVA1matched['MAG_AUTO_I'] - SVA1matched['MAG_AUTO_Z'] > -40) & #remove null vals
                  (SVA1matched['MAG_AUTO_I'] < 23.0 ) & #set mag limit based on completeness threshold - everything with a magnitude numerically SMALLER/physically BRIGHTER than limit. this value gives the FAINTEST detectable mag
                  (SVA1matched['MAG_AUTO_R'] > 23.8)] #magnitude limit of 10σ. we want detections to have LARGER magnitude/be BRIGHTER than this value to remove objects that are just noise]

COSMOS_cc = COSMOSmatched.loc[(COSMOSmatched['HSC_r_MAG_AUTO'] - COSMOSmatched['HSC_i_MAG_AUTO'] > 1.2) &
                        (COSMOSmatched['HSC_r_MAG_AUTO'] - COSMOSmatched['HSC_i_MAG_AUTO'] > 1.5*(COSMOSmatched['HSC_i_MAG_AUTO'] - COSMOSmatched['HSC_z_MAG_AUTO']) + 1) &
                        (COSMOSmatched['HSC_i_MAG_AUTO'] - COSMOSmatched['HSC_z_MAG_AUTO'] < 0.7) &
                        (COSMOSmatched['HSC_i_MAG_AUTO'] < 26.46) &
                        (COSMOSmatched['HSC_r_MAG_AUTO'] > 26.84)]

COSMOScc_rmini = COSMOS_cc['HSC_r_MAG_AUTO'] - COSMOS_cc['HSC_i_MAG_AUTO']
COSMOScc_iminz = COSMOS_cc['HSC_i_MAG_AUTO'] - COSMOS_cc['HSC_z_MAG_AUTO']

# %% plot SVA1 color-selection

fig, ax = plt.subplots(1,1)
anchored_text = AnchoredText("object total = {} \n area ~3.5 sq deg".format(len(SVA1_cc)), loc = 'lower left')
ax.add_artist(anchored_text)

plt.scatter(SVA1matched['MAG_AUTO_I']-SVA1matched['MAG_AUTO_Z'], SVA1matched['MAG_AUTO_R']-SVA1matched['MAG_AUTO_I'], alpha=0.7, s=0.5)
plt.scatter(SVA1_cc['MAG_AUTO_I']-SVA1_cc['MAG_AUTO_Z'], SVA1_cc['MAG_AUTO_R']-SVA1_cc['MAG_AUTO_I'], alpha=0.9, color = 'goldenrod', s=4, label = 'r-band dropouts')

def yvals(m,xval,intercept):
    return(m*xval + intercept)
xvals = SVA1matched['MAG_AUTO_I']-SVA1matched['MAG_AUTO_Z']

def xint(m,yval,intercept):
    return((yval-1)/m)

yintval = yvals(1.5,0.7,1)
xintval = xint(1.5, 1.2, 1)

xtrunc = xvals[(xvals>xintval) & (xvals<0.7)]

plt.plot(xtrunc, yvals(1.5, xtrunc, 1.0), linestyle = "--", color = 'black')
plt.hlines(y=1.2, xmin = -15, xmax = xintval, linestyle = "--", color = 'black')
plt.vlines(x=0.7, ymin = yintval, ymax = 20, linestyle = "--", color = 'black')

plt.xlabel('i - z')
plt.ylabel('r - i')
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.title('SVA1 r-band dropout selection')
plt.legend(loc = 'upper left')

#%% plot COSMOS color-selection
fig, ax = plt.subplots(1,1)
cosmos_text = AnchoredText("object total = {}".format(len(COSMOS_cc)), loc = 'lower left')
ax.add_artist(cosmos_text)

plt.scatter(COSMOSmatched['HSC_i_MAG_AUTO'] - COSMOSmatched['HSC_z_MAG_AUTO'], COSMOSmatched['HSC_r_MAG_AUTO'] - COSMOSmatched['HSC_i_MAG_AUTO'], alpha=0.7, s=0.5)
plt.scatter(COSMOScc_iminz, COSMOScc_rmini, color='goldenrod', alpha=0.9, label='r-band dropouts', s=2)


cosmosxvals = COSMOSmatched['HSC_i_MAG_AUTO'] - COSMOSmatched['HSC_z_MAG_AUTO']

cosmosxtrunc = cosmosxvals[(cosmosxvals>xintval) & (cosmosxvals<0.7)]

plt.plot(cosmosxtrunc, yvals(1.5, cosmosxtrunc, 1.0), linestyle = "--", color = 'black')
plt.hlines(y=1.2, xmin = -6, xmax = xintval, linestyle = "--", color = 'black')
plt.vlines(x=0.7, ymin = yintval, ymax = 10, linestyle = "--", color = 'black')
plt.xlabel('i - z')
plt.ylabel('r - i')
plt.title('COSMOS r-band dropout selection')
plt.legend(loc = 'upper left')

# %% output eazy catalog for SVA1

SVA1ezcat = SVA1_cc[['RA', 'DEC', 'MAG_AUTO_G', 'MAGERR_AUTO_G', 'MAG_AUTO_R',
                     'MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I',
                     'MAG_AUTO_Z', 'MAGERR_AUTO_Z', 'z_spec']]

SVA1ezcat = SVA1ezcat.replace(99, -99)
SVA1ezcat['z_spec'] = SVA1ezcat.z_spec.replace(np.nan, -1)

# create id col
SVA1ezcat.insert(0, 'id', range(1, len(SVA1ezcat)+1))

os.environ['EAZYINPUTS'] = '/Users/bflucero/eazy-photoz/inputs'
sva1catname = os.path.join(os.getenv('EAZYINPUTS'), 'SVA1ez.cat')

# np.savetxt(sva1catname, SVA1ezcat.values, fmt=' '.join(['%i'] + ['%0.3f']*2 + ['%.05e']*8 + ['%i']),
#            header="id RA DEC MAG_AUTO_G MAGERR_AUTO_G MAG_AUTO_R MAGERR_AUTO_R MAG_AUTO_I MAGERR_AUTO_I MAG_AUTO_Z MAGERR_AUTO_Z z_spec")

# %% output eazy catalog for COSMOS

COSMOSezcat = COSMOS_cc[['ALPHA_J2000', 'DELTA_J2000', 'HSC_g_MAG_AUTO',
                         'HSC_g_MAGERR_AUTO', 'HSC_r_MAG_AUTO',
                         'HSC_r_MAGERR_AUTO', 'HSC_i_MAG_AUTO',
                         'HSC_i_MAGERR_AUTO', 'HSC_z_MAG_AUTO',
                         'HSC_z_MAGERR_AUTO', 'HSC_y_MAG_AUTO',
                         'HSC_y_MAGERR_AUTO']]

COSMOSezcat = COSMOSezcat.replace(np.nan, -99)
COSMOSezcat.insert(0, 'id', range(1, len(COSMOSezcat)+1))

os.environ['EAZYINPUTS'] = '/Users/bflucero/eazy-photoz/inputs'
cosmoscatname = os.path.join(os.getenv('EAZYINPUTS'), 'COSMOSez.cat')

# np.savetxt(cosmoscatname, COSMOSezcat.values, fmt=' '.join(['%i'] + ['%0.3f']*2 + ['%.05e']*10),
#            header='id RA DEC HSC_g_MAG_AUTO HSC_g_MAGERR_AUTO HSC_r_MAG_AUTO HSC_r_MAGERR_AUTO HSC_i_MAG_AUTO HSC_i_MAGERR_AUTO HSC_z_MAG_AUTO HSC_z_MAGERR_AUTO HSC_y_MAG_AUTO HSC_y_MAGERR_AUTO')





