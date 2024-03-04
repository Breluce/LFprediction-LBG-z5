#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:18:59 2021

@author: bflucero
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib.offsetbox import AnchoredText
from astropy.coordinates import SkyCoord
from astropy import units as u

#%%
%%time
Y3gold = pd.read_hdf('GOODS-SOUTH.h5')

Y3ccloose = Y3gold.loc[(Y3gold['MAG_AUTO_R'] - Y3gold['MAG_AUTO_I'] > 1.5) &
                  #(Y3gold['MAG_AUTO_I'] < 22) &
                  (Y3gold['MAG_AUTO_R'] - Y3gold['MAG_AUTO_I'] < 25) & #remove null vals
                  (Y3gold['MAG_AUTO_I'] - Y3gold['MAG_AUTO_Z'] < 0) &
                  (Y3gold['MAG_AUTO_I'] - Y3gold['MAG_AUTO_Z'] > -40)& #remove null vals
                  (Y3gold['MAG_AUTO_I'] < 24.0 )& #set mag limit based on completeness threshold
                  (Y3gold['MAG_AUTO_R'] > 24.3), #magnitude limit of 10σ
                  ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'MAG_AUTO_Y', 'RA', 'DEC',
                   'MAGERR_AUTO_G', 'MAGERR_AUTO_R', 'MAGERR_AUTO_I', 'MAGERR_AUTO_Z', 'MAGERR_AUTO_Y']]

#obj_idx = np.array(Y3ccloose.index) #an array of indices for all the candidates - will be used below for fitting phot points

#%%
#color color plot of r-band droputs
fig, ax = plt.subplots(1,1)
anchored_text = AnchoredText("object total = {}".format(len(Y3ccloose)), loc = 'upper left')
ax.add_artist(anchored_text)
plt.scatter(Y3gold['MAG_AUTO_I']-Y3gold['MAG_AUTO_Z'], Y3gold['MAG_AUTO_R']-Y3gold['MAG_AUTO_I'], alpha=0.7)
plt.scatter(Y3ccloose['MAG_AUTO_I']-Y3ccloose['MAG_AUTO_Z'], Y3ccloose['MAG_AUTO_R']-Y3ccloose['MAG_AUTO_I'], alpha=0.75, color =  'tan', label = 'r-band dropouts')
plt.axhline(y=1.5, linestyle = "--", c = 'C3')
plt.axvline(x=0, linestyle = "--", c = 'C3')
plt.xlabel('i - z')
plt.ylabel('r - i')
plt.title('DES0332-2749 r-band droput selection')
plt.legend()
#plt.show()

#%% i vs r-i, i vs i-z
fig15, ax15 = plt.subplots(1,2)

rmini_cut = Y3gold.loc[(Y3gold['MAG_AUTO_R'] - Y3gold['MAG_AUTO_I'] > 1.5)]
iminz_cut = Y3gold.loc[(Y3gold['MAG_AUTO_I'] - Y3gold['MAG_AUTO_Z'] < 0)]

ax15[0].scatter(Y3ccloose['MAG_AUTO_I'], Y3ccloose['MAG_AUTO_R']-Y3ccloose['MAG_AUTO_I'], alpha=1, zorder=3, c = 'orange', label = 'i - z > 0')
ax15[0].scatter(rmini_cut['MAG_AUTO_I'], rmini_cut['MAG_AUTO_R']-rmini_cut['MAG_AUTO_I'], c = 'gray', zorder=2, label = 'r - i > 1.5')
ax15[0].scatter(Y3gold['MAG_AUTO_I'], Y3gold['MAG_AUTO_R']-Y3gold['MAG_AUTO_I'], alpha=0.2, c = 'blue', label = 'DES 0332-2749')
ax15[0].set(xlabel = 'i')
ax15[0].set(ylabel = 'r - i')
ax15[0].axhline(y=1.5, linestyle = "--", c = 'C3')
ax15[0].legend()
ax15[1].scatter(Y3ccloose['MAG_AUTO_I'], Y3ccloose['MAG_AUTO_I']-Y3ccloose['MAG_AUTO_Z'], alpha=1, zorder=3, c = 'orange', label = 'r - i > 1.5')
ax15[1].scatter(Y3gold['MAG_AUTO_I'], Y3gold['MAG_AUTO_I']-Y3gold['MAG_AUTO_Z'], alpha=0.2, c = 'blue', label =  'DES 0332-2749')
ax15[1].scatter(iminz_cut['MAG_AUTO_I'], iminz_cut['MAG_AUTO_I']-iminz_cut['MAG_AUTO_Z'], c = 'gray', zorder=2, label = 'i - z < 0')
ax15[1].set(xlabel = 'i')
ax15[1].set(ylabel = 'i - z')
ax15[1].axhline(y=0, linestyle = "--", c = 'C3')
ax15[1].legend()
ax15[0].invert_xaxis()
ax15[1].invert_xaxis()
plt.show()

#%%

#TO OUTPUT A CATALOG TO RUN IN EAZY W/O MATCHING TO CANDELS (NO SPEC Z'S ADDED)

Y3ccall_ezcat = Y3ccloose[['MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y']]

Y3ccall_ezcat = Y3ccall_ezcat.replace(99,-99)

#create id col
Y3ccall_ezcat.insert(0, 'id', range(1, len(Y3ccall_ezcat)+1))

np.savetxt('./Y3rccall_ezcat.txt', Y3ccall_ezcat.values, fmt=' '.join(['%i'] + ['%.05e']*10), 
                     header = "id, MAG_AUTO_G, MAGERR_AUTO_G, MAG_AUTO_R,MAGERR_AUTO_R, MAG_AUTO_I, MAGERR_AUTO_I, MAG_AUTO_Z, MAGERR_AUTO_Z, MAG_AUTO_Y, MAGERR_AUTO_Y" )

#%%

#read in fit from above cell

photz = open("/Users/bflucero/eazy-photoz/inputs/OUTPUT/photzDESr.zout", "r")

allpzcols = ['id', 'z_spec', 'z_1', 'z_m1', 'chi_1', 'temp_1', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']
#['id', 'z_spec', 'z_1', 'z_m1', 'chi_1', 'temp_1', 'z_p', 'chi_p', 'temp_p', 'z_m2', 'odds', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt','q_z', 'z_peak', 'peak_prob', 'z_mc']
#['id', 'z_spec', 'z_a', 'z_m1', 'chi_a', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']

allphotz_df = pd.read_csv("/Users/bflucero/eazy-photoz/inputs/OUTPUT/photzDESr.zout", sep = '\s+', names = allpzcols, comment="#", header = None)
                       
allphotz_df["RA"] = np.array(Y3ccloose.RA.values)    
allphotz_df["DEC"] = np.array(Y3ccloose.DEC.values)    

#see if there are any objects predicted to be z>5 (r-band dropout)

r_bp = allphotz_df.loc[(allphotz_df.z_1 >= 4.9), ['id','z_1','chi_1', 'RA', 'DEC']]
r_bp_wchi = allphotz_df.loc[(allphotz_df.z_1 >= 4.9) & (allphotz_df.chi_1 <= 1), ['id','z_1','chi_1', 'RA','DEC']]

#for thumbnails
rbp_pos = r_bp[['RA', 'DEC']]
rbp_pos.to_csv('positions_allrband_ccrun.csv', index = False, header = ['RA', 'DEC'])

#%%
#get file with EAZY objects, mags, and positions for thumbnail annotations

allphotz_df['MAG_AUTO_G'] = np.array(Y3ccall_ezcat['MAG_AUTO_G'])
allphotz_df['MAGERR_AUTO_G'] = np.array(Y3ccall_ezcat['MAGERR_AUTO_G'])
allphotz_df['MAG_AUTO_R'] = np.array(Y3ccall_ezcat['MAG_AUTO_R'])
allphotz_df['MAGERR_AUTO_R'] = np.array(Y3ccall_ezcat['MAGERR_AUTO_R'])
allphotz_df['MAG_AUTO_I'] = np.array(Y3ccall_ezcat['MAG_AUTO_I'])
allphotz_df['MAGERR_AUTO_I'] = np.array(Y3ccall_ezcat['MAGERR_AUTO_I'])
allphotz_df['MAG_AUTO_Z'] = np.array(Y3ccall_ezcat['MAG_AUTO_Z'])
allphotz_df['MAGERR_AUTO_Z'] = np.array(Y3ccall_ezcat['MAGERR_AUTO_Z'])
allphotz_df['MAG_AUTO_Y'] = np.array(Y3ccall_ezcat['MAG_AUTO_Y'])
allphotz_df['MAGERR_AUTO_Y'] = np.array(Y3ccall_ezcat['MAGERR_AUTO_Y'])

ezinfo = allphotz_df.loc[(allphotz_df.z_1 >= 4.9)]

ezinfo.to_csv('rband_ezinfo.csv', index = False, 
           header = ['id', 'z_spec', 'z_1', 'z_m1', 'chi_1', 'temp_1', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc', 'RA', 'DEC', 'MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y'])
              
#%%

#read in EAZY for single template fit w SED 2

allphotz_df_SED2 = pd.read_csv("/Users/bflucero/eazy-photoz/inputs/OUTPUT/photzDESrSED2.zout", sep = '\s+', names = allpzcols, comment="#", header = None)
                               
allphotz_df_SED2["RA"] = np.array(Y3ccloose.RA.values)    
allphotz_df_SED2["DEC"] = np.array(Y3ccloose.DEC.values)    

allphotz_df_SED2 = allphotz_df_SED2.join(Y3ccloose[['MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y']])

SED2_ezinfo = allphotz_df_SED2.loc[(allphotz_df_SED2.z_1 >= 4.9)]

SED2_ezinfo.to_csv('rbandSED2_ezinfo.csv', index = False, 
           header = ['id', 'z_spec', 'z_1', 'z_m1', 'chi_1', 'temp_1', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc', 'RA', 'DEC', 'MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y'])

#%%
#read in candels gds catalog
with fits.open('candelsgds.fits') as data:
    d = data[1].data
    dswap = Table(np.asarray(d).byteswap().newbyteorder('=')) #swap from big endian to little endian byte
    candels_cat = dswap.to_pandas()
    
#read in HST
with fits.open('hlsp_hlf_hst_60mas_goodss_v2.1_catalog.fits') as dat:
    d = dat[1].data
    swap = Table(np.asarray(d).byteswap().newbyteorder('='))
    HST_cat = swap.to_pandas()
    
#%%
    
#plot DES/HST/CANDELS overlap
fig2, ax2 = plt.subplots(1,1)
plt.scatter(candels_cat.RAdeg.values, candels_cat.DECdeg.values, s=.5, alpha = .25, color ='grey', label = 'CANDELS objects')
plt.scatter(Y3ccloose.RA.values, Y3ccloose.DEC.values,  s=25, color = 'tan', marker = '*', label = 'DES r-band dropout selection')
#plt.scatter(HST_cat.ra.values, HST_cat.dec.values, s=1, color = 'tan', alpha = .02, zorder = 0, label = 'HST goodss')
plt.xlabel('RA')
plt.ylabel('DEC')
plt.legend(framealpha = 1, fancybox=True, markerscale = 1)

#%%

RAmin = np.min(candels_cat.RAdeg.values)
RAmax = np.max(candels_cat.RAdeg.values)
DECmin = np.min(candels_cat.DECdeg.values)
DECmax = np.max(candels_cat.DECdeg.values)

#approximate number of objects overlapping the GDS region
approx_ct = len(Y3ccloose.loc[(Y3ccloose.RA.values < RAmax) & (Y3ccloose.RA.values > RAmin) &
                                                    (Y3ccloose.DEC.values < DECmax) & (Y3ccloose.RA.values > DECmin)])

#%%
#get spec z for each object in Y3ccloose from candels gds

#find nearest object for each g-band dropout in candels

ra1 = np.array(Y3ccloose.RA.values)
dec1 = np.array(Y3ccloose.DEC.values)

ra2 = np.array(candels_cat.RAdeg.values)
dec2 = np.array(candels_cat.DECdeg.values)

c = SkyCoord(ra = ra1*u.degree, dec = dec1*u.degree)
catalog = SkyCoord(ra = ra2*u.degree, dec = dec2*u.degree)

#specify the maximum separation between objects in order to be considered a matchß
maxsep = 0.5*u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraint = d2d < maxsep
c_matches = c[sep_constraint]
catalog_matches = catalog[idx[sep_constraint]]
#to access individual values: catalog[idx].ra or catalog[idx].dec

#%%

#plot the DES matches over the candels catalog
fig3, ax3 = plt.subplots(1,1)
plt.scatter(candels_cat.RAdeg.values, candels_cat.DECdeg.values, s=.5, alpha = .15, color ='grey', label = 'CANDELS objects')
plt.scatter(Y3ccloose.RA.values, Y3ccloose.DEC.values,  s=25, color = 'tan', marker = '*', label = 'DES r-band dropout selection')
plt.scatter(catalog_matches.ra, catalog_matches.dec, s = 70, color = 'red', marker = "x", label = 'DES rband w/ CANDELS match \n separation constraint = 0.5"')
#plt.scatter(HST_cat.ra.values, HST_cat.dec.values, s=.05, color = 'slategrey', alpha = .3, zorder = 0, label = 'HST goodss')
plt.xlabel('RA')
plt.ylabel('DEC')
plt.legend(framealpha = 1, fancybox=True, markerscale = 1)

#%%

#specz values need to be reindexed by ascending RA value BEFORE adding to DES overlap ----- DONE
#might need to rerun EAZY fit ----- DONE

#get spec z of objects with matches
specz = candels_cat.loc[(candels_cat.RAdeg.isin(catalog_matches.ra)) & (candels_cat.DECdeg.isin(catalog_matches.dec)), ['RAdeg', 'DECdeg','zspec']]
specz = specz.sort_values(by=['RAdeg'])

#%%
DESoverlap = Y3ccloose.loc[(Y3ccloose.RA.isin(c_matches.ra)) & (Y3ccloose.DEC.isin(c_matches.dec))]
DESoverlap = DESoverlap.sort_values(by=['RA'])
#%%
DESoverlap["z_spec"] = np.array(specz['zspec'].values) #required col name for EAZY
DESoverlap["RAdeg"] = np.array(specz['RAdeg'].values)
DESoverlap["DECdeg"] = np.array(specz['DECdeg'].values)

#create id col
DESoverlap.insert(0, 'id', range(1,len(DESoverlap)+1))

#of the DES objects predicted to be above z=4, which ones are in the candels field if any? -> need EAZY catalog of only matches
#use this catalog to determine best template to use in EAZY
#of the objects detected in both DES and candels, what are the predicted z values? -> need EAZY catalog of all DES objects
#%%
#output catalog of object for EAZY SED fitting code (mags + specz only) for DES g-band objects w candels match
#USE THIS TO DETERMINE THE BEST FIT TEMPLATE AND LATER GRAB THUMBNAILS FOR CONTOUR PLOTS (THESE OBJ ONLY)
rbandmatches_ezcat = DESoverlap[['id', 'MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y', 'z_spec']]

rbandmatches_ezcat = rbandmatches_ezcat.replace(99, -99)

np.savetxt('./GScat-rbandmatches.txt', rbandmatches_ezcat.values, fmt=' '.join(['%i'] + ['%.05e']*11), 
                     header = "id, MAG_AUTO_G, MAGERR_AUTO_G, MAG_AUTO_R,MAGERR_AUTO_R, MAG_AUTO_I, MAGERR_AUTO_I, MAG_AUTO_Z, MAGERR_AUTO_Z, MAG_AUTO_Y, MAGERR_AUTO_Y, z_spec" )

#add header with column names, copy output to EAZY folder
#check right file is listed in param file
#modify param files in templates as needed
#command: ../src/eazy -p zphot.param

#%%
#convert magnitudes to flux (change parameters in the zparam file!!)

#def mag_to_flux(mag):     
#    f_nu = ( 10 ** ((mag - 8.9 ) / -2.5 ))  #Jy
#    f_nu *= 10e6 #microJy
#    return(f_nu)
#    
#rbandmatches_mags = rbandmatches_ezcat[['MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
#                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y']].apply(mag_to_flux)
#
#rbandmatches_mags["id"] = rbandmatches_ezcat.id
#rbandmatches_mags["z_spec"] = rbandmatches_ezcat.z_spec
#
#gbandmag_UL = 23.8
#rbandmag_UL = 24.6
#
#gbandf_UL = (mag_to_flux(gbandmag_UL)/10)*3
#rbandf_UL = (mag_to_flux(rbandmag_UL)/10)*3
#
#rband_magcut = rbandmatches_mags.loc[(rbandmatches_mags.MAG_AUTO_G < gbandf_UL) & (rbandmatches_mags.MAG_AUTO_R < rbandmag_UL)]
        

#gbandmatches_ezcat.insert(0, 'id', range(1,len(gbandmatches_ezcat)+1))
#np.savetxt('GScat-rbandmatches_mags.txt', rbandmatches_mags.values, fmt=' '.join(['%i'] + ['%.05e']*11), 
#                     header = "id, MAG_AUTO_G, MAGERR_AUTO_G, MAG_AUTO_R,MAGERR_AUTO_R, MAG_AUTO_I, MAGERR_AUTO_I, MAG_AUTO_Z, MAGERR_AUTO_Z, MAG_AUTO_Y, MAGERR_AUTO_Y, z_spec" )

#%%

#extra output file for positions
#matchpos = DESoverlap[["RA","DEC"]]
#matchpos = matchpos.reindex()
#matchpos.to_csv('GScat-gbandmatchespos.csv',  index = False, header = ['RA', 'DEC'])

#%%
#output catalog of object for EAZY SED fitting code (mags + specz only) for ALL DES g-band objects
#gband_ezcat = Y3ccloose[['MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
#                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y']]
#
#gband_ezcat = gband_ezcat.replace(99, -99)
#gband_ezcat.insert(0, 'id', range(1,len(gband_ezcat)+1))
#np.savetxt('GScat-gband.txt', gband_ezcat.values, fmt=' '.join(['%i'] + ['%.05e']*10),
#           header = "id, MAG_AUTO_G, MAGERR_AUTO_G, MAG_AUTO_R,MAGERR_AUTO_R, MAG_AUTO_I, MAGERR_AUTO_I, MAG_AUTO_Z, MAGERR_AUTO_Z, MAG_AUTO_Y, MAGERR_AUTO_Y" )

#add header with column names, copy output to EAZY folder
#check right file is listed in param file
#command: ../src/eazy -p zphot.param

#%%
#read in EAZY photo z output file

photz = open("/Users/bflucero/eazy-photoz/inputs/OUTPUT/photz.zout", "r")

pzcols =  ['id', 'z_spec', 'z_a', 'z_m1', 'chi_a', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']
#2temp fit
#['id', 'z_spec', 'z_2', 'z_m1', 'chi_2', 'temp2a', 'temp_2b', 'z_p', 'chi_p', 'temp_pa', 'temp_pb', 'z_m2', 'odds', 'l68', 'u68',  'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']
#2temp fit no prior
# ['id', 'z_spec', 'z_2', 'z_m1', 'chi_2', 'temp2a', 'temp_2b', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']
#1temp fit no prior
#['id', 'z_spec', 'z_1', 'z_m1', 'chi_1', 'temp_1', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', ' nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']
#all temp fit, w prior
#['id', 'z_spec', 'z_a', 'z_m1', 'chi_a', 'z_p', 'chi_p', 'z_m2', 'odds', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99', 'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']

photz_df = pd.read_csv("/Users/bflucero/eazy-photoz/inputs/OUTPUT/photz.zout", sep = '\s+', names = pzcols, comment="#", header = None)
photz_df["RA"] = np.array(DESoverlap.RA.values)
photz_df["DEC"] = np.array(DESoverlap.DEC.values)

#%%
from scipy.stats import pearsonr

#check pred z vs z_spec values
#'z_2', 'z_m1', 'z_p', 'z_m2', 'z_peak', 'z_mc'

fig, axes = plt.subplots(2,2)
fig.suptitle('SPEC Z vs PHOTO Z (BR07 Templates) \n multi template fit')

#grab all values for obj with a specz 
z_df = photz_df.loc[(photz_df.z_spec != -99)]
ylower = z_df.l68
yupper = z_df.u68

#z_1
#m_z1, b_z1 = np.polyfit(z_df.z_spec, z_df.z_1, 1)
#r_z1, pval_z1 = pearsonr(z_df.z_spec, z_df.z_1)
#axes[0,0].scatter(z_df.z_spec, z_df.z_1)
#axes[0,0].plot(z_df.z_spec, m_z1*z_df.z_spec + b_z1)
#axes[0,0].set(xlabel='CANDELS z_spec')
#axes[0,0].set(ylabel = 'z_1')
#axes[0,0].text(3,4,'r = {:.3f}'.format(r_z1))
#axes[0,0].axis([0 ,6, 0, 6])
#axes[0,0].set_aspect('equal', 'box')

#z_2
#m_z2, b_z2 = np.polyfit(z_df.z_spec, z_df.z_2, 1)
#r_z2, pval_z2 = pearsonr(z_df.z_spec, z_df.z_2)
#axes[0,0].scatter(z_df.z_spec, z_df.z_2)
#axes[0,0].plot(z_df.z_spec, m_z2*z_df.z_spec + b_z2)
#axes[0,0].set(xlabel='CANDELS z_spec')
#axes[0,0].set(ylabel = 'z_2')
#axes[0,0].text(3,4,'r = {:.3f}'.format(r_z2))
#ylower = z_df.l68
#yupper = z_df.u68
#z2_yerr = [z_df.z_2 - ylower, yupper - z_df.z_2]
#axes[0,0].errorbar(z_df.z_spec, z_df.z_2, yerr = z2_yerr, ls = '', capsize = 5, label = '$1 \sigma$')
#axes[0,0].axis([0 ,6, 0, 6])
#axes[0,0].set_aspect('equal', 'box')

#z_a
m_za, b_za = np.polyfit(z_df.z_spec, z_df.z_a, 1)
r_za, pval_za = pearsonr(z_df.z_spec, z_df.z_a)
axes[0,0].scatter(z_df.z_spec, z_df.z_a)
axes[0,0].plot(z_df.z_spec, m_za*z_df.z_spec + b_za)
axes[0,0].text(2,5,'r = {:.3f}'.format(r_za))
axes[0,0].set(xlabel='CANDELS z_spec')
axes[0,0].set(ylabel = 'z_a')
axes[0,0].axis([0,7,0,7])
axes[0,0].set_aspect('equal', 'box')

#z_m1
m_zm1, b_zm1 = np.polyfit(z_df.z_spec, z_df.z_m1, 1)
r_zm1, pval_zm1 = pearsonr(z_df.z_spec, z_df.z_m1)
axes[0,1].scatter(z_df.z_spec, z_df.z_m1)
axes[0,1].plot(z_df.z_spec, m_zm1*z_df.z_spec + b_zm1)
axes[0,1].text(2,5,'r = {:.3f}'.format(r_zm1))
m1_yerr = [z_df.z_m1 - ylower, yupper - z_df.z_m1]
axes[0,1].errorbar(z_df.z_spec, z_df.z_m1, yerr = m1_yerr, ls = '', capsize = 5, label = '$1 \sigma$')
axes[0,1].set(xlabel='CANDELS z_spec')
axes[0,1].set(ylabel = 'z_m1')
axes[0,1].axis([0,7,0,7])
axes[0,1].set_aspect('equal', 'box')
axes[0,1].legend()

#z_p
#m_zp, b_zp = np.polyfit(z_df.z_spec, z_df.z_p,  1)
#r_zp, pval_zp = pearsonr(z_df.z_spec, z_df.z_p)
#axes[0,2].scatter(z_df.z_spec, z_df.z_p)
#axes[0,2].plot(z_df.z_spec, m_zp*z_df.z_spec + b_zp)
#axes[0,2].text(1,4,'r = {:.3f}'.format(r_zp))
#axes[0,2].set(xlabel='CANDELS z_spec')
#axes[0,2].set(ylabel = 'z_p')
#axes[0,2].axis([0 ,5, 0, 5])
#axes[0,2].set_aspect('equal', 'box')

#
#axes[1,0].plot(z_df.z_spec, m_zm2*z_df.z_spec + b_zm2)
#axes[1,0].text(1.2,4,'r = {:.3f}'.format(r_zm2))
#axes[1,0].set(xlabel='CANDELS z_spec')
#axes[1,0].set(ylabel = 'z_m2')
#axes[1,0].axis([0 ,5, 0, 5])
#axes[1,0].set_aspect('equal', 'box')

#z_peak
m_zpeak, b_zpeak = np.polyfit(z_df.z_spec, z_df.z_peak, 1)
r_zpeak, pval_zpeak = pearsonr(z_df.z_spec, z_df.z_peak)
axes[1,0].scatter(z_df.z_spec, z_df.z_peak)
axes[1,0].plot(z_df.z_spec, m_zpeak*z_df.z_spec+ b_zpeak)
axes[1,0].text(2,5,'r = {:.3f}'.format(r_zpeak))
axes[1,0].set(xlabel='CANDELS z_spec')
axes[1,0].set(ylabel = 'z_peak')
axes[1,0].axis([0,7,0,7])
axes[1,0].set_aspect('equal', 'box')

#z_mc
m_zmc, b_zmc = np.polyfit(z_df.z_spec, z_df.z_mc, 1)
r_zmc, pval_zmc = pearsonr(z_df.z_spec, z_df.z_mc)
axes[1,1].scatter(z_df.z_spec, z_df.z_mc)
axes[1,1].plot(z_df.z_spec, m_zmc*z_df.z_spec + b_zmc)
axes[1,1].text(2,5,'r = {:.3f}'.format(r_zmc))
axes[1,1].set(xlabel = 'CANDELS z_spec')
axes[1,1].set(ylabel = 'z_mc')
axes[1,1].axis([0,7,0,7])
axes[1,1].set_aspect('equal', 'box')

plt.show()
#%%

#see if there are any objects predicted to be z>5 (5-band dropout)
#r_prior = photz_df.loc[(photz_df.z_p >= 4), ['id','z_p','chi_p', 'RA', 'DEC']]

r_bp = photz_df.loc[(photz_df.z_a >= 4.7), ['id','z_a','chi_a', 'RA', 'DEC']]
r_bp_wchi = photz_df.loc[(photz_df.z_a >= 4.7) & (photz_df.chi_a <= 1), ['id','z_spec','z_a','chi_a', 'RA','DEC']]

r_zmc = photz_df.loc[(photz_df.z_mc >= 4.7), ['id','z_mc', 'RA', 'DEC']]

#for thumbnails
#rbp_bp_pos = r_bp[['RA', 'DEC']]
#rbp_bp_pos.to_csv('rband_ezpos.csv', index = False, header = ['RA', 'DEC'])

#%%

#get file with EAZY objects, mags, and positions for thumbnail annotations

photz_df['MAG_AUTO_G'] = np.array(DESoverlap['MAG_AUTO_G'])
photz_df['MAG_AUTO_R'] = np.array(DESoverlap['MAG_AUTO_R'])
photz_df['MAG_AUTO_I'] = np.array(DESoverlap['MAG_AUTO_I'])
photz_df['MAG_AUTO_Z'] = np.array(DESoverlap['MAG_AUTO_Z'])
photz_df['MAG_AUTO_Y'] = np.array(DESoverlap['MAG_AUTO_Y'])
photz_df['MAGERR_AUTO_G'] = np.array(DESoverlap['MAGERR_AUTO_G'])
photz_df['MAGERR_AUTO_R'] = np.array(DESoverlap['MAGERR_AUTO_R'])
photz_df['MAGERR_AUTO_I'] = np.array(DESoverlap['MAGERR_AUTO_I'])
photz_df['MAGERR_AUTO_Z'] = np.array(DESoverlap['MAGERR_AUTO_Z'])
photz_df['MAGERR_AUTO_Y'] = np.array(DESoverlap['MAGERR_AUTO_Y'])

ezinfo = photz_df.loc[(photz_df.z_a >= 4.7),['id', 'MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 
                                      'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y', 'z_spec', 'z_a', 'chi_a', 'RA', 'DEC']]
#%%
ezinfo.to_csv('rband_ezinfo.csv', index = False, header = ['id', 'MAG_AUTO_G','MAGERR_AUTO_G', 'MAG_AUTO_R','MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I', 'MAG_AUTO_Z','MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y', 'z_spec', 'z_a', 'chi_a','RA', 'DEC'])
#%%
test = pd.read_csv('rband_ezinfo.csv')

















# ----------------- SCRAP CODE ---------------------------

#%%
RAmatch = np.array( Y3ccloose.RA.round(decimals=3).isin(candels_cat.RAdeg.round(decimals=3)))
DECmatch= np.array( Y3ccloose.DEC.round(decimals=3).isin(candels_cat.DECdeg.round(decimals=3)))

#does this ra value exist in candels
for i in range(len(Y3ccloose)):
    #if yes, then get df1id and df2id
    if RAmatch[i] == True and DECmatch[i] == True:
        df1RA = Y3ccloose["RA"][i].round(decimals=3)
        df1DEC = Y3ccloose["DEC"][i].round(decimals=3)
        df2row = candels_cat.loc[(candels_cat.RAdeg.round(decimals=3) == df1RA) & (candels_cat.RAdeg.round(decimals=3) == df1DEC),["RAdeg","DECdeg"]]
        print("RA", df1RA, "DEC", df1DEC, "\n",df2row)
        
#does the dec[df1id] value eq the dec[df2id]2
#if yes get df1id and add to idx list
    
#%%

#check for g-band droupout objects in CANDELS catalog
overlapcoord = eazyobj_magspos.loc[eazyobj_magspos.RA.round(decimals = 3).isin(candels_cat.RAdeg.round(decimals = 3)) & 
                                         eazyobj_magspos.DEC.round(decimals = 3).isin(candels_cat.DECdeg.round(decimals = 3))]
#overlap = candels_cat.loc[(candels_cat.RAdeg.round(decimals = 3).isin(overlapcoord.RA.round(decimals = 3).values))]
    
#%%
    
# get subset from GDS truth catalog
    
#photz_truth = pd.DataFrame(columns = ['RA', 'DEC', 'zphot'])
#photz_truth['RA'] = gds_truthcat.RAdeg.values
#photz_truth['DEC'] = gds_truthcat.DECdeg.values
#photz_truth['zphot'] = gds_truthcat.zphot.values
