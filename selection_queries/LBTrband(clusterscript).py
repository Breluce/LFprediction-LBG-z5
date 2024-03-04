#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 19:48:11 2022

@author: bflucero
"""
#%%
import sys
import easyaccess as ea
import h5py
import hdf5plugin
import tables
from astropy.table import Table
import numpy as np
import pandas as pd
import nexusformat.nexus as nx

# %% read h5 file

# f = h5py.File(sys.argv[1])
# f = h5py.File('/Volumes/Samsung_T5/Y6A2_0_01sample.h5')
f = nx.nxload('/Volumes/Samsung_T5/Y6A2_0_01sample.h5')
# t = Table.read(f['data/table'])
# f.close()

# t = t['COADD_OBJECT_ID', 'RA', 'DEC', 'MAG_AUTO_G', 'MAGERR_AUTO_G',
#         'MAG_AUTO_R', 'MAGERR_AUTO_R', 'MAG_AUTO_I', 'MAGERR_AUTO_I',
#         'MAG_AUTO_Z', 'MAGERR_AUTO_Z', 'MAG_AUTO_Y', 'MAGERR_AUTO_Y']


#%% #Query DES Catalog
# connection = ea.connect()
# query = 'SELECT coadd_object_id, ra, dec, mag_auto_g, mag_auto_r, mag_auto_i, mag_auto_z, mag_auto_y, magerr_auto_g, magerr_auto_r, magerr_auto_i, magerr_auto_z, magerr_auto_y, tilename FROM y6_gold_2_0 WHERE rownum < 10;' #test first 10 rows query

# #create dataframe of catalog query
# Y3GOLD = connection.query_to_pandas(query)
# pd.set_option('display.max_rows',12) #Display only

# connection.close()

# #color-selection criteria for r-band dropouts
# col_sel = Y3GOLD.loc[(Y3GOLD.mag_auto_r - Y3GOLD.mag_auto_i > 1.2) & (Y3GOLD.mag_auto_r - Y3GOLD.mag_auto_i < 25) & 
#                       (Y3GOLD.mag_auto_r - Y3GOLD.mag_auto_i > (1.5*(Y3GOLD.mag_auto_i - Y3GOLD.mag_auto_z) + 1.0)) & 
#                       (Y3GOLD.mag_auto_i - Y3GOLD.mag_auto_z < 0.7) & (Y3GOLD.mag_auto_i - Y3GOLD.mag_auto_z > -25) &
#                       (Y3GOLD.mag_auto_i < 24.0) & (Y3GOLD.mag_auto_r > 24.3)]

# %% define dictionary of catalog columns needed + convert byteorder
# d = {'coadd_object_id':gcat['coadd_object_id'], 'ra':gcat['ra'],
#      'dec':gcat['dec'],  'g':gcat['sof_cm_mag_corrected_g'],
#      'i':gcat['sof_cm_mag_corrected_i'], 'r':gcat['sof_cm_mag_corrected_r'],
#      'z':gcat['sof_cm_mag_corrected_z'], 'err_g':gcat['sof_cm_mag_err_g'],
#      'err_i':gcat['sof_cm_mag_err_i'], 'err_r':gcat['sof_cm_mag_err_r'],
#      'err_z':gcat['sof_cm_mag_err_z'], 'tilename':gcat['tilename']}

# d.update({key:np.array(x).byteswap().newbyteorder() for key,x in d.items()})

