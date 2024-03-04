#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:53:09 2020

@author: bflucero
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import aplpy
from tqdm import trange
import glob
# from astropy.nddata.utils import Cutout2D
from matplotlib.colors import LogNorm
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
import numpy as np
import os
from astropy.visualization import make_lupton_rgb

# %%
os.chdir('/Users/bflucero/eazy-photoz/inputs/')

pos = pd.read_csv('cosmosz5pos.csv')

for ra, dec in zip(pos.RA.values, pos.DEC.values):
    rfile = glob.glob('OUTPUT/COSMOS/thumbnails/*_{}_{}_COSMOS.rp.original_psf.v2.fits'.format(ra, dec))
    r = fits.open(rfile[0])
    ifile = glob.glob('OUTPUT/COSMOS/thumbnails/*_{}_{}_COSMOS.ip.original_psf.v2.fits'.format(ra, dec))
    i = fits.open(ifile[0])
    zfile = glob.glob('OUTPUT/COSMOS/thumbnails/*_{}_{}_COSMOS.zp.original_psf.v2.fits'.format(ra, dec))
    z = fits.open(zfile[0])

    fig = plt.figure()
    image = make_lupton_rgb(z[0].data, i[0].data, r[0].data, stretch=20,
                            interpolation='spline16',)
    plt.imshow(image)

    # aplpy.make_rgb_cube([rfile[0], ifile[0], zfile[0]], 'OUTPUT/COSMOS/rgb_cube.fits')
    # aplpy.make_rgb_image('OUTPUT/COSMOS/rgb_cube.fits',
    #                      'OUTPUT/COSMOS/rgb_image.png')


    # f = aplpy.FITSFigure('OUTPUT/COSMOS/rgb_cube.fits', figure=fig)
    # f.show_rgb('OUTPUT/COSMOS/rgb_image.png')
    # f.recenter(ra, dec, radius=.0045)


#%%
fig = plt.figure()
ff = aplpy.FITSFigure('COSMOS.rp.original_psf.v2.fits', figure=fig)


#%%
#load in info for all gband objects
allobj = pd.read_csv('HSTgband_ezinfo.csv')
#%%

#code to get ONE cutout 
fig = plt.figure()
f = aplpy.FITSFigure('hlsp_hlf_hst_acs-60mas_goodss_f850lp_v2.0_sci.fits', figure = fig)
f.show_grayscale()
f.recenter(allobj.RA[0], allobj.DEC[0], radius = .0025)
f.show_markers(allobj.RA[0], allobj.DEC[0],marker='c',s=1100,edgecolor='white',facecolor='white')
text = 'id #{} \n (RA, DEC) = ({:.2f},{:.2f}) \n z_1 = {:.2f} \n chi_1 = {:.2f}'.format( allobj.id[0], allobj.RA[0], allobj.DEC[0], allobj.z_1[0], allobj.chi_1[0])
f.add_label(0.6,0.8, text, size = 'large', color = 'white', stretch = 'ultra-condensed', style = 'oblique', weight = 'semibold', relative = True)
fig.canvas.draw()

#this takes a very long time for the entire F160W filter


#%%
#alternative cutout method

#open HST fits file (entire GDS field)

f = fits.open('hlsp_hlf_hst_acs-60mas_goodss_f850lp_v2.0_sci.fits')
image = f[0].data
wcs = WCS(f[0].header)
f.close()

for i in range(len(allobj)):
    fname = 'HSTtn{}'.format(i)
    size = u.Quantity((5, 5), u.arcsec)
    position = SkyCoord(allobj.RA[i]*u.deg, allobj.DEC[i]*u.deg, frame='icrs')
    pos = skycoord_to_pixel(position, wcs=wcs)
    cutout = Cutout2D(image, pos, size, wcs=wcs)
    text = 'id #{} \n (RA, DEC) = ({:.2f},{:.2f}) \n z_1 = {:.2f} \n chi_1 = {:.2f}'.format( allobj.id[i], allobj.RA[i], allobj.DEC[i], allobj.z_1[i], allobj.chi_1[i])
    plt.figure()
    plt.text(20,20, s=text, c  = 'white', weight = 'bold')
    plt.scatter(40,40, marker='c', s=9000, c = 'white', linewidths=5)
    plt.imshow(cutout.data, cmap = 'gray')
    plt.savefig(fname=fname)
    plt.show()

#size = u.Quantity((5, 5), u.arcsec)
#position = SkyCoord(allobj.RA[0]*u.deg, allobj.DEC[0]*u.deg, frame='icrs')
#pos = skycoord_to_pixel(position, wcs=wcs)
#cutout = Cutout2D(image, pos, size, wcs=wcs)
#plt.figure()
#plt.text(20,20, s=text, c  = 'white', weight = 'bold')
#plt.scatter(40,40, marker='c', s=9000, c = 'white', linewidths=5)
#plt.imshow(cutout.data, cmap = 'gray')
#plt.show()
    
    
#labels = [pixel_to_skycoord(pos, wcs=wcs) for item in pos]
#y = plt.yticks()
#loc = pixel_to_skycoord(x[0], y[0], wcs=wcs)
#raticks = loc.ra*u.deg
#ravals = np.array(raticks.value)
#ralabel = ['%.2f'%float(i) for i in ravals]
#decticks= loc.dec*u.deg
#decvals = np.array(decticks)
#declabel = ['%.2f'%float(i) for i in decvals]
#plt.xticks(x[0], ralabel, fontsize = 8)
#plt.yticks(y[0], declabel)


#for i in trange(len(allobj)):
#    size = u.Quantity((5, 5), u.arcsec)
#    position = SkyCoord(allobj.RA[i]*u.deg, allobj.DEC[i]*u.deg, frame='icrs')
#    pos = skycoord_to_pixel(position, wcs=wcs)
#    cutout = Cutout2D(image, pos, size, wcs=wcs)
#    plt.figure()
#    plt.imshow(cutout.data, cmap = 'gray')#, norm = LogNorm())

#%%

filenames = glob.glob('./HST-GDS-fits/*')

#%%

# --------------- FOR GENERATING NEW FITS FILE EXTENSIONS ----------------------------

#file_r = filenames[0]
#file_g = filenames[1]
#file_b = filenames[2]

file_r = 'DESJ033211.0227-274959.9268_g.fits'
file_g = 'DESJ033211.0227-274959.9268_i.fits'
file_b = 'DESJ033211.0227-274959.9268_z.fits'

r = fits.open(file_r, ignore_missing_end = True)
g = fits.open(file_g, ignore_missing_end = True)
b = fits.open(file_b, ignore_missing_end = True)

#i_mag = i[0].data
#r_mag = r[1].data
#z_mag = z[0].data

#plt.figure(figsize = (8,10))
#plt.imshow(r_mag, cmap="gray", vmin=1, vmax =50)

#%%

def isolate_image_extension(fits_file, extension):
    '''
        Saves the data + header of the specified extension as
        new FITS file

        input
        ------
        fits_file: file path to FITS image
        extension: Number of HDU extension containing the image data
    '''

    header = fits.getheader(fits_file, extension)
    data = fits.getdata(fits_file, extension)

    #fits.writeto('%s_image.fits' % fits_file.strip('./seen/DESJ033334.9526-275048.5484/DESJ033334.9526-275048.5484_'), data, header)
    fits.writeto('%s_image.fits' % fits_file.strip('/Users/bflucero/Desktop/research/DESJ032011.0227-174959.9268_'), data, header)

# Create new FITS files containing only the image extension
for exposure in [file_r, file_g, file_b]:
    isolate_image_extension(exposure, 0)
    
    #must go in and copy/paste names and paths for the aplpy function below

#%%
    
# ---------------- MAKE RGB CUBE OF FITS FILES --------------------------
    
aplpy.make_rgb_cube(['g.fi_image.fits', 
                     'i.fi_image.fits', 
                     'z.fi_image.fits'], 'rgb_cube.fits')

#%%
    
# ---------------- MAKE RGB IMAGE OF FITS FILES ---------------------------
    
aplpy.make_rgb_image('rgb_cube.fits', 'rband-maglim-rgb_image.png', embed_avm_tags=False, 
                      pmin_r=30.0, pmax_r=98.50, pmin_g=30.0, pmax_g=98.50, pmin_b=30.0, pmax_b=98.50)

#%%

# ---------------- READ IN OBJECT INFO FROM FILE GENERATED IN DESGOODSOUTH.PY --------------------------

obj = pd.read_csv('/Volumes/Samsung_T5/research files/code/rband_ezinfo.csv', header = 'infer' )

#%%

# RUN THIS FOR NEW OBJECT THUMBNAILS
#
#for i in trange(len(obj)):
#    fig = plt.figure(figsize=(5,5))
#    f = aplpy.FITSFigure('irz_cube_2d.fits', figure = fig)
#    f.show_rgb('rgb_image.png')
#    f.recenter(obj.RA[i],obj.DEC[i], radius = .0045)
#    f.show_markers(obj.RA[i],obj.DEC[i],marker='c',s=1000,edgecolor='white',facecolor='white')
#    text = '(RA, DEC) = ({:.2f},{:.2f}) \n MAG_AUTO_R - MAG_AUTO_I = {:.2f} \n MAG_AUTO_Z = {:.2f}'.format(obj.RA[i], obj.DEC[i], (obj.mag_r[i] - obj.mag_i[i]), obj.mag_z[i])
#    f.add_label(0.3,0.9, text, size = 'small', color = 'white', stretch = 'ultra-condensed', style = 'oblique', weight = 'semibold', relative = True)
#    fig.canvas.draw()
#    fig.savefig(str(obj.RA[i])+'_'+str(obj.DEC[i])+'_thumbnail.png')
#    f.close()
    
 #%%
 
 #make cutouts for DES objects in HST for each filter F160W, F140W, F125W
 
#testing
fig = plt.figure(figsize=(5,5))
f = aplpy.FITSFigure('rgb_cube_2d.fits', figure = fig)
f.show_rgb('rband-maglim-rgb_image.png')
f.recenter(obj.RA[2], obj.DEC[2], radius = .0045)
f.show_markers(obj.RA[2], obj.DEC[2],marker='c',s=2500,edgecolor='white',facecolor='white', linewidth = 4)
text = '(RA, DEC) = ({:.2f},{:.2f}) \n z_a = {:.2f} \n chi_a = {:.2f}'.format(obj.RA[2], obj.DEC[2], obj.z_a[2], obj.chi_a[2])
f.add_label(0.6,0.8, text, size = 'large', color = 'white', stretch = 'ultra-condensed', style = 'oblique', weight = 'semibold', relative = True)
f.canvas.draw()

#%%
fig = plt.figure(figsize=(5,5))
f = aplpy.FITSFigure('irz_cube_2d.fits', figure = fig)
f.show_rgb('rgb_image.png')
f.recenter(obj.RA[36], obj.DEC[36], radius = .0045)
f.show_markers(obj.RA[36], obj.DEC[36],marker='c',s=1100,edgecolor='white',facecolor='white')
text = '(RA, DEC) = ({:.2f},{:.2f}) \n z_p = {:.2f} \n chi_p = {:.2f}'.format(obj.RA[36], obj.DEC[36], obj.z_p[36], obj.chi_p[36])
f.add_label(0.6,0.8, text, size = 'large', color = 'white', stretch = 'ultra-condensed', style = 'oblique', weight = 'semibold', relative = True)
fig.canvas.draw()
#f.close()
#%%
f.axis_labels.set_xtext('Right Ascension')
f.axis_labels.set_ytext('Declination')
f.axis_labels.set_ypad(-8)
f.tick_labels.set_yposition('left')
f.axis_labels.set_yposition('left')
f.set_theme('publication')
f.close()
