# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:00:26 2018

Fit a muse_nfm datacube with the PSFAO model over all the wavelengths

This script was used to generate figures for the slides of AO4ASTRO,
the seminar at OCA and maybe for the paper (Fétick+2019,A&A)

Updated on 3rd Nov 2019 to match new implementation of STORM

@author: rfetick

This works now --> need to migrate to the plotting script

19 Nov 2020
Fixed the error with float - int; a new error came up --> 

    MUSE_FIT_PSD.py:290: RuntimeWarning: invalid value encountered in true_divide
    ERR = norm((FIT-PSF)*good)/np.sum(PSF*good,axis=(1,2))

    which suggests that the code is trying to divide by zero or nan

    potential solution: 
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore') --> ignore divide by zero and nan error but this is not ideal

    --> this line has been commented out at the moment

    exec(open('muse_fitting.py').read())

TODO 
1/ Organise the plotting functions 
2/ Accept cmd argument


"""
# import library 
from typing import KeysView
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from amiral import array
import os

import matplotlib as mpl
mpl.rcParams['font.size'] = 20

from maoppy.psfmodel import psffit, Psfao, Moffat
from maoppy.instrument import muse_nfm
from maoppy.utils import circavg, circavgplt
from scipy.optimize import least_squares

import argparse
from argparse import ArgumentParser

import datetime

def cost(r0 : np.ndarray,r: np.ndarray, wvl: np.ndarray, index: np.ndarray, pow: float = 6./5.) -> 'function':
    """
        Fitting function for r0 against wavelength: r0 ~ wvl^(6/5)

    Args:
        r0 (np.ndarray): Fried parameters in metre. 
        r (np.ndarray): Initial guess for the cost function (a function which would calculate the vector residual)
        wvl (np.ndarray): Corresponding wavelengths for r0
        index (np.ndarray): Filtered index of the r0 and wvl array
        pow (float, optional): power of the wavelength. Defaults to 6/5..

    Returns:
        residual between 2 vectors (function)
    """
    return r0[index] - r*(wvl[index]/500.*1e9)**(6./5.)

def plot_r0_wvl (r0: np.ndarray,x0: float, wvl: np.ndarray, index: np.ndarray, pow: float = 6./5., save = False):
    """
    TODO - Add a flexibilty for defining the path

        Plot r0 (cm) vs wavelength (nm) with a theoretical prediction, 
        i.e. scatter plot for your input and a best fit based on your input

    Args:
        r0 (np.ndarray): r0 (np.ndarray): fried parameters in metre. 
        x0 (float): best fit parameter from the cost function 
        wvl (np.ndarray): corresponding wavelengths for r0
        index (np.ndarray): Filtered index of the r0 and wvl array
        pow (float, optional): power of the wavelength. Defaults to 6/5..
        save (boolean,optional): option to save the figrue. Defaults to False.
            If True, YY_MM_DD_HHMM would be added in front of the 
    """
    fig, ax = plt.subplot()
    ax.plot(wvl[index]*1e9,(x0*(wvl[index]/500.*1e9)**(pow))*100.,color="C1",linewidth=4,label="Theory",zorder=0)
    ax.scatter(wvl[index]*1e9,R0[index]*100.,label="Fit",zorder=1,s=25)
    ax.set_xlim(5, 25)
    ax.set_ylabel(r'$\mathrm{r_0[cm]}$', fontsize = 14)
    ax.set_xlabel(r'$\mathrm{wavelength [nm]}$', fontsize = 14)
    ax.legend()

    if save == True: 
        date  = datetime.datetime.now()
        time_stamp = date.strftime("%Y_%m_%d_%H%M")
        fig.savefig(time_stamp+'r0_vs_wvl.pdf', dpi = 300)
        print("%s has been saved to %s" %(time_stamp+'r0_vs_wvl.pdf', os.getcwd()))
    pass 

def plot_residual_image (data, fit):

    pass





#%% PARAMETERS TO MODIFY
folder = "/Users/alau/Data/MUSE_DATA/HD_146233/"
folderOUT = "/Users/alau/Data/MUSE_DATA/HD_146233/"

filename = "HD_146233_cube_2_binned_10.fits"

# [rjlf] center the PSF a little bit, to help fitting convergence ;)
xmin = 105 # [rjlf] my bad: it might be better to define center instead of min...
ymin = 111 # (same comment)
Npix = 200 # [rjlf] I think you experienced sizes issues below. It seems that if you specify a shape bigger than the actual array shape, the numpy indexation below does not cut properly the array

#%% READ PSF
fitsPSF = fits.open(folder+filename)
fitsPSF.info()

hdr = fitsPSF[0].header
PSF = fitsPSF[1].data
# PSF = array.resize_array(PSF, size = 100, cent=(ymin,xmin))

print(muse_nfm)

# Masking all nan values
PSF = np.ma.masked_invalid(PSF)

wvl_min = fitsPSF[1].header['CRVAL3']*1e-10
wvl_slice = fitsPSF[1].header['CD3_3']*1e-10
wvl_list = []

print('\nShape of the PSF', PSF.shape) # shape of the PSF is odd (should have been rejected anyways)

#%% INITIALIZE
Nslice = PSF.shape[0]
Npix = PSF.shape[1]

Fao = muse_nfm.Nact/(2.0*muse_nfm.D)
FIT = np.zeros_like(PSF)
good = []
R0 = np.zeros(Nslice)
PARAM = np.zeros((Nslice,6))
AMP = np.zeros(Nslice)
BCK = np.zeros(Nslice)
DXDY = np.zeros((Nslice,2))


# [rjlf] !!! IMPORTANT: The parameter definition evolved a little bit since the puvblished paper and STORM
# [rjlf] Take care of 'alpha' and 'ratio' that are the new parameters!
r0 = 0.13 # Fried parameter [m]
b = 1e-7 # Phase PSD background [rad² m²]
amp = 1.4 # Phase PSD Moffat amplitude [rad²]
alpha = 0.1 # Phase PSD Moffat alpha [1/m]
ratio = 1.0 # ratio sqrt(a_x/a_y)
theta = 0.0 # angle (useful if ratio != 1) [rad]
beta = 1.6 # Phase PSD Moffat beta power law

x0 = [r0,b,amp,alpha,ratio,theta,beta]
# [rjlf] !!! (end of the important note)


# [rjlf] You can also fix some secondary parameters:
# it increases the fitting speed for debugging!
# the order is the same as the one of 'x0' above
fixed = [False,True,False,False,True,True,False]
guess = [0.145,2e-7,1.2,0.08,ratio,theta,1.5]

#%% FIT PSF
num = 50

# samp = muse_nfm.samp(wvl_min)

# for i in range(Nslice):
for i in range(num-1, num):
    wvl = wvl_min+i*wvl_slice
    wvl_list.append(wvl)
    samp = muse_nfm.samp(wvl)
    print("Iteration %2u / %2u"%(i+1,Nslice))
    print("Current wavelength: %.2e" %(wvl))
    print("Sampling: %2f" %(samp))
    
    if np.sum(PSF[i,...]) > 1e5:
        good += [i]
        weights = 1.0/(PSF[i]*(PSF[i]>0)+muse_nfm.ron**2) #np.ones_like(PSF[i,...])
        bad = np.where(PSF[i].mask == True) # masking the nan values in here
        weights[bad[0],bad[1]] = 0.

        out = psffit(PSF[i],Psfao,guess,weights=weights,system=muse_nfm,fixed_k = int(5), samp=samp,fixed=fixed)
        
        x0 = out.x # [rjlf] Use fitted values for the next slice -> improves convergence speed!
        R0[i] = x0[0]
        PARAM[i,:] = x0[1:]
        AMP[i] = out.flux_bck[0]
        BCK[i] = out.flux_bck[1]
        DXDY[i,:] = out.dxdy
        FIT[i,...] = AMP[i]*out.psf + BCK[i]
        print("  r0=%.1f cm | amp=%.2f rad2"%(x0[0]*100,x0[2]))
    else:
        print("  issue with this image ") # [rjlf] This string was in French, sorry!

#%% FIT MOFFAT
MOFF = np.zeros((Nslice,4))
MOFFAMP = np.zeros(Nslice)
MOFFBCK = np.zeros(Nslice)
FITMOFF = np.zeros_like(PSF)

# for i in range(Nslice):
#     if np.sum(PSF[i,...]) > 1e5:
#         weights = 1.0/(PSF[i,...]*(PSF[i,...]>0)+muse_nfm.ron**2) #np.ones_like(PSF[i,...])
#         bad = np.where(PSF[0].mask == True) # all columns and rows in [i]
#         weights[bad[0],bad[1]] = 0.
#         out = psffit(PSF[i,...],Moffat,[1.2,1.2,0,1.5],weights=weights,positive_bck=True)
#         MOFF[i,:] = out.x
#         MOFFAMP[i] = out.flux_bck[0]
#         MOFFBCK[i] = out.flux_bck[1]
#         FITMOFF[i,...] = MOFFAMP[i]*out.psf + MOFFBCK[i]

#%% PLOTS
wvl = np.arange(Nslice)*wvl_slice + wvl_min
laser_min = 575 -5 # [rjlf] approximative wavelength of the dichroic on MUSE due to LGS
laser_max = 595 +7

index = np.where(R0 > 0)[0] # [rjlf] do not account for r0 that where not properly fitted

# [rjlf] Fit all the r0 found with the theoretical curve: r0 ~ wvl^(6/5)
# def cost(r):
#     # print('Hi',r)
#     return R0[index] - r*(wvl[index]/500.*1e9)**(6./5.)

roptim = least_squares(cost,0.13).x[0]

# plt.figure(1)
# plt.clf()
# ax = plt.subplot(111)
# plt.plot(wvl[index]*1e9,(roptim*(wvl[index]/500.*1e9)**(6./5.))*100.,color="C1",linewidth=4,label="Theory",zorder=0)
# plt.scatter(wvl[index]*1e9,R0[index]*100.,label="Fit",zorder=1,s=25)
#plt.xlabel("Lambda [nm]",fontsize=fs)
#plt.ylabel("r0 [cm]",fontsize=fs)
# plt.ylim(5,25)
# plt.xlim(400,950)
# plt.grid()
#plt.fill([laser_min,laser_max,laser_max,laser_min],[plt.ylim()[0],plt.ylim()[0],plt.ylim()[1],plt.ylim()[1]],color="gray",alpha=.5)
#plt.title("Fried parameter")
# plt.legend()
# plt.xticks()
# plt.yticks()
#plt.fill([laser_min,laser_max,laser_max,laser_min],[plt.ylim()[0],plt.ylim()[0],plt.ylim()[1],plt.ylim()[1]],color="gray",alpha=.5)

# print("r0[500nm] = %.2f cm"%(roptim*100))

# plt.legend(fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('$\lambda$ [nm]',fontsize=20)
# plt.ylabel('$r_0$ [cm]',fontsize=20)


# plt.figure(2)
# plt.clf()

# index = np.max(np.arange(Nslice)*((wvl*1e9)<laser_min))+1
# plt.plot(wvl[1:index]*1e9,AMP[1:index]*1e-8,color='C0',linewidth=2,label='Flux [1e8]')
# plt.plot(wvl[1:index]*1e9,BCK[1:index],color='C1',linewidth=2,label='bck')

# index = np.max(np.arange(Nslice)*((wvl*1e9)<laser_max))+1
# plt.plot(wvl[index:]*1e9,AMP[index:]*1e-8,color='C0',linewidth=2)
# plt.plot(wvl[index:]*1e9,BCK[index:],color='C1',linewidth=2)

# plt.grid()
# plt.xticks(plt.xticks()[0])
# plt.xlabel("Lambda [nm]")
# plt.ylabel("Flux & Bck [photon]")
# plt.legend()
# plt.xlim(wvl[1]*1e9,wvl[-1]*1e9)



plt.figure(4)
plt.clf()
cmap = ['Blues','Greens','Reds']
nb = 3
it = 1

# [rjlf] These 3 lines can be removed, they don't seem to be used
x = np.arange(len(circavg(PSF[0,...])))*1.0
x[0] = 1e-1
xPSD = None

for i in range(1,Nslice-nb,int(Nslice/nb)):
    
    PSFplot = np.arcsinh(PSF[i,...])
    FITplot = np.arcsinh(FIT[i,...])
    diff = np.arcsinh(PSF[i,...]-FIT[i,...])
    
    cmin = min([PSFplot.min(),FITplot.min()])
    cmax = max([PSFplot.max(),FITplot.max()])
    #cmin = -cmax
    
    plt.subplot(3,nb,it)
    #plt.title("wvl = %unm"%(wvl[i]*1e9))
    plt.pcolormesh(PSFplot,cmap=cmap[it-1])
    plt.axis('image')
    plt.axis('off')
    plt.clim(cmin,cmax)
    
    plt.subplot(3,nb,nb+it)
    plt.pcolormesh(FITplot,cmap=cmap[it-1])
    plt.axis('image')
    plt.axis('off')
    plt.clim(cmin,cmax)
    
    plt.subplot(3,nb,2*nb+it)
    plt.pcolormesh(diff,cmap=cmap[it-1])
    plt.axis('image')
    plt.axis('off')
    plt.clim(cmin,cmax)
    
    it+=1

plt.tight_layout()
#plt.savefig(folderOUT+'psf_muse.png', transparent=True, overwrite=True)

# [rjlf] This plot was useful since my PSF was centered
# for this specific data, the center moves with the wavelength!!!
plt.figure(5)


plt.clf()
cmap = 'hot'

center = (ymin,xmin)

nb = 3
it = 1

for i in range(1,Nslice-nb,int(Nslice/nb)):
    
    plt.subplot(1,3,it)
    x,y = circavgplt(PSF[i,...])
    plt.semilogy(x,y,label="PSF")
    xm,ym = circavgplt(FIT[i,...],center=center)
    plt.semilogy(xm,ym,label="MODEL")
    xp,yp = circavgplt(np.abs(PSF[i,...]-FIT[i,...]),center=center)
    plt.semilogy(xp,(yp),label="|PSF-MODEL|")
    plt.xlabel('Pixel')
    plt.xlim(-50,50)
    plt.ylim(1e2,6e8)
    plt.title("wvl = %unm"%(wvl[i]*1e9))
    plt.legend()
    it+=1

#%% ERRORS
def norm(x,p=1.0,axis=(1,2)):
    if p<=0: raise ValueError('`p` should be strictly positive')
    return (np.sum(np.abs(x)**p,axis=axis))**(1.0/p)

good = PSF > (-50)
ERR = norm((FIT-PSF)*good)/np.sum(PSF*good,axis=(1,2))
ERRMOFF = norm((FITMOFF-PSF)*good)/np.sum(PSF*good,axis=(1,2))

noise = np.random.randn(*PSF.shape) * muse_nfm.ron
ERRN = norm(noise)/np.sum(PSF*good,axis=(1,2))

# aberrant slices
ERR[0] = np.nan
ERR[25] = np.nan
ERRMOFF[0] = np.nan
ERRMOFF[25] = np.nan
ERRN[0] = np.nan
ERRN[25] = np.nan

v = np.where(1-np.isnan(ERR))
p = np.polyval(np.polyfit(wvl[v],ERR[v],1),wvl)
pm = np.polyval(np.polyfit(wvl[v],ERRMOFF[v],1),wvl)

plt.figure(6)
plt.clf()

plt.plot(wvl*1e9,ERR*100,label='PSFAO',lw=3)
plt.plot(wvl*1e9,ERRMOFF*100,label='Moffat',lw=3)
#plt.plot(wvl*1e9,ERRN*100,label='Noise',lw=1)

plt.plot(wvl*1e9,p*100,lw=3,ls=':')
plt.plot(wvl*1e9,pm*100,lw=3,ls=':')

plt.legend(loc='lower right')
plt.xlabel('$\lambda$ [nm]')
plt.ylabel('Error [%]')
plt.ylim(0,30)
plt.xlim(wvl.min()*1e9,wvl.max()*1e9)
plt.grid()




outputdata = {
    'wvl': wvl, 
    'r0': R0,
    'bck': PARAM[:,0], 
    'amplitude': PARAM[:,1], 
    'alpha': PARAM[:,2],
    'ratio': PARAM[:,3], 
    'theta': PARAM[:,4], 
    'beta':  PARAM[:,5], 
    'flux(amp)': AMP, 
    'flux(bck)': BCK, 
    'dx':  DXDY[:,0], 
    'dy':  DXDY[:,1]
}
keys = ['wvl', 'r0', 'background', 'amplitude', 'alpha', 'ratio','theta', 'beta', 'flux(amp)', 'flux(bck)', 'dx', 'dy']

# PARAM = [b,amp,alpha,ratio,theta,beta]
fitted_psf_param = pd.DataFrame(data=outputdata,columns=keys)
fitted_psf_param.to_csv('HD_146233_cube_2_binned_10_FULL_samp.csv')

# center=(50,43)
# plt.clf()

num = 0

plt.clf()
center = (111,104)
x,y = circavgplt(PSF[num-1],center=center)
plt.semilogy(x,y,label="PSF")
xm,ym = circavgplt(FIT[num-1],center=center)
plt.semilogy(xm,ym,label="MODEL")
xp,yp = circavgplt(np.abs(PSF[num-1]-FIT[num-1]),center=center)
plt.semilogy(xp,(yp),label="|PSF-MODEL|")
plt.xlabel('Pixel')
# plt.xlim(-50,50)
# plt.ylim(1e2,6e8)
plt.title("wvl = %unm"%(wvl[num-1]*1e9))
plt.legend()

plt.show()

# def main ():

#     # Setting flags and cmd arugments for the scipt
#     parser = argparse.ArgumentParser()

#     parser.add_argument('filename', 
#     help = 'the config file which contains path of the data and the name of the file')

#     # Store commend line arguments to args 
#     args = parser.parse_args()

    
#     pass

# if __name__ ==  "__main__":
#     print("\nPerform PSF fitting with maoppy\n") 
#     main()
#     print("\nEnd of Programme\n")