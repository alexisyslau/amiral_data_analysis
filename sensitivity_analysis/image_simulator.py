# Packages required
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import image, rcParams
import csv
import pandas as pd
import os
import argparse
from argparse import ArgumentParser
import configparser
import time
from pathlib import Path
from amiral import instructment, utils, parameter, config, array
from amiral.extension import data_generator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# TODO - Add SNR and SR function in

def get_snr (array, noise):
    
    mean = np.mean(array)
    sig2 = np.std(noise)
    
    snr = mean / sig2
    
    return snr

def get_otf_total(aosys_cls, psf_param): 
    """
    Quick squence for getting the OTF

    Args:
        aosys_cls ([type]): [description]
        psf_param ([type]): [description]

    Returns:
        [type]: [description]
    """

    psd_ao = aosys_cls.psd_residual_ao (psf_param)
    psd_halo = aosys_cls.psd_residual_halo(psf_param[0])

    psd = psd_halo + psd_ao 

    pupil = aosys_cls.get_pupil_plane()
    otf_tel = aosys_cls.pupil_to_otf_tel(pupil)

    otf_atmo = aosys_cls.otf_atmo(psd)

    otf_total = otf_atmo * otf_tel

    return otf_total

def gauss_noise (dimension, RON): 
    return np.random.randn(dimension,dimension)*RON

def forced_zero (array):

    """
    Checking the input image array and make sure there is no zero in the input array 
    Since the input array is treated as an object, val < 0 / = nan would result in error 
    while estimating the photon noise of your simulated observations


    Args:
        array ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    ind = np.where(array < 0)

    if ((len(ind))) != 0: 
        array[ind] = 0.
        return array

    else: 
        return array

def read_image (config_file, flux): 

    input_dir = config_file.get('path', 'data_path')
    data_fname = config_file.get('path', 'data_file')

    img_obj = fits.open(input_dir + data_fname + ".fits")
    obj = img_obj[0].data 
    obj = obj/np.sum(obj)*flux

    return obj 

def write2header (param,flux,snr ,keys): 

    hdr = fits.Header()

    param = np.append(param, flux)
    param = np.append(param, snr)

    for i in range (len(keys)): 
        hdr[keys[i]] = param[i]

    return hdr

def add_noise (image, dimension, aosys, args, RON):

    if args.noise == True:

        print("Add noise to the object")

        _noise = gauss_noise(dimension*aosys.samp_factor[0], RON = RON)
        rng = np.random.default_rng()
        _photon_noise = rng.poisson(image)
        
        noise = _photon_noise+_noise
        image = image + noise

        return image
    else: 
        print("No noise")
        return image

def get_snr (array, noise):
    
    mean = np.mean(array)
    sig2 = np.std(noise)
    
    snr = mean / sig2
    
    return snr


def plot_images_noise (obj,conv_obj,noise,img):

    default_size = 200

    zoom_obj = array.zoom_array(obj, default_size)
    zoom_conv_obj = array.zoom_array(conv_obj, default_size)
    zoom_noise = array.zoom_array(noise, default_size)
    zoom_img = array.zoom_array(img, default_size)


    fig, ax = plt.subplots(2,2)

    rcParams['figure.figsize'] = 33 ,24

    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax[0,0].imshow(zoom_conv_obj)
    fig.colorbar(im,cax ,ax=ax[0,0])
    ax[0,0].set_title('Convoloved Object\nFlux [e-]: %f' %(np.sum(conv_obj)), fontsize = '12')

    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im1 = ax[0,1].imshow(zoom_noise)
    fig.colorbar(im1,cax ,ax=ax[0,0])
    ax[0,1].set_title('Sum of the noise\nFlux [e-]: %f' %(np.sum(noise)), fontsize = '12')

    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im2 = ax[1,0].imshow(img)
    fig.colorbar(im2,cax ,ax=ax[1,0])
    ax[1,0].set_title('Observed image\nFlux[e-]: %f' %(np.sum(img)), fontsize = '12')

    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im3 = ax[1,1].imshow(img-conv_obj-noise)
    fig.colorbar(im3,cax ,ax=ax[1,0])
    ax[1,1].set_title('Residual\nFlux[e-]: %f' %(np.sum(img-conv_obj-noise)), fontsize = '12')

    plt.show()

    pass

def main ():

    # define vairable
    data_file = []

    # Setting flags and cmd arugments for the scipt
    parser = argparse.ArgumentParser()
    
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    parser.add_argument('number', 
    help = 'numebr of images')

    # Parameter search
    parser.add_argument('--noise', '--n', dest = 'noise',
    help = 'Add noise', action = 'store_true')

    # Store commend line arguments to args 
    args = parser.parse_args()

    # config object to be read
    config_file = config.load_config(args)

    # get the data name
    data_fname = config_file.get('path', 'data_file')

    # Set output path
    output_path = config_file.get('path', 'output_path')

    # If the output path doesnt exist, make one
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Setting up dummy psf parameters for looping
    paramGuess, hyperparamGuess = config.set_paramdict(config_file, True)

    # Seperate the dict into np.array and keys 
    psf_key, psf_guess = utils.dict2array(paramGuess)
    hyper_key, hyper_guess = utils.dict2array(hyperparamGuess)
    psf_guess = np.concatenate((psf_guess, hyper_guess))

    # Get the system profile
    aosys_profile = config.get_instructment_profile(config_file)
    dimension = config_file.getint('custom', 'dimension')

    # Setting the telescope system for generating PSF
    aosys = instructment.aoSystem(diameter = aosys_profile['d'], 
        occ_ratio = aosys_profile['occ'],no_acutuator = aosys_profile['nact'], 
        sampling = aosys_profile['sampling'], wavelength = aosys_profile['wvl']*1e-9, 
        resolution_rad = aosys_profile['res'], dimension = dimension)

    # Setting up variables to be looped over
    guess_flux = 5.*np.logspace(4,14,int(args.number))

    noise_run = 10

    # Setting the column index
    keys = psf_key + hyper_key + ['flux'] + ['snr']

    # Main loop for generating simulated observations
    # BUG - Should have padded the image before any scaling 
    for i in range (int(args.number)): 
        guess_1 = np.array([psf_guess[0], psf_guess[1],psf_guess[2]])
        guess_2 = psf_guess[3:]
        _guess = np.concatenate((guess_1,guess_2))

        # calculate the otf
        _otf = np.fft.ifftshift(get_otf_total(aosys, _guess))
        # plt.imshow(np.real(np.log10(_otf)))
        # plt.show()
 
        # Dealing with the input object
        obj = read_image(config_file, guess_flux[i])

        # Check if there is any zero for the input object
        obj = forced_zero(obj)

        padded_obj = array.scale_array(obj, aosys.samp_factor[0])        
        ft_obj = np.fft.fft2(np.fft.ifftshift(padded_obj))

        for j in range (noise_run): 
            
            # Looping over different noise 
            # get the noise
            count = i*noise_run+j
            print("Count", count)

            # Read out noise
            _noise = gauss_noise(dimension*aosys.samp_factor[0], RON = 10)

            # All this should be centred
            ft_img = ft_obj*_otf # ft_obj * _otf
            _img = np.real(np.fft.ifft2(ft_img))

            # Photon noise
            rng = np.random.default_rng()
            _photon_noise = rng.poisson(_img)
            noise = _photon_noise+_noise

            img = add_noise (_img, dimension, aosys, args, RON = 10)

            # plot_images_noise(obj, conv_obj=_img, noise=noise, img = img)

            # # Get the convoloved image
            # img = np.real((_img+noise))

            # plt.imshow(img-noise)
            # plt.show()

            # plt.imshow(padded_obj)
            # plt.show()
  
            # Get an SNR for each output, which would be saved to an output
            snr = get_snr(img, noise)

            print("Sum: ", np.sum(img))
            print("Flux(object): ", np.sum(_img))
            print("Noise of the object: ", np.sum((noise)))
            print("Photon Noise of the object: ", np.sum((_photon_noise)))
            print("Retrieved Flux: ",(np.sum(img) - np.sum(noise) - guess_flux[i]))

            hdr = write2header (_guess,guess_flux[i],snr,keys) 
            
            # output an image to a specific dir with name
            fits.writeto(output_path+ data_fname +'_'+ str(count) + '.fits', img, hdr)
            fits.writeto(output_path+ data_fname+'_noise_'+ str(count) + '.fits', noise, hdr)
            print("%s has been saved to %s " %(data_fname+'_'+ str(count) + '.fits', output_path))

            case_info = np.append(_guess, guess_flux[i])
            case_info = np.append(case_info, snr)
            print(case_info)
            data_file.append(case_info)
    
    print(data_file)
    data_generator.csv_generator(keys=keys, data = data_file, config = config_file)

    pass 

if __name__ ==  "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    print("Run Time (min): ",runtime/60)
    print("\nEnd of Programme\n")