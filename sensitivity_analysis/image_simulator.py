# Packages required
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv
import pandas as pd
import os
import argparse
from argparse import ArgumentParser
import configparser
import time

from amiral import instructment, utils, parameter, config, array
from amiral.extension import data_generator


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

def read_image (config_file): 

    input_dir = config_file.get('path', 'data_path')
    data_fname = config_file.get('path', 'data_file')

    img_obj = fits.open(input_dir + data_fname)
    obj = img_obj[0].data 

    return obj 

def main ():

    # define vairable
    data_file = []

    # Setting flags and cmd arugments for the scipt
    parser = argparse.ArgumentParser()
    
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    parser.add_argument('number', 
    help = 'numebr of images')

    # Store commend line arguments to args 
    args = parser.parse_args()

    config_file = config.load_config(args)

    paramGuess, hyperparamGuess = config.set_paramdict(config_file, True)

    psf_key, psf_guess = utils.dict2array(paramGuess)
    hyper_key, hyper_guess = utils.dict2array(hyperparamGuess)
    psf_guess = np.concatenate((psf_guess, hyper_guess))

    # Set up the AO system for the PSF profile
    aosys_profile = config.get_instructment_profile(config_file)
    dimension = config_file.getint('custom', 'dimension')

    # Setting the telescope system
    aosys = instructment.aoSystem(diameter = aosys_profile['d'], 
        occ_ratio = aosys_profile['occ'],no_acutuator = aosys_profile['nact'], 
        sampling = aosys_profile['sampling'], wavelength = aosys_profile['wvl']*1e-9, 
        resolution_rad = aosys_profile['res'], dimension = dimension)

    # Dealing with the input object
    obj = read_image(config_file)
    padded_obj = array.scale_array(obj, aosys.samp_factor[0])
    ft_obj = utils.fft2D(padded_obj, norm=True)

    # Setting up variables to be looped over
    guess_r0 = np.linspace(0.1, 0.25,int(args.number))
    guess_sig2 = np.linspace(0.1,2.6,int(args.number))
    guess_flux = np.linspace(5e6,5e9,int(args.number))

    # Setting the column index
    keys = psf_key + hyper_key + 'Flux'
    
    for i in range (int(args.number)): 
        guess_1 = np.array([guess_r0[i], psf_guess[1],guess_sig2[i]])
        guess_2 = psf_guess[3:]
        guess_3 = guess_flux[i]
        _guess = np.concatenate((guess_1,guess_2))
        guess = np.concatenate((_guess,guess_3))

        # calculate the otf
        _otf = get_otf_total(guess)

        # get the noise
        _noise = gauss_noise(dimension, RON = 10)
        
        # output an image to a specific dir with name

        print(guess)
        data_file.append(guess)
    
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