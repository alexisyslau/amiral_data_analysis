# deconv_image
# perform deconvolution for input images 
"""
    Running deconvolution for output criterion from processed data. 
    Your input data and .csv file needs to be in the same directory(?)

    Usage: python run.py <filename>
        <filename> : .ini file for loading in required parameters for running deconvolution

"""
# Packages required
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
import os
from amiral import instructment, utils, parameter, config, array
from deconvbench import Deconvbench
from mpl_toolkits.axes_grid1 import make_axes_locatable
from amiral.extension import preproc_muse
import argparse
from argparse import ArgumentParser
import configparser
import time
import yaml
import json
import pandas as pd

def amiral (config_amiral, img_obj, telescope, mode, count):

    data = []
    # Setting up the PSF param and hyperparameter dictionary
    symmetric = config_amiral.getboolean('fitting', 'symmetric')
    paramGuess, hyperparamGuess = config.set_paramdict(config_amiral, symmetric)
    print("\nPSF Parameter:\n",paramGuess)
    print("\nHyper Parameter\n",hyperparamGuess)

    if mode == 'muse_nfm':
        # Get keywords for cube splitting because there is a sodium notch filter for muse data 
        wvl_min = img_obj[1].header['CRVAL3']*1e-10/1e-9
        wvl_slice = img_obj[1].header['CD3_3']*1e-10/1e-9
        n_slice = img_obj[1].data.shape[0] 

    else: 
        wvl_min = config_amiral.getfloat(telescope, 'wvl_min') * config_amiral.getfloat(telescope, 'wvl_unit')
        wvl_slice = config_amiral.getfloat(telescope, 'wvl_slice')
        n_slice = img_obj[0].data.shape[0] 
    
    for i in range (count):
        # TODO : Add  a function in here such that it would use a abritrary condition
        img = img_obj[1].data[i,:,:]
        print("\nShape: ", img.shape)
        wvl = (wvl_min+i*wvl_slice)*1e-9
        
        instrucment_file_path = config_amiral.get(telescope, 'ymal_file')
        aosys_profile = config.get_instructment_profile(instrucment_file_path)
        print(aosys_profile)
        
        dimension = img.shape[0]
        
        
        aosys = instructment.aoSystem(diameter = aosys_profile['d'], 
            occ_ratio = aosys_profile['occ'],no_acutuator = aosys_profile['nact'], wavelength = wvl, 
            resolution_rad = aosys_profile['res'], dimension = dimension)

        print(aosys.sampling)

        pupil = aosys.get_pupil_plane()
        plt.imshow(pupil)

        otf_tel = aosys.pupil_to_otf_tel(pupil)

        fX, fY, freqnull = aosys.psd_frequency_array(img.shape[0],aosys.samp_factor[0])
        #plt.imshow(fX**2 + fY**2)

        # Converting dict to array and defining numerical condition
        psf_key, psf_guess = utils.dict2array(paramGuess)
        hyper_key, hyper_guess = utils.dict2array(hyperparamGuess)

        amiralparam = parameter.amiralParam(img, guess = psf_guess, aosys = aosys)

        # BUG - in here, the psf_guess is not the full guess ...
        hyper_min, hyper_max = amiralparam.hyperparam_bound(psf_guess, p_upperbound = 100.)

        # move this part to be internal functions 
        if np.sum(hyper_guess) == 0 : 
            print("\nHyperparameter guess is not set\n")
            hyper_guess = amiralparam.hyperparam_initial(psf_guess)

        psf_guess = np.concatenate((psf_guess, hyper_guess))

        param_min = np.asarray([0.01,0,0,1e-8,1.01])
        param_max =  np.asarray([1.,1e8,1e8,1e3,10])

        upperbound = np.concatenate((param_max, hyper_max))
        lowerbound = np.concatenate((param_min, hyper_min))

        param_numerical_condition = np.array([1., 1e-4, 1., 1., 1.])
        hyperparam_numerical_condition = np.array([hyper_guess[0], hyper_guess[1], 1.])

        numerical_condition = np.concatenate((param_numerical_condition, hyperparam_numerical_condition))

        param_mask = np.asarray(json.loads(config_amiral.get('psf parameter','mask')))
        hyper_param_mask = np.asarray(json.loads(config_amiral.get('psf hyperparameter','mask')))
        mask = np.concatenate((param_mask,hyper_param_mask))

        amiral = parameter.amiral(img=img, guess=psf_guess, aosys = aosys, upperbound = upperbound, lowerbound= lowerbound, numerical_condition = numerical_condition, fourier_variable = amiralparam.fourier_variable, mask = mask)
        est_criterion, value_criterion, value_grad = amiral.minimisation(psf_guess)
        
        # Output the estimated criterion to the panda data frame
        data.append(np.append(est_criterion,wvl))
        print("\nCurrent wavelength: ", wvl)

    keys = psf_key+hyper_key+ ['wvl']
    df = pd.DataFrame(data, columns=keys)
    df_name = config_amiral.get('path', 'output_crit_file')+'.csv'
    df.to_csv(df_name)
    print('\nEstimated criterion has be save to %s'%(df_name))

    return df

def main ():

    # Setting flags and cmd arugments for the scipt
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    # Store commend line arguments to args 
    args = parser.parse_args()

    # Load .ini file as an object
    config_amiral = config.load_config(args)

    # Getting keywords for processing data
    telescope = config_amiral.get('telescope', 'name')
    mode = config_amiral.get('telescope', 'mode')
    output_path = config_amiral.get('path', 'output_path')

    # Read the data
    img_obj = array.read_data(config_amiral, telescope)

    if telescope == 'muse': 
        img = img_obj[1].data 
    else: 
        img = img_obj[0].data 

    # Cases: Hyperspectral data or only 2D data
    if img.ndim == 3: # Hyperspectral data
        print("\n3D array. Run amiral in loop")
        if mode == 'muse_nfm':
            n_slice = img.shape[0] 
            #count = n_slice
            count = 2
            result_frame = amiral(config_amiral, img_obj, telescope, mode, count)
        else:
            raise Exception("Mode %s is not supported" %mode)
            
    elif img.ndim == 2: # Data in a particular wavelength
        count = 1
        raise Exception("Mode %s is not supported yet (Working on it...)" %mode)
        result_frame = amiral(config_amiral, telescope, mode, count)

if __name__ ==  "__main__":
    print("\nRun the Moon\n") 
    main()
    print("\nEnd of Programme\n")