"""
    Image simulator for generating simulated observations with PSFAO19 model. 

    If the directory doesnt exist, it will create one because I am lazy zzzzz. 

Returns:
    [type]: [description]


    TODO - 1/ Add maoppy in here!!!!!!!!!!
    TODO - 2/ Add a function which allow you to import PSF for direct convolution (in FFT space)
    TODO - 3/ Is the noise mode working? Need to check!

"""


# Packages required
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from argparse import ArgumentParser
import time
from pathlib import Path
from amiral import instructment, utils, config

# Global variable
RON = 10 

# TODO - Add SNR and SR function in

def get_snr (array, flux ,RON, photon_noise):
    # source of the equation: https://www.eso.org/~ohainaut/ccd/sn.html

    signal = flux 
    dim = np.shape(array)[0]

    noise = np.sqrt(np.sum(photon_noise) + (dim**2)*RON**2)
    snr = signal / noise
    
    return snr

def get_otf_total(aosys_cls, psf_param): 
    # TODO - Use maoppy in here
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

def gauss_noise (dimension: int, RON: int) -> np.ndarray: 
    return np.random.randn(dimension,dimension)*RON

def forced_zero (array: np.ndarray) -> np.ndarray:
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

def read_image (input_dir: str, data_fname: str): 
    # TODO - Consider changing this functions, or use the one directly from the ... 
    """
    Read the image path from the config file. 

    Args:
        config_file (config object): [description]
        flux (float): Flux 

    Returns:
        [type]: [description]
    """

    img_obj = fits.open(input_dir + data_fname + ".fits")
    obj = img_obj[0].data 
    obj = obj / np.sum(obj)

    return obj 

def write2header (param,flux,snr ,keys): 

    hdr = fits.Header()

    param = np.append(param, flux)
    param = np.append(param, snr)

    for i in range (len(keys)): 
        hdr[keys[i]] = param[i]

    return hdr

def add_noise (image, dimension, aosys, RON, flux):

    print("Add noise to the object")
    print("Dimension:", dimension)
    print("Image shape:", np.shape(image))
    print("Flux:", np.sum(image))

    if aosys.samp_factor[1] == False:
        _noise = gauss_noise(dimension, RON = RON)
        rng = np.random.default_rng()
        _photon_noise = rng.poisson(image)

    else:
        _noise = gauss_noise(dimension*aosys.samp_factor[0], RON = RON)
        rng = np.random.default_rng()
        _photon_noise = rng.poisson(image)
        
    
    noise = _photon_noise+_noise

    # plt.imshow(image, cmap = 'gray')
    # plt.show()

    image = image + noise

    # plt.imshow(array.resize_array(image, 100), cmap = 'gray')
    # plt.show()

    snr = get_snr(image, flux, RON, _photon_noise)
    print("SNR:", snr)

    return image,snr


def convolve_2D (config_file, arg, guess_flux,psf_obj = None):
 
    # Set the path - TODO - Put this into a dictionary? Personally, I dont like a lot of inputs for a functions 
    data_fname = config_file.get('path', 'data_file')
    input_dir = config_file.get('path', 'data_path')

    # Set the PSF path
    psf_dir = config_file.get('path', 'psf_path')
    psf_fname = config_file.get('path', 'psf_file')

    # Set output path
    output_path = config_file.get('path', 'output_path')
    # Setting up dummy psf parameters for looping


    # Get your object
    print(input_dir+data_fname)
    obj = read_image(input_dir, data_fname)*guess_flux

    # Needed for later
    dimension = np.shape(obj)[0]

    # Setting the telescope system for generating PSF
    telescope = config_file.get('telescope', 'name')

    # Get the system profile
    aosys_profile = config.get_instructment_profile(config_file)

    if telescope == 'custom': 
        aosys = instructment.aoSystem(diameter = aosys_profile['d'], 
            occ_ratio = aosys_profile['occ'],no_acutuator = aosys_profile['nact'], 
            sampling = aosys_profile['sampling'], wavelength = aosys_profile['wvl']*1e-9, 
            resolution_rad = aosys_profile['res'], dimension = dimension)
    else: 
        aosys = instructment.aoSystem(diameter = aosys_profile['d'], 
            occ_ratio = aosys_profile['occ'],no_acutuator = aosys_profile['nact'], 
            wavelength = aosys_profile['wvl']*1e-9, 
            resolution_rad = aosys_profile['res'], dimension = dimension)



    if arg.mode == 'image' and psf_obj == None:         
        psf_obj = fits.open(psf_dir+psf_fname+'.fits')
        psf_obj.info()
        psf = psf_obj[0].data
    else: 
        psf = psf_obj[0].data

    # A dark hole can be seen around the corner

    # FT(transform) of the quantities
    otf = np.fft.fft2(psf)
    # plt.imshow(np.real(np.log10((otf))))
    # plt.show()

    ft_obj = np.fft.fft2(obj)
    # plt.imshow(np.real(np.log10(ft_obj)))
    # plt.show()

    ft_img = otf*ft_obj 
    # plt.imshow(np.real(np.log10((ft_img))))
    # plt.show()

    # Get the simulated image.
    img = np.real(np.fft.fftshift(np.fft.ifft2(ft_img)))
    # plt.imshow(np.real(np.log10(img)))
    # plt.show()

    if arg.noise == True :
        print("Adding noise into the image") 
        img, snr = add_noise(img, dimension, aosys, RON, guess_flux)
        # plt.imshow(np.log10(img))
        # plt.show()
    else: 
        print("No noise case")
        snr = 0.

    paramGuess, hyperparamGuess = config.set_paramdict(config_file, True)

    paramGuess['r0']= psf_obj[0].header['R0']
    paramGuess['background'] = psf_obj[0].header['BACKGROUND']
    paramGuess['amplitude'] = psf_obj[0].header['AMPLITUDE']
    paramGuess['ax'] = psf_obj[0].header['ALPHA']
    paramGuess['beta'] = psf_obj[0].header['BETA']

    # Seperate the dict into np.array and keys 
    psf_key, psf_guess = utils.dict2array(paramGuess)
    hyper_key, hyper_guess = utils.dict2array(hyperparamGuess)
    psf_guess = np.concatenate((psf_guess, hyper_guess))
    
    # Setting the column index
    keys = psf_key + hyper_key + ['flux'] + ['snr'] 

    hdr = write2header (psf_guess,guess_flux,snr,keys)
        
    return hdr, img 


def convolve_batch (config_file, arg, guess_flux):
    
    # Set the path
    data_fname = config_file.get('path', 'data_file')
    input_dir = config_file.get('path', 'data_path')

    # Set the PSF path
    psf_dir = config_file.get('path', 'psf_path')
    psf_fname = config_file.get('path', 'psf_file')

    # Set output path
    output_path = config_file.get('path', 'output_path')
    input_df_name = config_file.get('path', 'data_csv')

    data_input = pd.read_csv(os.path.join(psf_dir, input_df_name)+ '.csv')
    _data_fname = data_input['Unnamed: 0']

    psf_data_fname = config_file.get('path', 'psf_file') + '_'+ _data_fname.astype(str)+'.fits'
    print(psf_data_fname)
    data_fname = config_file.get('path', 'data_file') + '_'+ _data_fname.astype(str)+'.fits'

    for i in range (len(psf_data_fname)):
        # Read the psf files
        psf_obj = fits.open(psf_dir + psf_data_fname[i])
        print("Processing %s " %(psf_dir + psf_data_fname[i]))

        # Input a PSF file and write to .fits file
        _hdr, _img = convolve_2D(config_file, arg, guess_flux,psf_obj)

        print("Output to %s" %(output_path+ data_fname[i]))
        fits.writeto(output_path+ data_fname[i], _img, _hdr)

    pass

def main ():

    # define vairable
    data_file = []

    # Setting flags and cmd arugments for the scipt
    parser = argparse.ArgumentParser()
    
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    # Add noise or not
    parser.add_argument('--noise', '--n', dest = 'noise',
    help = 'Add noise', action = 'store_true')

    # Input PSF files or not
    # TODO - Can you do store False? 
    parser.add_argument('--psf', '--h', dest = 'psf',
    help = 'Input PSF', action = 'store_true')

    # Single images, a batch of image or a 3D cube? 
    parser.add_argument('--mode', '--m', 
    help = 'mode for the image convolution: 1. image (1 PSF); 2. ??? - 3D PSF Cube (x,y,lambda)')

    # Store commend line arguments to args 
    args = parser.parse_args()

    print(args.noise)

    # config object to be read
    config_file = config.load_config(args)

    # Setting up variables to be looped over
    guess_flux = 5e7 # Fixed for now!
    # guess_flux = 5.*np.logspace(4,14,10) # 5e4

    # Set the path
    data_fname = config_file.get('path', 'data_file')
    input_dir = config_file.get('path', 'data_path')

    # Set the PSF path
    psf_dir = config_file.get('path', 'psf_path')
    psf_fname = config_file.get('path', 'psf_file')

    # Set output path
    output_path = config_file.get('path', 'output_path')

    # If the output path doesnt exist, make one
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if args.mode != None: 

        if args.mode == 'spectral': 
            raise Exception("Spectral mode (not developed yet!)")

        elif args.mode == 'image':
            hdr, img = convolve_2D(config_file, args, guess_flux)
            fits.writeto(output_path+ data_fname+ '.fits', img, hdr)
            print("Perform convolution with a single image")
            print("Calling from image.py")
        
        elif args.mode == 'batch':
            print("Batch mode")
            convolve_batch(config_file, args, guess_flux)
            #print("Intake files from  %s" %(config_amiral.get('path', 'data_path')))
            
        else:
            raise Exception ("--mode input does not match current defintion ('spectral', 'image' or 'batch')")
    else: 
        raise Exception ("--mode input is not None ('spectral', 'image' or 'batch')")

    pass 

if __name__ ==  "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    print("Run Time (min): ",runtime/60)
    print("\nEnd of Programme\n")