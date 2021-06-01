"""
    PSF generator using maoppy

    Usage: 
        python psf_generator.py <filename>
        <filename> : .ini file which contains information where to dump the files


    TODO 
        1/ Add the custom mode back into the function
"""

# Packages required
import numpy as np
from astropy.io import fits
import csv
import pandas as pd
from argparse import ArgumentParser
import time
from pathlib import Path
from amiral import config

from maoppy.psfmodel import Psfao
from maoppy.instrument import muse_nfm

def set_param_dict (): 
    pass 

def gen_psf (aosys, psfParam) -> np.ndarray:
    """
        PSF generator using the model from maoppy 
        
        arguements are passed to this function such that it allows the user to add noise into the system
    Args:
        args ([type]): [description]
    """

    # Initialize PSF model
    samp = muse_nfm.samp(wvl) # sampling (2.0 for Shannon-Nyquist)
    Pmodel = Psfao((Npix,Npix),system=muse_nfm,samp=samp)

    # Choose parameters and compute PSF
    r0 = 0.15 # Fried parameter [m]
    bck = 1e-5 # Phase PSD background [rad² m²]
    amp = 5.0 # Phase PSD Moffat amplitude [rad²]
    alpha = 0.1 # Phase PSD Moffat alpha [1/m]
    ratio = 1.2
    theta = np.pi/4
    beta = 1.6 # Phase PSD Moffat beta power law

    # from a dictionary back to array

    param = [r0,bck,amp,alpha,ratio,theta,beta]

    psf = Pmodel(param,dx=0,dy=0) 
    pass


def main ():    
    # Setting flags and cmd arugments for the scipt
    parser = ArgumentParser()
    
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    # # Noise 
    # parser.add_argument('--noise', '--n', dest = 'noise',
    # help = 'Add noise', action = 'store_true')

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

    aosys = {
        'nPix': config_file.getint('psf parameter', 'nPix'), 
        'system': config_file.get('telescope', 'mode')
    }


    if args.number == 1:
        paramGuess, hyperparamGuess = config.set_paramdict(config_file, True)
        psf = gen_psf(aosys, psfParam)

    else: 
        pass


    pass


if __name__ ==  "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    print("Run Time (min): ",runtime/60)
    print("\nEnd of Programme\n")
