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
from amiral import config, array
import matplotlib.pyplot as plt
from maoppy.psfmodel import Psfao
from maoppy.instrument import muse_nfm
import datetime
import os

armstrong = 1e10

def set_psf_param_dict (config_file, inputFile = False):

    if inputFile == False: 
        psf_param = {
            "r0": config_file.getfloat('psf parameter','r0'),             
            "background": config_file.getfloat('psf parameter','background') ,      
            "amplitude": config_file.getfloat('psf parameter','amplitude'),       
            "alpha": config_file.getfloat('psf parameter','alpha'), 
            "ratio": config_file.getfloat('psf parameter','ratio'), 
            "theta": config_file.getfloat('psf parameter','theta'),                          
            "beta": config_file.getfloat('psf parameter','beta'), 
            "dx": config_file.getfloat('psf parameter','dx'), 
            "dy": config_file.getfloat('psf parameter','dy')
        }
    else: 
        csv_path = config_file.get('path', 'input_param')
        dfParam = pd.read_csv(csv_path)
        # print(dfParam.keys())

        # Check if the below part is a duplicate
        psf_param = {
            "r0": dfParam['r0'],             
            "background": dfParam['background'] ,      
            "amplitude":dfParam['amplitude'],       
            "alpha": dfParam['alpha'], 
            "ratio": dfParam['ratio'], 
            "theta": dfParam['theta'],                          
            "beta":dfParam['beta'], 
            "dx": dfParam['dx'], 
            "dy": dfParam['dy'],
            "wvl": dfParam['wvl'], 
            "flux(amp)": dfParam['flux(amp)'],
            "flux(bck)": dfParam['flux(bck)']
        }

    return psf_param 

def gen_psf (aosys, psfParam, noise = False) -> np.ndarray:
    """
        PSF generator using the model from maoppy.
        
        Arguements are passed to this function such that it allows the user to add noise into the system.

    Args:
        aosys (dictionary): [description]
        psfParam (dictionary): [description]
    """
    # Initialize PSF model
    samp = muse_nfm.samp(psfParam['wvl']) 
    psfmodel = Psfao((aosys['nPix'],aosys['nPix']),system=muse_nfm,samp=samp, fixed_k= int(5))

    # from a dictionary back to array
    param = [psfParam['r0'],psfParam['background'],psfParam['amplitude'],psfParam['alpha'],psfParam['ratio'],psfParam['theta'],psfParam['beta']]
    
    if noise == False: 
        psf = psfParam['flux(amp)'] * psfmodel(param,dx=psfParam['dx'],dy=psfParam['dy']) + psfParam['flux(bck)']
    else: 
        psf = psfParam['flux(amp)'] * psfmodel(param,dx=psfParam['dx'],dy=psfParam['dy']) + psfParam['flux(bck)'] + np.random.randn(aosys['nPix'],aosys['nPix'])*muse_nfm.ron

    return psf


def main ():    
    # Setting flags and cmd arugments for the scipt
    parser = ArgumentParser()
    
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    parser.add_argument('--mode', '--m', 
    help = 'mode of amiral: hyperspectral, image or batch')

    # Noise 
    parser.add_argument('--noise', '--n', dest = 'noise',
    help = 'Add noise', action = 'store_true')

    # Store commend line arguments to args 
    args = parser.parse_args()

    # keywords for the noise
    noise = args.noise

    # config object to be read
    config_file = config.load_config(args)

    # Set output path
    output_path = config_file.get('path', 'output_path')
    output_fname = config_file.get('path', 'output_file')

    # If the output path doesnt exist, make one
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Input file
    input_file = config_file.getboolean('psf parameter','file')

    aosys = {
        'nPix': config_file.getint('psf parameter', 'nPix'), 
        'system': config_file.get('telescope', 'mode')
    }

    psfParam = set_psf_param_dict(config_file, input_file)
    
    # Setting the conditions for generating the PSF from maoppy
    if args.mode == 'image':
        # Make sure input_file is false
        input_file = False
        psf = gen_psf(aosys, psfParam, noise)
        
    elif (args.mode != 'image' and input_file == True): 
        psf = []

        # Read the psf parameters from .csv file
        for i in range (len(psfParam['wvl'])):
            _df = {
                "r0": psfParam['r0'][i],             
                "background": psfParam['background'][i],      
                "amplitude":psfParam['amplitude'][i],       
                "alpha": psfParam['alpha'][i], 
                "ratio": psfParam['ratio'][i], 
                "theta": psfParam['theta'][i],                          
                "beta":psfParam['beta'][i], 
                "dx": psfParam['dx'][i], 
                "dy": psfParam['dy'][i],
                "wvl": psfParam['wvl'][i], 
                "flux(amp)": psfParam['flux(amp)'][i], 
                "flux(bck)": psfParam['flux(bck)'][i]
                # Need to change the wvl to the bluest in here --> to avoid the jump
            }
            # print(_df)
            _psf = gen_psf(aosys, _df, noise)
            psf.append(_psf)

        # Convert  
        psf = np.asarray(psf)

    else:
        # In theory, you can use the theoretical relationship to derive the r0 
        # But things get a little more complicated with other parameter 
        # Like alpha and beta, but you should be able to fix them since they are small (?)
        raise Exception ('Without .csv file as input for spectral cube is not supported yet')

    # TODO - put a save fits function in here
    # TODO - Do you need a header

    # Output the file to a fits file in here
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(data=psf)
    new_hdul = fits.HDUList([hdu1, hdu2])
    
    hdr = new_hdul[1].header

    new_hdul.info()

    # Get the today's date
    date  = datetime.datetime.now()

    # Add the date to the header
    hdr['DATE'] = date.strftime("%Y-%m-%d")
    hdr['CRVAL3'] = psfParam['wvl'][0]*armstrong
    hdr['CD3_3'] = armstrong*(psfParam['wvl'][1] - psfParam['wvl'][0])

    # the name can input from the .ini file
    array.save_fits(img_obj=new_hdul, name = os.path.join(output_path, output_fname))

    # fits.info(+'.fits')
    # _img = fits.open('hi.fits')

    print("The output file!")
    pass


if __name__ ==  "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    print("Run Time (min): ",runtime/60)
    print("\nEnd of Programme\n")
