"""
    PSF generator using maoppy model

    Usage: 
        python psf_generator.py <filename> --m <mode>
        
        <filename> : .ini file which contains information where to dump the file
        <mode>: which mode to be called
            image - 2D 
            cube - 
            batch - a batch of 2D PSF which is saved individually to the output directory set in the .ini file. 
    TODO 
        1/ Add the custom mode back into the function
        2/ Add strehl ratio!
"""
# Packages required
from io import StringIO
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

# Global vairables
armstrong = 1e10

# Functions
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
            "dy": config_file.getfloat('psf parameter','dy'), 
            "wvl": config_file.getfloat('telescope','wvl')*1e-9, # converting [nm] to [m]
            "flux(amp)": config_file.getfloat('psf parameter','flux_amp'),
            "flux(bck)": config_file.getfloat('psf parameter','flux_bck')
        }
    else: 
        print("Read from input file")
        csv_path = config_file.get('path', 'input_param')
        psf_param = pd.read_csv(csv_path)

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
    if aosys['system'] == 'muse_nfm':
        sys = muse_nfm
    elif aosys['system'] == 'zimpol': 
        sys = None # TODO - Find out what it is! 
    else: 
        raise Exception('input value is not supported yet. If it is, it should be the same as the setup for the format')

    samp = sys.samp(psfParam['wvl']) 
    psfmodel = Psfao((aosys['nPix'],aosys['nPix']),system=sys,samp=samp, fixed_k= int(5))

    # from a dictionary back to array
    param = [psfParam['r0'],psfParam['background'],psfParam['amplitude'],psfParam['alpha'],psfParam['ratio'],psfParam['theta'],psfParam['beta']]
    print(param)

    if noise == False: 
        psf = psfParam['flux(amp)'] * psfmodel(param,dx=psfParam['dx'],dy=psfParam['dy']) + psfParam['flux(bck)']
    else: 
        psf = psfParam['flux(amp)'] * psfmodel(param,dx=psfParam['dx'],dy=psfParam['dy']) + psfParam['flux(bck)'] + np.random.randn(aosys['nPix'],aosys['nPix'])*muse_nfm.ron

    strehl = psfmodel.strehlOTF(param)

    return psf, strehl

def save_muse_fits (psf: np.ndarray, param: dict, output_path: StringIO, output_fname: StringIO, mode: StringIO) -> None: 
    """
        Create an astropy.fits object with headers, then save the muse .fits files. 

    Args:
        psf (np.ndarray): 
        param (dict): 
        output_path (StringIO): 
        output_fname (StringIO): 
    """
    # Get the today's date
    date  = datetime.datetime.now()

    if mode == "cube": 
        # If it is the 3D PSF cube, Follow the format of the muse file.
        hdu1 = fits.PrimaryHDU()
        hdu2 = fits.ImageHDU(data=psf)
        new_hdul = fits.HDUList([hdu1, hdu2])
        
        hdr = new_hdul[1].header
        # Add the date to the header
        hdr['DATE'] = date.strftime("%Y-%m-%d")
        hdr['CRVAL3'] = param['wvl'][0]*armstrong
        hdr['CD3_3'] = armstrong*(param['wvl'][1] - param['wvl'][0])
    else: 
        hdu1 = fits.PrimaryHDU(data=psf)
        new_hdul = fits.HDUList([hdu1])
        hdr = new_hdul[0].header
        hdr['DATE'] = date.strftime("%Y-%m-%d")
        hdr['CRVAL3'] = param['wvl']*armstrong
        hdr['R0'] = param['r0']
        hdr['BACKGROUND'] = param['background']
        hdr['AMPLITUDE'] = param['amplitude']
        hdr['ALPHA'] = param['alpha']
        hdr['RATIO'] = param['ratio']
        hdr['THETA'] = param['theta']
        hdr['BETA'] = param['beta']     
        hdr['DX'] = param['dx'] 
        hdr['DY'] = param['dy'] 
        hdr['WVL'] = param['wvl'] 
        hdr['FLUX(AMP)'] = param['flux(amp)'] 
        hdr['FLUX(BCK)'] = param['flux(bck)']
        hdr['STREHL'] = param['strehl']

    # the name can input from the .ini file
    array.save_fits(img_obj=new_hdul, name = os.path.join(output_path, output_fname))

    pass 


def main ():    
    # Setting flags and cmd arugments for the scipt
    parser = ArgumentParser()
    
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    parser.add_argument('--mode', '--m', 
    help = 'mode for the PSF: 1. image (1 PSF); 2. ??? - 3D PSF Cube (x,y,lambda)')

    # Noise 
    parser.add_argument('--noise', '--n', dest = 'noise',
    help = 'Add noise', action = 'store_true')

    # Store commend line arguments to args 
    args = parser.parse_args()

    # keywords for the noise
    noise = args.noise
    mode = args.mode

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
        psf, strehl = gen_psf(aosys, psfParam, noise)

        # Add strehl to the PSF parameters
        psfParam['strehl'] = strehl

        save_muse_fits(psf, psfParam, output_path, output_fname, mode)
        
    elif (args.mode == 'cube' and input_file == True): 
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
                "wvl": psfParam['wvl'][i]*1e-9, 
                "flux(amp)": psfParam['flux(amp)'][i], 
                "flux(bck)": psfParam['flux(bck)'][i], 
                # Need to change the wvl to the bluest in here --> to avoid the jump
            }

            _psf, _strehl = gen_psf(aosys, _df, noise)
            psf.append(_psf)

        # Convert  
        psf = np.asarray(psf)
        save_muse_fits(psf, psfParam, output_path, output_fname, mode)

    elif (args.mode ==  'batch' and input_file == True):
        _strehl_list = []
        
        for i in range (len(psfParam['wvl'])):
            _fname = output_fname + '_'+ str(i)

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
                "wvl": psfParam['wvl'][i]*1e-9, 
                "flux(amp)": psfParam['flux(amp)'][i], 
                "flux(bck)": psfParam['flux(bck)'][i], 
                "strehl": 0. 
            }

            # TODO - Add the strehl ratio!
            _psf, _strehl = gen_psf(aosys, _df, noise)
            _df['strehl'] = _strehl

            _strehl_list.append(_strehl)
        
            print("Wavelength:",psfParam['wvl'])


            save_muse_fits(_psf, _df, output_path, _fname, mode)

        psfParam['strehl'] = _strehl_list
        psfParam.to_csv(config_file.get('path', 'input_param'))
        
    else: 
        # In theory, you can use the theoretical relationship to derive the r0 
        # But things get a little more complicated with other parameter 
        # Like alpha and beta, but you should be able to fix them since they are small (?)
        raise Exception ('Without .csv file as input for spectral cube is not supported yet')

    pass


if __name__ ==  "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    print("Run Time (min): ",runtime/60)
    print("\nEnd of Programme\n")
