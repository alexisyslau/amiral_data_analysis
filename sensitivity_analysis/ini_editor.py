# Packages required
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
import os
from amiral import instructment, utils, parameter, config, array
from amiral.extension import preproc_muse
import argparse
from argparse import ArgumentParser
import configparser
import time
import yaml
import json
import pandas as pd

def edit_ini_file (output_dir, output_fname, r0, sig2, data_path, data_csv, output_file, output_crit_file, output_path):

    fname = output_dir+output_fname
    
    ini_config = configparser.ConfigParser()
    ini_config.read(fname)

    # Editing the ini file
    ini_config.set('psf parameter','r0', str(r0))
    ini_config.set('psf parameter','amplitude', str(sig2))

    ini_config.set('path','data_path', data_path)
    ini_config.set('path','data_csv', data_csv)
    ini_config.set('path','output_file', output_file)
    ini_config.set('path','output_crit_file', output_crit_file)
    ini_config.set('path','output_path', output_path)

    # set the output csv_file
    with open(output_dir+output_fname, 'w') as configfile:
        ini_config.write(configfile)
    pass 

def copy_ini_file (input_dir, input_fname, output_dir,output_fname): 
    os.system("cp {input} {output}".format(input=input_dir+input_fname, output = output_dir+output_fname))
    print("cp {input} {output}".format(input=input_dir+input_fname, output = output_dir+output_fname))
    pass 

def main ():
    
    # Setting flags and cmd arugments for the scipt
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', 
    help = 'the config file which contains path of the data and the name of the file')

    # Store commend line arguments to args 
    args = parser.parse_args()

    # Load .ini file as an object
    ini_config = config.load_config(args)

    # Getting the variables
    input_dir = ini_config.get('path', 'input_dir')
    output_dir = ini_config.get('path', 'output_dir')
    data_path = ini_config.get('path', 'data_path')
    output_path = ini_config.get('path', 'output_path')

    csv_fname = ini_config.get('file name', 'csv_fname')
    input_ini = ini_config.get('file name', 'input_ini')
    
    data_csv = ini_config.get('file name', 'data_csv')
    output_file  = ini_config.get('file name', 'output_file')
    output_crit_file = ini_config.get('file name', 'output_crit_file')

    input_ini += ".ini"
    output_fname = ini_config.get('file name', 'output_ini')

    data_input = pd.read_csv(csv_fname)
    print("Read parameters from %s" %(csv_fname))

    data_fname = data_input['Unnamed: 0']
    r0 = data_input['r0']
    sig2 = data_input['amplitude']

    for i in range (len(data_fname)): 
        output_ini = output_fname + "_"+ str(data_fname[i]) + ".ini"
        print(output_ini)
        copy_ini_file(input_dir, input_ini, output_dir,output_ini)

        _data_path = data_path + 'case_' + str(data_fname[i]) + '/'
        _data_csv = data_csv + "_"+ str(data_fname[i]) 
        _output_file = output_file + "_"+ str(data_fname[i]) 
        _output_crit_file = output_crit_file + "_"+ str(data_fname[i]) 
        _output_path = output_path + 'case_' + str(data_fname[i]) + '/'

        edit_ini_file(output_dir, output_ini, r0[i], sig2[i], _data_path,_data_csv,_output_file,_output_crit_file,_output_path)

    pass

# call the main
if __name__ ==  "__main__":
    print("\n============   .ini file editor    ============\n")
    main()
    print("\n============   End of Programme    ============\n")







