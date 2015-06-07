#title           :denoising-non-local-means.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/06/06
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
### Module to performed parallel processing
import multiprocessing
# OS library
import os
from os.path import join, isdir, isfile
# sys library
import sys

from protoclass.tool.dicom_manip import OpenRawImageOCT
from protoclass.preprocessing.denoising import Denoising3D

# Get the path to file
filename_data = sys.argv[1]
print 'Opening the following file: {}'.format(filename_data)

# Open the data
if not filename_data.endswith('.img'):
    raise ValueError('denoising-non-local: The image in input is not a raw image.')
else:
    # Read the volume using the raw image
    vol = OpenRawImageOCT(filename_data, (512, 128, 1024))

    # Apply the filtering using 8 cores
    num_cores = 8
    vol_denoised = Denoising3D(vol, denoising_method='non-local-means', num_cores=num_cores)
    #vol_denoised = Denoising3D(vol, denoising_method='non-local-means')
    

    # Directory where to save the data
    storing_folder = sys.argv[2]

    # Create the folder if it is not existing
    if not os.path.exists(storing_folder):
        os.makedirs(storing_folder)

    # Get only the filename without path directory of the input file
    _, filename_patient = os.path.split(filename_data) 
    
    # Get the input filename without .img
    filename_root, _ = os.path.splitext(filename_patient)

    # Get the filename for numpy and matlab
    filename_matlab = os.path.join(storing_folder, filename_root + '_nlm.mat')
    filename_numpy = os.path.join(storing_folder, filename_root + '_nlm.npz')

    # Save the matfile
    from scipy.io import savemat
    savemat(filename_matlab, {'vol_denoised': vol_denoised})

    # Save the numpy array
    np.savez(filename_numpy, vol_denoised=vol_denoised)
