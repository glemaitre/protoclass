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
    vol = OpenRawImageOCT(filename_data)

    # Apply the filtering using 8 cores
    num_cores = 8
    vol_denoised = Denoising3D(vol, denoising_method='non-local-means', num_cores=num_cores)

    # Get the input filename without .img
    filename_root, _ = os.path.splitext(filename_data)

    # Get the filename for numpy and matlab
    filename_matlab = filename_root + '.mat'
    filename_numpy = filename_root + '.npz'

    # Save the matfile
    from scipy.io import savemat
    savemat(filename_matlab, {'vol_denoised': vol_denoised})

    # Save the numpy array
    np.save(filename_numpy, vol_denoised=vol_denoised)
