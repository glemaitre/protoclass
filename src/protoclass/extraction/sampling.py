#title           :sampling.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/12
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

from protoclass.tool.dicom_manip import OpenVolumeNumpy
from protoclass.tool.dicom_manip import GetGTSamples

def SamplingVolumeFromGT(path_to_volume, path_to_gt):
    # TODO: write the help for this function
    # GOAL: extract the pixels from the data
    ### Return the matrix 
    ### Return the label vector
    ### We work for a single patient - not the time to send the size of the volume

    # Load the volume
    volume_mod = OpenVolumeNumpy(path_to_volume)

    # Get the data correspoding to the prostate
    array_gt = GetGTSamples(path_to_gt)

    # Return a vector 
    return volume_mod[array_gt]

def SamplingHaralickFromGT(path_to_haralick, path_to_gt, vec_angle=np.arange(4), vec_feat=np.arange(13)):
    # TODO: write the help for this function
    # GOAL: Extract the data from the haralick volume
    ### We work for a single patient - not the time to send the size of the volume

    matrix_haralick = []
    for a in vec_angle:
        for f in vec_feat:
            volume_filename = 'volume_' + str(a) + '_' + str(f)
            matrix_haralick.append(SamplingVolumeFromGT(join(path_to_haralick, volume_filename), path_to_gt))

    return np.array(matrix_haralick)
