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
# itertools library
import itertools

from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.tool.dicom_manip import OpenVolumeNumpy
from protoclass.tool.dicom_manip import GetGTSamples

def SamplingVolumeFromGT(path_to_volume, path_to_gt, reverse_volume=False, reverse_gt=True):
    # TODO: write the help for this function
    # GOAL: extract the pixels from the data
    ### Return the matrix 
    ### Return the label vector
    ### We work for a single patient - not the time to send the size of the volume

    print path_to_volume

    # Check if we have either a file or a directory
    if isdir(path_to_volume):
        # Read a volume for the current patient
        volume_mod = OpenOneSerieDCM(path_to_volume, reverse_volume)
    elif isfile(path_to_volume):
        volume_mod = OpenVolumeNumpy(path_to_volume, reverse_volume)

    # Get the data correspoding to the prostate
    array_gt = GetGTSamples(path_to_gt, reverse_gt)

    # Return a vector 
    return volume_mod[array_gt]

def ParallelProcessingHaralick(it, path_h, path_gt):

    volume_filename = 'volume_' + str(it[0]) + '_' + str(it[1]) + '.npy'
    return SamplingVolumeFromGT(join(path_h, volume_filename), path_gt)

def SamplingHaralickFromGT(path_to_haralick, path_to_gt, vec_angle=np.arange(4), vec_feat=np.arange(13)):
    # TODO: write the help for this function
    # GOAL: Extract the data from the haralick volume
    ### We work for a single patient - not the time to send the size of the volume

    # Compute the Haralick statistic in parallel
    num_cores = multiprocessing.cpu_count()
    matrix_haralick = Parallel(n_jobs=num_cores)(delayed(ParallelProcessingHaralick)(p, path_to_haralick, path_to_gt) for p in itertools.product(vec_angle, vec_feat))

    return np.array(matrix_haralick).T
