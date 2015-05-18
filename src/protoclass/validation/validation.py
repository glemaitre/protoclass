#title           :validation.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/18
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Scipy library
import scipy as sp
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing
# OS library
import os
from os.path import join, isdir, isfile

from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.tool.dicom_manip import OpenVolumeNumpy
from protoclass.tool.dicom_manip import GetGTSamples
from protoclass.tool.dicom_manip import OpenResult

def ResultToVolume(path_to_result, path_to_gt, reverse_gt=True):
    
    # We have to open the gt volume in order to know the information
    # about the size of the image

    # Check if the result is an npz file
    if not (isfile(path_to_result) and path_to_result.endswith('.npz')):
        raise ValueError('protoclass.validation.ResultVolume: The result file should be an *.npz file.')
    else:

        # Check if we have either a file or a directory
        if isdir(path_to_gt):
            # Read a volume for the current patient
            volume_gt = OpenOneSerieDCM(path_to_gt, reverse_gt)
        elif isfile(path_to_gt):
            volume_gt = OpenVolumeNumpy(path_to_gt)

        # Get only the samples that are consider prostate samples
        array_gt = GetGTSamples(path_to_gt, reverse_gt)
    
        # Now we have all the information in order to restruct the stored results
        pred_label, roc = OpenResult(path_to_result)

        # Restructure the pred_label into a volume
        # Initialise the volume to the right size
        volume_out = np.zeros(volume_gt.shape)
        # Affect the label to right voxels
        volume_out[array_gt] = pred_label
            
        return volume_out
