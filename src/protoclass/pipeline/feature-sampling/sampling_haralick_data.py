#title           :haralick_sampling.py
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

# The purpose will get all the data. We need to save:
### A data matrix of size N x M -> data
### A vector of size N -> label
### A list with the patient information regarding the indexes

from protoclass.extraction.sampling import SamplingHaralickFromGT
from protoclass.extraction.sampling import SamplingVolumeFromGT

# Get the path where all the patients are stored
path_patients = sys.argv[1]
path_to_haralick = "haralick"
path_to_GT_prostate = "GT/prostate"
path_to_GT_cap = "GT/cap"

# Go through each patient directory
for dirs in os.listdir(path_to_data):
    
    # Get the path for the patient
    path_patient = join(path_patients, dirs)

    # Get the path to the haralick volume for this patient
    path_patient_haralick = join(path_patient, path_to_haralick)
    # Get the path to the prostate GT for this patient
    path_patient_GT_prostate = join(path_patient, path_to_GT_prostate)
    # Get the path to the cap GT for this patient
    path_patient_GT_cap = join(path_patient, path_to_GT_cap)

    # Extract the data for this specific patient - Haralick
    ### No other parameter since that we need all the haralick
    SamplingHaralickFromGT(path_patient_haralick, 
                           path_to_gt)


### Example to get data for one patient
data = SamplingVolumeFromGT('/DATA/prostate/public/Siemens/Patient 1036/volume_0_0.npy', '/DATA/prostate/public/Siemens/Patient 1036/T2WSeg/prostate')

label = BinarizeLabel(SamplingVolumeFromGT('/DATA/prostate/public/Siemens/Patient 1036/T2WSeg/cap', '/DATA/prostate/public/Siemens/Patient 1036/T2WSeg/prostate', reverse_volume=True))

