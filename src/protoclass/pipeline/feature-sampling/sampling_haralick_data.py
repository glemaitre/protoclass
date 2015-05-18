#title           :sampling_haralick_data.py
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
# SYS library
import sys

# The following argument will be important
### sys.argv[1] = path in which all patients folders are located
### sys.argv[2] = path to consider for haralick -> 'haralick', 'haralick_rnorm', 'haralick_gnorm'
### sys.argv[3] = path where to save the data

from protoclass.extraction.sampling import SamplingHaralickFromGT
from protoclass.extraction.sampling import SamplingVolumeFromGT
from protoclass.tool.dicom_manip import BinariseLabel

# Get the path where all the patients are stored
path_patients = sys.argv[1]
path_to_haralick = sys.argv[2]
path_to_GT_prostate = "GT/prostate"
path_to_GT_cap = "GT/cap"
path_to_exp = sys.argv[3]

data=[]
label=[]
# Go through each patient directory
for dirs in os.listdir(path_patients):
    
    # Get the path for the patient
    path_patient = join(path_patients, dirs)

    # Print the current path
    print 'Processing path: {}'.format(path_patient)

    # Get the path to the haralick volume for this patient
    path_patient_haralick = join(path_patient, path_to_haralick)
    # Get the path to the prostate GT for this patient
    path_patient_GT_prostate = join(path_patient, path_to_GT_prostate)
    # Get the path to the cap GT for this patient
    path_patient_GT_cap = join(path_patient, path_to_GT_cap)

    # Extract the data for this specific patient - Haralick
    ### No other parameter since that we need all the haralick
    data.append(SamplingHaralickFromGT(path_patient_haralick, path_patient_GT_prostate))

    # Extract the label for this specific patient
    label.append(BinariseLabel(SamplingVolumeFromGT(path_patient_GT_cap,
                                                    path_patient_GT_prostate,
                                                    reverse_volume=True)))

# Convert into np array
data = np.array(data)
label = np.array(label)

# Create a directory if not existing
if not os.path.exists(path_to_exp):
    os.makedirs(path_to_exp)

# We will save the data and the label in the same file to ensure that we have everything properly embedded    
filename = join(path_to_exp, 'data.npz')
# Save the data with their keyword name
np.savez(filename, data=data, label=label)
