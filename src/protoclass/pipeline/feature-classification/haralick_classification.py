#title           :haralick_classiciation.py
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
# sys library
import sys

from protoclass.tool.dicom_manip import OpenDataLabel
from protoclass.classification.classification import Classify

# We will perform classification with a LOPO strategy

# We need to have as inputs:
### the path to the data file
### the index of the patient to use for testing

# Get the path to file
filename_data = sys.argv[1]

# Get the index of the patient to use for testing
idx_patient_test = float(sys.argv[2])

# Read the data
data, label = OpenDataLabel(filename_data)

# Split into training and testing sets
testing_data = data[idx_patient_test]
testing_label = label[idx_patient_test]

training_data = np.delete(data, idx_patient_test)
training_label = np.delete(label, idx_patient_test)

# Delete the original data and label just to take less memory
del data
del label

# Run the classification for this specific data
roc = Classify(training_data, training_label, testing_data, testing_label, classifier_str='random-forest', n_estimators=1000)
