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
print 'Opening the following file: {}'.format(filename_data)

# Get the index of the patient to use for testing
idx_testing_patient = int(sys.argv[2])

# Read the data
data, label = OpenDataLabel(filename_data)

# Split into training and testing sets
testing_data = data[idx_testing_patient]
testing_label = label[idx_testing_patient]

# Generate the training patient index
idx_training_patient = np.arange(data.shape[0])
idx_training_patient = np.delete(idx_training_patient, idx_testing_patient)

training_data = data[idx_training_patient[0]]
training_label = label[idx_training_patient[0]]
idx_training_patient = np.delete(idx_training_patient, 0)
for p in idx_training_patient:
    training_data = np.concatenate((training_data, data[p]))
    training_label = np.concatenate((training_label, label[p]))

# Delete the original data and label just to take less memory
del data
del label

print '----- DATA READ -----'
print 'Training data: {}'.format(training_data.shape)
print 'Testing data: {}'.format(testing_data.shape)
print '---------------------'

# Run the classification for this specific data
pred_label, roc = Classify(training_data, training_label, testing_data, testing_label, classifier_str='random-forest', n_estimators=1000)

# Save the results somewhere
path_to_save = sys.argv[3]

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

saving_filename = 'result_' + str(idx_testing_patient) + '.npz'
saving_path = join(path_to_save, saving_filename)
np.savez(saving_path, pred_label=pred_label, roc=roc)
