#title           :test_normalisation.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/26
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Joblib library
### Module to performed parallel processing
import multiprocessing
# OS library
import os
from os.path import join

# Our module to read DCM volume from a serie of DCM
from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.tool.dicom_manip import OpenSerieUsingGTDCM

# Give the path to a patient
path_to_data = '/work/le2i/gu5306le/experiments'
#path_to_data = '/home/lemaitre/Documents/Data/experiments'
path_t2w = 'T2W'
path_gt = ''
path_haralick = 'haralick'

# Go through all the patients directories
for dirs in os.listdir(path_to_data):
    path_patient = join(path_to_data, dirs)
    path_dcm = join(path_patient, path_t2w)
    print 'Reading data from the directory {}'.format(path_dcm)

    # Read a volume
    volume = OpenOneSerieDCM(path_dcm)

