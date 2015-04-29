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
from joblib import Parallel, delayed
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

