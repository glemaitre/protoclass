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
# Matplotlib library
import matplotlib.pyplot as plt
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
### Module to performed parallel processing
import multiprocessing
# OS library
import os
from os.path import join
# SYS library
import sys

# Our module
from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.tool.dicom_manip import OpenSerieUsingGTDCM
from protoclass.preprocessing.normalisation import GaussianNormalisation
from protoclass.preprocessing.normalisation import RicianNormalisation

#################################################################################
## GAUSSIAN NORMALISATION



#################################################################################
## RICIAN NORMALISATION
