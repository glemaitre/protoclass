#title           :test_sampling.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/20
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Matplotlib library
import matplotlib.pyplot as plt
# Numpy libreary
import numpy as np
# OS library
import os
from os.path import join

from protoclass.tool.dicom_manip import GetGTSamples
from protoclass.extraction.sampling import SamplingVolumeFromGT
from protoclass.extraction.sampling import SamplingHaralickFromGT

data = SamplingVolumeFromGT('/DATA/prostate/public/Siemens/Patient 1036/volume_0_0.npy', '/DATA/prostate/public/Siemens/Patient 1036/T2WSeg/prostate')

#array = GetGTSamples('/DATA/prostate/public/Siemens/Patient 1036/T2WSeg/prostate')
