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

from protoclass.extraction.sampling import SamplingVolumeFromGT
from protoclass.extraction.sampling import SamplingHaralickFromGT

data = SamplingVolumeFromGT('/work/le2i/gu5306le/experiments/Patient 1036/haralick/volume_0_0.npy', '/work/le2i/gu5306le/experiments/Patient 1036/GT/prostate')
