#title           :test_extraction.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/20
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
import matplotlib.pyplot as plt
import numpy as np

from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.extraction.texture_analysis import HaralickMapExtraction

# Give the path to a patient
path_to_t2 = '/work/le2i/gu5306le/experiments/Patient 383/T2W'
# Read a volume
volume = OpenOneSerieDCM(path_to_t2)
# Select a slice
im_2d = volume[:, :, 35]
# Define the parameters for glcm
### Window size
tp_win_size = (9,9)
### Number of gray levels
tp_n_gray_levels = 8
### Compute the Haralick maps
maps = HaralickMapExtraction(im_2d, win_size=tp_win_size, n_gray_levels=tp_n_gray_levels)
# Save the maps
np.save('../data/maps.npy', maps)
