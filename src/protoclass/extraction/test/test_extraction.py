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
from protoclass.extraction.texture_analysis import Build2DPatch
from protoclass.extraction.texture_analysis import ReshapePatchsToMaps

path_to_t2 = '/work/le2i/gu5306le/experiments/Patient 383/T2W'
volume = OpenOneSerieDCM(path_to_t2)

im_2d = volume[:, :, 35]
tp_win_size = (9,9)
maps = HaralickMapExtraction(im_2d)
