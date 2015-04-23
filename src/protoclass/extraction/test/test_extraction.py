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

from protoclass.classification.classification import Classify
from protoclass.extraction.extraction import OpenOneSerieDCM
from protoclass.extraction.extraction import HaralickMapExtraction
from protoclass.extraction.extraction import Init2DMap
from protoclass.extraction.extraction import Build2DMap


path_to_t2 = '/work/le2i/gu5306le/experiments/Patient 383/T2W'
volume = OpenOneSerieDCM(path_to_t2)

im_2d = volume[:, :, 35]
tp_win_size = (5, 5)
patch = HaralickMapExtraction(im_2d)

maps = Init2DMap(im_2d, patch)
maps = Build2DMap(patch, maps)
