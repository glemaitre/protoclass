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

path_to_t2 = '/home/lemaitre/Documents/Data/experiments/Patient 383/T2W'
volume = OpenOneSerieDCM(path_to_t2)
