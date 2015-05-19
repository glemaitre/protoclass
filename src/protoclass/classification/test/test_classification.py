#title           :test_classiciation.py
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

from protoclass.classification.classification import BalancingTraining

# Generate a vector 
label = np.array([-1]*100 + [1]*20)
data = np.random.random((label.shape[0], 5))

# Check if the selection is done properly
data, label = BalancingTraining(data, label)
