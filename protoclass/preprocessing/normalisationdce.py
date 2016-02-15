#title           :normalisationdce.py
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
# Scipy library
from scipy.integrate import simps
### Scipy library for Gaussian statistics
from scipy.stats import norm
### Scipy library for Rician statistics
from scipy.stats import rice
### Scipy library for curve fitting
from scipy.optimize import curve_fit
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing
