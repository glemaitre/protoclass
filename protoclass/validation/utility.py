#title           :utility.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/18
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Scipy library
import scipy as sp
# Scikit-learn library
from sklearn.metrics import confusion_matrix
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing
# OS library
import os
from os.path import join, isdir, isfile
# Matplotlib library 
import matplotlib.pyplot as plt
# import the system package
import sys

def MakeTable(data, featurelist, savepath, filename, ext='.tex'):
    """Function to build a latex table from a numpy matrix.
    Parameters
    ----------
    filename: str
        Name of the file where to save the table
    data: ndarray of size (M, N)
        The data to save into a latex array. 
    featurelist: list of str of size (M,)
        The name of the features in data
    savepath: str
        String indicating the path where to save the tables
    Returns
    -------
    None
    """

    filename = join(savepath, filename+ ext)
    
    fi = open(filename, 'w+')
    for fId in range(0, data.shape[0]): 
       line = []
       line =  featurelist[fId]
       for vId in range(0,data.shape[1]): 
           line = line + '&' + str(round(data[fId][vId]))
       
       line = line + '\\'+ '\\' + '\n'
       fi.write(line)       
    fi.close()
    
    return None
