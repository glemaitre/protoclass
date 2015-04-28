#title           :normalisation.py
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
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing

def NormalisationT2W():
    """Function to perform normalisation on T2W images

    Parameters
    ----------
    im: ndarray of int
        2D or 3D array containing the image information
    win_size: tuple (optional)
        Tuple containing 2 or 3 values defining the window size
        to consider during the extraction
    n_gray_levels: int (optional)
        Number of gray level to use to rescale the original image
    gray_limits: tuple (optional)
        Tuple containing 2 values defining the minimum and maximum
        gray level.

    Returns
    -------
    maps: ndarray of np.double
        If 2D image - maps is of size 4 x 14 x image height x image width)
        which will contain the map corresponding to the different 
        orientations and statistic of Haralick features.
    
    """
    
