#title           :edge_analysis.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/20
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Scipy library
import scipy as sp
### Scipy generic gradient magnitude computation
from scipy.ndimage.filters import generic_gradient_magnitude, generic_laplace
### Scipy edge detector
from scipy.ndimage.filters import prewitt, sobel, laplace
# Scikit-learn
### Module to extract 2D patches
from sklearn.feature_extraction import image
# Mahotas library
### Module to extract haralick features
from mahotas.features import haralick
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing
# Operator library
import operator

def EdgeMapExtraction(im, **kwargs):
    # TODO: write important information
    # GOAL: return a volume containing
    ### edge_detector: can be 'Sobel1stDev', 'Prewitt1stDev', 'Sobel2ndDev', 'Prewitt2ndDev'

    # Check the dimension of the input image
    if len(im.shape) == 2:
        nd_im = 2
    elif len(im.shape) == 3:
        nd_im = 3
    else:
        raise ValueError('mahotas.edge: Can only handle 2D and 3D images.')

    # Assign the edge detector
    edge_detector= kwargs.pop('edge_detector', 'Sobel1stDev')
    detector_list = ['Sobel1stDev', 'Prewitt1stDev', 'Sobel2ndDev', 'Prewitt2ndDev']
    if not any(edge_detector in dl for dl in detector_list):
        raise ValueError('mahotas.edge: The name of the detector is unknown.')

    if edge_detector == 'Sobel1stDev':
        edge_im = generic_gradient_magnitude(im, sobel)
    elif edge_detector == 'Prewitt1stDev':
        edge_im = generic_gradient_magnitude(im, prewitt)
    elif edge_detector == 'Sobel2ndDev':
        edge_im = generic_laplace(im, sobel)
    elif edge_detector == 'Prewitt2ndDev':
        edge_im = generic_laplace(im, prewitt)

    return edge_im
