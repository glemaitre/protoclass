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
### Scipy filtering
from scipy.ndimage import convolve
# Scikit-image library
### Gabor filtering
from skimage.filters import gabor_kernel
# # Scikit-learn
# ### Module to extract 2D patches
# from sklearn.feature_extraction import image
# # Mahotas library
# ### Module to extract haralick features
# from mahotas.features import haralick
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
    detector_list = ['Sobel1stDev', 'Prewitt1stDev', 'Sobel2ndDev', 'Prewitt2ndDev', 'GaborBank']
    if not any(edge_detector in dl for dl in detector_list):
        raise ValueError('mahotas.edge: The name of the detector is unknown.')

    if edge_detector == 'GaborBank':
        # Extract the value of the parameters
        n_freq = kwargs.pop('n_freq', 10)
        freq_range = kwargs.pop('freq_range', (.05, .2))
        n_theta = kwargs.pop('n_theta', 6)
        win_size = kwargs.pop('win_size', (15., 15.))
        # Generate the different kernel which are needed
        kernels_gabor, kernels_gabor_params = GaborKernelBank(n_freq, freq_range, n_theta, win_size)
        # Extract the maps from Gabor
        edge_im = BuildMaps2D(im, kernels_gabor)
        
        # Return the maps and the parameters of the Gabor kernels
        return (edge_im, kernels_gabor_params)

    if edge_detector == 'Sobel1stDev':
        edge_im = generic_gradient_magnitude(im, sobel)
    elif edge_detector == 'Prewitt1stDev':
        edge_im = generic_gradient_magnitude(im, prewitt)
    elif edge_detector == 'Sobel2ndDev':
        edge_im = generic_laplace(im, sobel)
    elif edge_detector == 'Prewitt2ndDev':
        edge_im = generic_laplace(im, prewitt)

    return edge_im

def GaborFiltering(im, kernel):
    # TODO: write important information
    # GOAL: Function to filer the image using a gabor filter 

    # Compute Haralick feature
    return convolve(im, kernel, mode='wrap')

########################################################################################
### 2D implementation

def GaborKernelBank(n_freq=10, freq_range=(.05, .2), n_theta=6, win_size=(15., 15.)):
    # TODO: write important information
    # GOAL: return the different kernel to use when performing the gabor filtering 

    # Affect the right values for the sigmas
    s_y = (win_size[0] - 1.) / 6. 
    s_x = (win_size[1] - 1.) / 6.

    # Generate the values for the different thetas
    thetas = np.linspace(0, np.pi, int(n_theta))

    # Generate the values for the different frequencies
    freqs = np.linspace(freq_range[0], freq_range[1], int(n_freq))

    kernels = []
    kernels_params = []
    for theta in thetas:
        for freq in freqs:
            kernel = np.real(gabor_kernel(freq, 
                                          theta=theta,
                                          sigma_x=s_x, 
                                          sigma_y=s_y))
            kernels_params.append((freq, theta, s_y, s_x))
            kernels.append(kernel)

    return (kernels, kernels_params)

def BuildMaps2D(im, kernels):
    # TODO: write important information
    # GOAL: Build map from gabor filtering 

    # Compute the Haralick statistic in parallel
    num_cores = multiprocessing.cpu_count()
    return Parallel(n_jobs=num_cores)(delayed(GaborFiltering)(im, k) for k in kernels)
