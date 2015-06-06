#title           :denoising.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/06/06
#version         :0.1
#notes           :
#python_version  :2.7.6 
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Matplotlib library
import matplotlib.pyplot as plt
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing

def Denoising3D(volume, denoising_method='non-local-mean', **kwargs):

    if denoising_method == 'non-local-mean':
        return DenoisingNLM3D(volume, **kwargs)

def DenoisingNLM3D(volume, **kwargs):
    # GOAL: denoise a 3D image and return the denoised image using NLM

    # Import the function to apply nl means in 3D images
    from skimage.restoration import nl_means_denoising

    # Get the parameters for the denoising
    min_dim = float(min(volume.shape))


    patch_size = kwargs.pop('patch_size', int(np.ceil(min_dim / 10.)))
    patch_distance = kwargs.pop('patch_distance', int(np.ceil(min_dim / 4.)))
    h = kwargs.pop('h', 0.1)
    multichannel = kwargs.pop('multichannel', False)
    fast_mode = kwargs.pop('fast_mode', True)
    
    # Perform the denoising
    return nl_means_denoising(volume, patch_size=patch_size, patch_distance=patch_distance, h=h, 
                              multichannel=multichannel, fast_mode=fast_mode)


def Denoising2D(image, denoising_method='non-local-mean', **kwargs):

    if denoising_method == 'non-local-mean':
        return DenoisingNLM2D(image, **kwargs)

def DenoisingNLM2D(image, **kwargs):
    # GOAL: denoise a 2D image and return the denoised image using NLM

    # Import the function to apply nl means in 2D images
    from skimage.restoration import nl_means_denoising

    # Get the parameters for the denoising
    min_dim = float(min(image.shape))

    patch_size = kwargs.pop('patch_size', int(np.ceil(min_dim / 30.)))
    patch_distance = kwargs.pop('patch_distance', int(np.ceil(min_dim / 15.)))
    h = kwargs.pop('h', 0.04)
    multichannel = kwargs.pop('multichannel', False)
    fast_mode = kwargs.pop('fast_mode', True)
    
    print 'Parameters {} - {} - {} - {} - {}'.format(patch_size, patch_distance, h, multichannel, fast_mode)

    img_den = nl_means_denoising(image, patch_size=patch_size, patch_distance=patch_distance,
                              h=h, multichannel=multichannel, fast_mode=fast_mode)

    # Perform the denoising
    return img_den
