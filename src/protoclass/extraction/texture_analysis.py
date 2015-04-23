#title           :texture_analysis.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/20
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
import numpy as np
from sklearn.feature_extraction import image
from mahotas.features import haralick
from joblib import Parallel, delayed
import multiprocessing
import operator


def HaralickMapExtraction(im, **kwargs):
    """Function to extract the Haralick map from a 2D or 3D image

    Parameters
    ----------
    im: ndarray
        2D or 3D array containing the image information
    win_size: tuple
        tuple containing 2 or 3 values defining the window size
        to consider during the extraction
    n_gray_level: int
        number of gray level in order to quantisize

    Returns
    -------
    
    """
    
    # Check the dimension of the input image
    if len(im.shape) == 2:
        nd_im = 2
        win_size = kwargs.pop('win_size', (7, 7))        
    elif len(im.shape) == 3:
        nd_im = 3
        win_size = kwargs.pop('win_size', (7, 7, 7))
    else:
        raise ValueError('mahotas.texture.haralick: Can only handle 2D and 3D images.')

    # Check that win_size is a tuple
    if not isinstance(win_size, tuple):
        raise ValueError('win_size has to be a tuple with 2 or 3 values depending of the image.')

    # Check that nd_im is of the same dimension than win_size
    if nd_im != len(win_size):
        raise ValueError('The dimension of the image do not agree with the window size dimensions: 2D - 3D')

    # Call the 2D patch extractors
    if nd_im == 2:
        # Extract the patches to analyse
        patches = Extract2DPatches(im, win_size)
        # Compute the Haralick maps
        i_h, i_w = im.shape[:2]
        p_h, p_w = win_size[:2]
        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1
        maps = Build2DMaps(patches, n_h, n_w)
    
    return maps

def Extract2DPatches(im, win_size):
    """Function to extract the 2D patches which which will feed haralick

    Parameters
    ----------
    im: ndarray
        2D array containing the image information
    win_size: tuple
        array containing 2 values defining the window size in order to 
        perform the extraction

    Returns
    -------
    
    """

    if len(im.shape) != 2:
        raise ValueError('extraction.Extract2DPatches: The image can only be a 2D image.')
    if len(win_size) != 2:
        raise ValueError('extraction.Extract2DPatches: The win_size can only be a tuple with 2 values.')

    return image.extract_patches_2d(im, win_size)

def Build2DMaps(patches_or_im, i_h, i_w):
    """Function to compute Haralick features for all patch

    Parameters
    ----------

    Returns
    -------
    
    """

    # Compute the Haralick statistic in parallel
    num_cores = multiprocessing.cpu_count()
    patch_arr = Parallel(n_jobs=num_cores)(delayed(HaralickProcessing)(p) for p in patches_or_im)

    # Convert the patches into maps
    return ReshapePatchsToMaps(patch_arr, i_h, i_w)


def HaralickProcessing(patch_in):
    """Function to compute Haralick for a patch

    Parameters
    ----------

    Returns
    -------
    
    """
    
    # Compute Haralick feature
    return haralick(patch_in, compute_14th_feature=True)

def ReshapePatchsToMaps(patches, i_h, i_w):
    """Function to reshape the array of patches into proper maps

    Parameters
    ----------

    Returns
    -------
    
    """

    # Conver the list into a numpy array
    patches_numpy = np.array(patches)

    # Get the current size
    n_patch, n_orientations, n_statistics = patches_numpy.shape

    # Reshape the patches into a map first
    maps = patches_numpy.reshape((i_h, i_w, n_orientations, n_statistics))

    # Swap the good dimension in order to have [Orientation][Statistic][Y, X]
    maps = maps.swapaxes(0, 2)
    maps = maps.swapaxes(1, 3)

    # We would like to have a list for the orientations and a list for the statistics 
    return maps
