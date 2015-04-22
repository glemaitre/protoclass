#title           :extraction.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/20
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
import numpy as np

import SimpleITK as sitk

from sklearn.feature_extraction import image

def OpenOneSerieDCM(path_to_serie):
    """Function to read a single serie DCM to return a 3D volume

    Parameters
    ----------
    path_to_serie: str
        The path to the folder containing all the dicom images.
    
    Returns
    -------
    im_numpy: ndarray
        A 3D array containing the volume extracted from the DCM serie 
    """
    
    # Define the object in order to read the DCM serie
    reader = sitk.ImageSeriesReader()

    # Get the DCM filenames of the serie
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_serie)

    # Set the filenames to read
    reader.SetFileNames(dicom_names)

    # Build the volume from the set of 2D images
    im = reader.Execute()

    # Convert the image into a numpy matrix
    im_numpy = sitk.GetArrayFromImage(im)

    # The Matlab convention is (Y, X, Z)
    # The Numpy convention is (Z, Y, X)
    # We have to swap these axis
    ### Swap Z and X
    im_numpy = np.swapaxes(im_numpy, 0, 2)
    im_numpy = np.swapaxes(im_numpy, 0, 1)
    
    return im_numpy

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
        patches = Extract2DPatches(im, win_size)
    elif nd_im == 3:
        # Implement a 3D extraction method

    # Compute Haralick
    
    return patches

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

def BuildMap(patches):
    
    
