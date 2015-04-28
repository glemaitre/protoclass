#title           :dicom_manip.py
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
# SimpleITK library
import SimpleITK as sitk

def OpenOneSerieDCM(path_to_serie):
    """Function to read a single serie DCM to return a 3D volume

    Parameters
    ----------
    path_to_serie: str
        The path to the folder containing all the dicom images.
    
    Returns
    -------
    im_numpy: ndarray
        A 3D array containing the volume extracted from the DCM serie.
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

def OpenSerieUsingGTDCM(path_to_data, path_to_gt):
    """Function to read a DCM volume and apply a GT mask

    Parameters
    ----------
    path_to_data: str
        Path containing the modality data.
    path_to_gt: str
        Path containing the gt.
    
    Returns
    -------
    volume_data: ndarray
        A 3D array containing the volume extracted from the DCM serie.
        The data not corresponding to the GT of interest will be tagged NaN.
    """

    # Open the data volume
    volume_data = OpenOneSerieDCM(path_to_data)

    # Open the gt volume
    volume_gt = OpenOneSerieDCM(path_to_gt)

    # Affect all the value which are 0 in the gt to NaN
    volume_data[(volume_gt == 0).nonzero()] = np.NaN

    # Return the volume read
    return volume_data
