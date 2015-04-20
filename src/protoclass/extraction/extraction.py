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


def OpenOneSerieDCM(path_to_serie):
    """Function to read a single serie DCM to return a 3D volume"""
    
    # Define the object in order to read the DCM serie
    reader = sitk.ImageSeriesReader()

    # Get the DCM filenames of the serie
    dicom_names = reader.GetGDCMSeriesFileNames(path_to_serie)

    # Set the filenames to read
    reader.SetFileNames(dicom_names)

    # Build the volume from the set of 2D images
    image = reader.Execute()

    # Convert the image into a numpy matrix
    image_numpy = sitk.GetArrayFromImage(image)

    # The Matlab convention is (Y, X, Z)
    # The Numpy convention is (Z, Y, X)
    # We have to swap these axis
    ### Swap Z and X
    image_numpy = np.swapaxes(image_numpy, 0, 2)
    image_numpy = np.swapaxes(image_numpy, 0, 1)
    return image_numpy
