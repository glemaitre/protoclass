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
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
### Module to performed parallel processing
import multiprocessing
# OS library
import os
from os.path import join, isdir, isfile
# Import namedtuple
from collections import namedtuple

def OpenOneSerieDCM(path_to_serie, reverse=False):
    """Function to read a single serie DCM to return a 3D volume

    Parameters
    ----------
    path_to_serie: str
        The path to the folder containing all the dicom images.
    reverse: bool
        Since that there is a mistake in the data we need to flip in z the gt.
        Have to be corrected in the future.
    
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
    
    im_numpy_cp = im_numpy.copy()
    if reverse == True:
        #print 'Inversing the GT'
        for sl in range(im_numpy.shape[2]):
            im_numpy[:,:,-sl] = im_numpy_cp[:,:,sl]

    
    return im_numpy.astype(float)

def OpenVolumeNumpy(filename, reverse_volume=False):
    """Function to read a numpy array previously saved

    Parameters
    ----------
    filename: str
        Filename of the numpy array *.npy.
    reverse_volume: bool
        Since that there is a mistake in the data we need to flip in z the gt.
        Have to be corrected in the future.
    
    Returns
    -------
    im_numpy: ndarray
        A 3D array containing the volume.
    """

    # Open the volume
    im_numpy = np.load(filename)

    # Copy the volume temporary
    im_numpy_cp = im_numpy.copy()
    if reverse_volume == True:
        #print 'Inversing the GT'
        for sl in range(im_numpy.shape[2]):
            im_numpy[:,:,-sl] = im_numpy_cp[:,:,sl]

    return im_numpy


def OpenSerieUsingGTDCM(path_to_data, path_to_gt, reverse_gt=True, reverse_data=False):
    """Function to read a DCM volume and apply a GT mask

    Parameters
    ----------
    path_to_data: str
        Path containing the modality data.
    path_to_gt: str
        Path containing the gt.
    reverse_gt: bool
        Since that there is a mistake in the data we need to flip in z the gt.
        Have to be corrected in the future.
    
    Returns
    -------
    volume_data: ndarray
        A 3D array containing the volume extracted from the DCM serie.
        The data not corresponding to the GT of interest will be tagged NaN.
    """

    # Open the data volume
    volume_data = OpenOneSerieDCM(path_to_data, reverse_data)

    # Open the gt volume
    tmp_volume_gt = OpenOneSerieDCM(path_to_gt)
    volume_gt = tmp_volume_gt.copy()
    if reverse_gt == True:
        #print 'Inversing the GT'
        for sl in range(volume_gt.shape[2]):
            volume_gt[:,:,-sl] = tmp_volume_gt[:,:,sl]

    # Affect all the value which are 0 in the gt to NaN
    volume_data[(volume_gt == 0).nonzero()] = np.NaN

    # Return the volume read
    return volume_data

def OpenDataLabel(path_to_data):
    """Function to read data and label form an *.npz file

    Parameters
    ----------
    path_to_serie: str
        The path to the *.npz file.
    
    Returns
    -------
    data: ndarray
        A list of 2D matrix containing the data.
    label: ndarray
        A list of 1D vector containing the label associated to the data matrix.
    """
    
    if not (isfile(path_to_data) and path_to_data.endswith('.npz')):
        # Check that the path is in fact a file and npz format
        raise ValueError('protoclass.tool.OpenDataLabel: An *.npz file is expected.')
    else:
        # The file can be considered
        npzfile = np.load(path_to_data)

        # return the desired variable
        return (npzfile['data'], npzfile['label'])

def GetGTSamples(path_to_gt, reverse_gt=True, pos_value=255.):
    """Function to return the samples corresponding to the ground-truth

    Parameters
    ----------
    path_to_gt: str
        Path containing the gt.
    reverse_gt: bool
        Since that there is a mistake in the data we need to flip in z the gt.
        Have to be corrected in the future.
    reverse_gt: numeric or bool
        Value considered as the positive class. By default it is 255., but it could be
        1 or True
    
    Returns
    -------
    idx_gt: ndarray
        A 3D array containing the volume extracted from the DCM serie.
        The data not corresponding to the GT of interest will be tagged NaN.
    """

    # Open the gt volume
    tmp_volume_gt = OpenOneSerieDCM(path_to_gt)
    volume_gt = tmp_volume_gt.copy()
    if reverse_gt == True:
        #print 'Inversing the GT'
        for sl in range(volume_gt.shape[2]):
            volume_gt[:,:,-sl] = tmp_volume_gt[:,:,sl]

    # Get the samples that we are interested with
    return np.nonzero(volume_gt == pos_value)

def VolumeToLabelUsingGT(volume, path_to_gt, reverse_gt=True):

    return BinariseLabel(volume[GetGTSamples(path_to_gt, reverse_gt)])

def OpenResult(path_to_result):
    """Function to read results: label and roc information

    Parameters
    ----------
    path_to_result: str
        Path containing the filename of the result file.
    
    Returns
    -------
    pred_label: 1D array
        The label results for the patient considered as test.
    roc: namedtuple
        A named tuple such as roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])
    """
    
    # The results are saved into a npz file
    if not (isfile(path_to_result) and path_to_result.endswith('.npz')):
        raise ValueError('protoclass.tool.dicom_manip: The result file is not an *.npz file')
    else:
        # Load the file
        npzfile = np.load(path_to_result)

        # Define our namedtuple
        roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])
        roc = roc_auc._make(npzfile['roc'])
        pred_label = npzfile['pred_label']

        return (pred_label, roc)

def __VolumeMinMax__(path_patient):
    """Private function in order to return min max of a 3D volume

    Parameters
    ----------
    path_patient: str
        Path where the data are localised.
    
    Returns
    -------
    (min_int, max_int): tuple
        Return a tuple containing the minimum and maximum for the patient.
    """

    # Check if we have either a file or a directory
    if isdir(path_patient):
        # Read a volume for the current patient
        volume = OpenOneSerieDCM(path_patient)
    elif isfile(path_patient):
        volume = OpenVolumeNumpy(path_patient)

    # Return a tuple with the min and max
    return(np.min(volume), np.max(volume))

def FindExtremumDataSet(path_to_data, **kwargs):
    """Function to find the minimum and maximum intensities
       in a 3D volume

    Parameters
    ----------
    path_to_data: str
        Path containing the modality data.
    modality: str
        String containing the name of the modality to treat.
    
    Returns
    -------
    (min_int, max_int): tuple
        A tuple containing the minimum and the maximum intensities.
    """
    
    # Define the path to the modality
    path_modality = kwargs.pop('modality', 'T2W')

    # Create a list with the path name
    path_patients = []
    for dirs in os.listdir(path_to_data):
        # Create the path variable
        path_patient = join(path_to_data, dirs)
        path_patients.append(join(path_patient, path_modality))
       
    # Compute the Haralick statistic in parallel
    num_cores = multiprocessing.cpu_count()
    # Check if we have original DICOM or Numpy volume
    min_max_list = Parallel(n_jobs=num_cores)(delayed(__VolumeMinMax__)(path) for path in path_patients)
    # Convert the list into numpy array
    min_max_array = np.array(min_max_list)

    return (np.min(min_max_array), np.max(min_max_array))

def BinariseLabel(label):
    """Function to find the minimum and maximum intensities
       in a 3D volume

    Parameters
    ----------
    label: array
        Array with values usually 0. and 255. .

    Returns
    -------
    label: array
        Array with values either -1. or 1. .
    """

    label[np.nonzero(label>0)] = 1.
    label = label * 2. - 1.

    return label
