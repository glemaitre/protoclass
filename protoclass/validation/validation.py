#title           :validation.py
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

from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.tool.dicom_manip import OpenVolumeNumpy
from protoclass.tool.dicom_manip import GetGTSamples
from protoclass.tool.dicom_manip import OpenResult

def ResultToLabel(path_to_result):
    # Check if the result is an npz file
    if not (isfile(path_to_result) and path_to_result.endswith('.npz')):
        raise ValueError('protoclass.validation.ResultVolume: The result file should be an *.npz file.')
    else:
        # Now we have all the information in order to restruct the stored results
        pred_label, roc = OpenResult(path_to_result)

        return pred_label

def ResultToVolume(path_to_result, path_to_gt, reverse_gt=True):
    
    # We have to open the gt volume in order to know the information
    # about the size of the image

    # Check if the result is an npz file
    if not (isfile(path_to_result) and path_to_result.endswith('.npz')):
        raise ValueError('protoclass.validation.ResultVolume: The result file should be an *.npz file.')
    else:

        # Check if we have either a file or a directory
        if isdir(path_to_gt):
            # Read a volume for the current patient
            volume_gt = OpenOneSerieDCM(path_to_gt, reverse_gt)
        elif isfile(path_to_gt):
            volume_gt = OpenVolumeNumpy(path_to_gt)

        # Get only the samples that are consider prostate samples
        array_gt = GetGTSamples(path_to_gt, reverse_gt)
    
        # Now we have all the information in order to restruct the stored results
        pred_label, roc = OpenResult(path_to_result)

        # Restructure the pred_label into a volume
        # Initialise the volume to the right size
        volume_out = np.zeros(volume_gt.shape)
        # Affect the label to right voxels
        volume_out[array_gt] = pred_label
            
        return volume_out

def OpenROCPatients(path_to_results):
    
    # Check that the path is in fact a directory
    if not isdir(path_to_results):
        raise ValueError('protoclass.validation.validation: The path given is not a directory')
    else:
        # Read each file from the directory
        roc = []
        for f in os.listdir(path_to_results):
            if isfile(join(path_to_results, f)) and f.endswith('.npz'):
                # Read the file
                tmp_pred_label, tmp_roc = OpenResult(join(path_to_results, f))
                roc.append(tmp_roc)

        return roc

def PlotROCPatients(rocs):

    fig = plt.figure()
    mk = ['.', 
          ',', 
          'o', 
          'v', 
          '^', 
          '<', 
          '>', 
          '1', 
          '2', 
          '3', 
          '4', 
          '8', 
          's', 
          'p', 
          '*', 
          'h', 
          'H', 
          '+', 
          'x', 
          'D', 
          'd' ]

    for roc in rocs:
        plt.plot(roc.fpr, roc.tpr, mk[np.random.randint(20)], ls='-', label='AUC={:.2f}'.format(roc.auc), markevery=50)

    # Put some x and y labels
    plt.xlabel('False positive rate')
    plt.ylabel('True negative rate')
    # Put the legend on the bottom left
    plt.legend(loc=4)
    plt.show()
