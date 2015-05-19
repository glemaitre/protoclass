#title           :test_validation.py
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
# Matplotlib library
import matplotlib.pyplot as plt
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing
# Sys library
import sys
# OS library
import os
from os.path import join

from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.tool.dicom_manip import OpenSerieUsingGTDCM
from protoclass.tool.dicom_manip import GetGTSamples
from protoclass.tool.dicom_manip import BinariseLabel
from protoclass.tool.dicom_manip import VolumeToLabelUsingGT

from protoclass.validation.validation import LabelsToSensitivitySpecificity
from protoclass.validation.validation import ResultToVolume
from protoclass.validation.validation import ResultToLabel
from protoclass.validation.validation import BuildConfusionFromVolume
from protoclass.validation.validation import PlotROCPatients
from protoclass.validation.validation import OpenROCPatients

path_data = '/DATA/prostate/experiments'
patient_list = [ 'Patient 513', 
                 'Patient 383',
                 'Patient 784',
                 'Patient 387',
                 'Patient 996',
                 'Patient 410',
                 'Patient 416',
                 'Patient 430',
                 'Patient 1041',
                 'Patient 1036',
                 'Patient 778',
                 'Patient 804',
                 'Patient 782',
                 'Patient 799',
                 'Patient 634',
                 'Patient 836',
                 'Patient 870']


path_results = '../results/experiments/haralick_gaussian/lopo_results'

for idx, dirs in enumerate(patient_list):
    path_gt = join(path_data, dirs, 'T2WSeg/prostate')
    path_gt_2 = join(path_data, dirs, 'T2WSeg/cap')
    result_filename = 'result_{}.npz'.format(idx)
    path_result = join(path_results, result_filename)

    # Import the data from the GT
    volume_gt = OpenSerieUsingGTDCM(path_gt_2, path_gt, reverse_gt=True, reverse_data=True)
    label_gt = VolumeToLabelUsingGT(volume_gt, path_gt)

    # Import the predicted data
    volume_res = ResultToVolume(path_result, path_gt)
    pred_label = ResultToLabel(path_result)

    # Compute the confusion matrix using the different label
    conf_mat = BuildConfusionFromVolume(label_gt, pred_label)
    stats = LabelsToSensitivitySpecificity(label_gt, pred_label)
    print 'Sensitivity - Specificity: {:.3f} - {:.3f}'.format(stats[0], stats[1])

# Open all the ROC curve
rocs = OpenROCPatients(path_results)
PlotROCPatients(rocs)
