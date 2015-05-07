#title           :test_normalisation.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/26
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Matplotlib library
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
### Module to performed parallel processing
import multiprocessing
# OS library
import os
from os.path import join
# SYS library
import sys

# Our module
from protoclass.tool.dicom_manip import OpenOneSerieDCM
from protoclass.tool.dicom_manip import OpenSerieUsingGTDCM
from protoclass.tool.dicom_manip import FindExtremumDataSet
from protoclass.preprocessing.normalisation import GaussianNormalisation
from protoclass.preprocessing.normalisation import RicianNormalisation

#################################################################################
## FIND EXTREMUM
path_patients = '/DATA/prostate/public/Siemens'

range_dataset_float = FindExtremumDataSet(path_patients)
range_dataset_int = (int(range_dataset_float[0]), int(range_dataset_float[1]))

#################################################################################
## Read the volume

# Give the path to a patient
path_patient = '/DATA/prostate/public/Siemens/Patient 799'
path_t2w = 'T2W'
path_gt = 'T2WSeg/prostate'

path_dcm_t2w = join(path_patient, path_t2w)
path_dcm_gt = join(path_patient, path_gt)

# Read a volume
volume = OpenOneSerieDCM(path_dcm_t2w)
volume_emd_gt = OpenSerieUsingGTDCM(path_dcm_t2w, path_dcm_gt)

plt.imshow(volume_emd_gt[:,:,35], cmap=cm.Greys_r)
plt.show()

#################################################################################
## GAUSSIAN NORMALISATION

path_gaussian_norm = 'gaussian_norm'

gaussian_norm_t2w = GaussianNormalisation()

# Extract only the prostate data
prostate_data = volume_emd_gt[np.nonzero(~np.isnan(volume_emd_gt))]
gaussian_norm_t2w.Fit(prostate_data, range_dataset_int)

# Normalise the whole data
data_normalised = gaussian_norm_t2w.Normalise(volume)

#################################################################################
## RICIAN NORMALISATION

path_rician_norm = 'rician_norm'

rician_norm_t2w = RicianNormalisation()

# Extract only the prostate data
prostate_data = volume_emd_gt[np.nonzero(~np.isnan(volume_emd_gt))]
rician_norm_t2w.Fit(prostate_data, range_dataset_int)

# Normalise the whole data
data_normalised = rician_norm_t2w.Normalise(volume)
