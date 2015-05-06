#title           :test_extraction_haralick_rnormalised.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/20
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Matplotlib library
import matplotlib.pyplot as plt
# Numpy libreary
import numpy as np
# OS library
import os
from os.path import join
# SYS library
import sys

# Our module to read DCM volume from a serie of DCM
from protoclass.tool.dicom_manip import OpenVolumeNumpy
# Our module to extract edges feature maps
from protoclass.extraction.edge_analysis import EdgeMapExtraction

# Give the path to a patient
path_t2w = 'rician_norm/volume_rnorm.npy'
path_sobel1 = 'sobel1'
path_sobel2 = 'sobel2'
path_prewitt1 = 'prewitt1'
path_prewitt2 = 'prewitt2'
path_gabor = 'gaborbank'
path_phase = 'phasecong'

# Build the path of the current patient
path_patient = sys.argv[1]
path_dcm = join(path_patient, path_t2w)
print 'Reading data from the directory {}'.format(path_dcm)

# Read a volume
volume = OpenVolumeNumpy(path_dcm)


#################################################################################
### SOBEL 1ST ORDER

# Create the ouput list
patient_maps = []

# Go through each slice of the volume
for sl in range(volume.shape[2]):
    ### Compute the Haralick maps
    patient_maps(EdgeMapExtraction(volume[:, :, sl]), edge_detector='Sobel1stDev')
    
# Convert maps to an Numpy array
patient_maps = np.array(patient_maps)

# Roll the first axis to obtain [y,x,z]
patient_maps = np.rollaxis(patient_maps, 0, 3)

# Create a directory if not existing
path_saving = join(path_patient, path_sobel1)
if not os.path.exists(path_saving):
    os.makedirs(path_saving)

# Save the volume
filename = join(path_saving, 'volume_sobel1.npy')
np.save(filename, patient_maps)

#################################################################################
### SOBEL 2ND ORDER

# Create the ouput list
patient_maps = []

# Go through each slice of the volume
for sl in range(volume.shape[2]):
    ### Compute the Haralick maps
    patient_maps(EdgeMapExtraction(volume[:, :, sl]), edge_detector='Sobel2ndDev')
    
# Convert maps to an Numpy array
patient_maps = np.array(patient_maps)

# Roll the first axis to obtain [y,x,z]
patient_maps = np.rollaxis(patient_maps, 0, 3)

# Create a directory if not existing
path_saving = join(path_patient, path_sobel2)
if not os.path.exists(path_saving):
    os.makedirs(path_saving)

# Save the volume
filename = join(path_saving, 'volume_sobel2.npy')
np.save(filename, patient_maps)

#################################################################################
### PREWITT 1ST ORDER

# Create the ouput list
patient_maps = []

# Go through each slice of the volume
for sl in range(volume.shape[2]):
    ### Compute the Haralick maps
    patient_maps(EdgeMapExtraction(volume[:, :, sl]), edge_detector='Prewitt1stDev')
    
# Convert maps to an Numpy array
patient_maps = np.array(patient_maps)

# Roll the first axis to obtain [y,x,z]
patient_maps = np.rollaxis(patient_maps, 0, 3)

# Create a directory if not existing
path_saving = join(path_patient, path_prewitt1)
if not os.path.exists(path_saving):
    os.makedirs(path_saving)

# Save the volume
filename = join(path_saving, 'volume_prewitt1.npy')
np.save(filename, patient_maps)

#################################################################################
### PREWITT 2ND ORDER

# Create the ouput list
patient_maps = []

# Go through each slice of the volume
for sl in range(volume.shape[2]):
    ### Compute the Haralick maps
    patient_maps(EdgeMapExtraction(volume[:, :, sl]), edge_detector='Prewit2ndtDev')
    
# Convert maps to an Numpy array
patient_maps = np.array(patient_maps)

# Roll the first axis to obtain [y,x,z]
patient_maps = np.rollaxis(patient_maps, 0, 3)

# Create a directory if not existing
path_saving = join(path_patient, path_prewitt2)
if not os.path.exists(path_saving):
    os.makedirs(path_saving)

# Save the volume
filename = join(path_saving, 'volume_prewitt2.npy')
np.save(filename, patient_maps)

#################################################################################
### GABOR FILTER BANK

# Create the ouput list
patient_maps = []
kernel_params_list = []
# Go through each slice of the volume
for sl in range(volume.shape[2]):
    # Build edges maps
    maps, kernel_params = EdgeMapExtraction(volume[:,:,sl], edge_detector='GaborBank')
    patient_maps.append(maps)
    kernel_params_list.append(kernel_params)
    
# Convert maps to an Numpy array
patient_maps = np.array(patient_maps)
kernel_params_list = np.array(kernel_params_list)

# Roll the first axis to obtain [y,x,z]
patient_maps = np.rollaxis(patient_maps, 0, 4)
kernel_params_list = np.rollaxis(kernel_params_list, 0, 3)

# Create a directory if not existing
path_saving = join(path_patient, path_gabor)
if not os.path.exists(path_saving):
    os.makedirs(path_saving)

# Save the volume
for f in range(kernel_params_list.shape[0]):
    filename = join(path_saving, 'volume_gabor_freq_' + str(kernel_params_list[f, 0, 0]) + '_angle_' + str(kernel_params_list[f, 1, 0]) + '.npy')
    np.save(filename, patient_maps[f, :, :, :])

#################################################################################
### PHASE CONGRUENCY

# Create the ouput list
patient_maps = []

# Go through each slice of the volume
for sl in range(volume.shape[2]):
    # Build edges maps
    patient_maps.append(EdgeMapExtraction(volume[:,:,sl], edge_detector='PhaseCong'))
    
# Convert maps to an Numpy array
patient_maps = np.array(patient_maps)

# Roll the first axis to obtain [y,x,z]
patient_maps = np.rollaxis(patient_maps, 0, 4)

# Create a directory if not existing
path_saving = join(path_patient, path_phase)
if not os.path.exists(path_saving):
    os.makedirs(path_saving)

# Save the volume
### Save the edge detection
filename = join(path_saving, 'volume_phasecong_M.npy')
np.save(filename, patient_maps[0, :, :, :])
### Save the blob detection
filename = join(path_saving, 'volume_phasecong_m.npy')
np.save(filename, patient_maps[1, :, :, :])
