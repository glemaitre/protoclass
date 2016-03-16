#title           :test.py
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

#################################################################################
## FIND EXTREMUM
def test_find_extremum():

    from protoclass.tool.dicom_manip import FindExtremumDataSet

    path_patients = '/DATA/prostate/public/Siemens'

    range_dataset_float = FindExtremumDataSet(path_patients)
    range_dataset_int = (int(range_dataset_float[0]), int(range_dataset_float[1]))

#################################################################################
## Read the volume

def test_read_volume():

    from protoclass.tool.dicom_manip import OpenOneSerieDCM
    from protoclass.tool.dicom_manip import OpenSerieUsingGTDCM

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

def test_gaussian():

    from protoclass.preprocessing.normalisation import GaussianNormalisation

    path_gaussian_norm = 'gaussian_norm'

    gaussian_norm_t2w = GaussianNormalisation()

    # Extract only the prostate data
    prostate_data = volume_emd_gt[np.nonzero(~np.isnan(volume_emd_gt))]
    gaussian_norm_t2w.Fit(prostate_data, range_dataset_int)

    # Normalise the whole data
    data_normalised = gaussian_norm_t2w.Normalise(volume)

#################################################################################
## RICIAN NORMALISATION

def test_rician():

    from protoclass.preprocessing.normalisation import RicianNormalisation

    path_rician_norm = 'rician_norm'
    
    rician_norm_t2w = RicianNormalisation()

    # Extract only the prostate data
    prostate_data = volume_emd_gt[np.nonzero(~np.isnan(volume_emd_gt))]
    rician_norm_t2w.Fit(prostate_data, range_dataset_int)

    # Normalise the whole data
    data_normalised = rician_norm_t2w.Normalise(volume)

#################################################################################
## LINEAR NORMALISATION BY PARTS

def test_linear():

    from protoclass.tool.dicom_manip import FindLandmarksDataset

    path_patients = '/DATA/prostate/public/Siemens'
    path_t2w = 'T2W'
    path_gt = 'T2WSeg/prostate'

    atlas = FindLandmarksDataset(path_patients, path_t2w, path_gt, n_landmarks=5, min_perc=0., max_perc=100.)

    print atlas

    from protoclass.preprocessing.normalisation import LinearNormalisationByParts
    
    linear_norm_t2w = LinearNormalisationByParts(atlas, min_perc=0., max_perc=100.)

    from protoclass.tool.dicom_manip import OpenOneSerieDCM
    from protoclass.tool.dicom_manip import OpenSerieUsingGTDCM

    # Give the path to a patient
    path_patient = '/DATA/prostate/public/Siemens/Patient 799'
    path_t2w = 'T2W'
    path_gt = 'T2WSeg/prostate'

    path_dcm_t2w = join(path_patient, path_t2w)
    path_dcm_gt = join(path_patient, path_gt)

    # Read a volume
    volume = OpenOneSerieDCM(path_dcm_t2w)
    volume_emd_gt = OpenSerieUsingGTDCM(path_dcm_t2w, path_dcm_gt)

    # Extract only the prostate data
    prostate_data = volume_emd_gt[np.nonzero(~np.isnan(volume_emd_gt))]
    linear_norm_t2w.Fit(prostate_data)

    print linear_norm_t2w.GetParameters()
    
    # Normalise the whole data
    data_normalised = linear_norm_t2w.Normalise(volume)

    n, bins, patches = plt.hist(data_normalised.reshape(-1), np.max(data_normalised), 
                                normed=1, facecolor='green', alpha=0.75)
    plt.show()


#################################################################################
## OCT DENOISING

def test_denoise_nlm():

    # Read one volume
    from protoclass.tool.dicom_manip import OpenRawImageOCT

    filename = '/DATA/OCT/data/DR with DME cases/741009_OS cystoid spaces with HE causing central retinal thickening/P741009 20110228/P741009_Macular Cube 512x128_2-28-2011_14-42-18_OS_sn44088_cube_raw.img'
    volume = OpenRawImageOCT(filename, (512, 128, 1024))

    # Call the denoising function using the NLM approach
    from protoclass.preprocessing.denoising import Denoising3D
    volume_denoised = Denoising3D(volume)
    
    # from protoclass.preprocessing.denoising import Denoising2D
    # im_den = Denoising2D(volume[:, 60, :].T)

    plt.figure()
    plt.imshow(volume[:, 60, :].T)
    plt.figure()
    plt.imshow(volume_denoised[:, 60, :].T)
    plt.show()

if __name__ == "__main__":
    #### NEED TO REVIEW THE NORMALISATION TEST
    
    #test_denoise_nlm()
    test_linear()
