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

from protoclass.validation.validation import ResultToVolume
from protoclass.tool.dicom_manip import OpenOneSerieDCM

path_result = '../results/experiments/haralick_unormalised/lopo_results/result_0.npz'
path_gt = '/DATA/prostate/public/Siemens/Patient 513/T2WSeg/prostate'

volume_gt = OpenOneSerieDCM(path_gt, reverse=True)
volume_res = ResultToVolume(path_result, path_gt)

sl = 25

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.imshow(volume_gt[:, :, sl])
ax2 = fig.add_subplot(2, 1, 2, sharex=ax, sharey=ax)
ax2.imshow(volume_res[:, :, sl])
plt.show()
