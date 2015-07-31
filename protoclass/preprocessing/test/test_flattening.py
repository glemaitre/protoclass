#title           :test_flattening.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/06/06
#version         :0.1
#notes           :
#python_version  :2.7.6 
#==============================================================================

from ..flattening import Flatten3D
from ..flattening import Flatten2DMphMath
from protoclass.tool.dicom_manip import OpenVolumeNumpy

def test_3d_flattening():
    # Define the filename of the test volume
    filename = '/work/le2i/gu5306le/OCT/nlm_data_npz/P741009OS_nlm.npz'

    # Extract the volume of interest 
    name_var_extract = 'vol_denoised'
    vol = OpenVolumeNumpy(filename, name_var_extract=name_var_extract)

    # Flatten the volume
    out = Flatten3D(vol)

# The function that will be tested
if __name__ == "__main__":
    test_3d_flattening()
