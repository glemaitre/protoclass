#title           :test_flattening.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/06/06
#version         :0.1
#notes           :
#python_version  :2.7.6 
#==============================================================================

from numpy import testing

from protoclass.preprocessing.flattening import Flatten3D
from protoclass.preprocessing.flattening import Flatten2DMphMath
from protoclass.tool.dicom_manip import OpenVolumeNumpy

def test_3d_flattening():
    # Define the filename of the test volume
    filename = '/work/le2i/gu5306le/retinopathy/OCT/SERI/pre_processed_data/nlm_data_npz/P741009OS_nlm.npz'

    # Extract the volume of interest 
    name_var_extract = 'vol_denoised'
    vol = OpenVolumeNumpy(filename, name_var_extract=name_var_extract)
    
    # Flatten the volume
    out = Flatten3D(vol, thres_type='otsu', num_cores=8, crop=True)

    # Assert if the size are not identical
    testing.assert_equal(vol.shape, out.shape)
    
    return out

# The function that will be tested
if __name__ == "__main__":
    vol_out = test_3d_flattening()
