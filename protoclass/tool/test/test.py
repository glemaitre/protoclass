#title           :test.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/06/06
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

import matplotlib.pyplot as plt

def test_raw_reading():
    
    from protoclass.tool.dicom_manip import OpenRawImageOCT

    filename = '/DATA/OCT/data/DR with DME cases/741009_OS cystoid spaces with HE causing central retinal thickening/P741009 20110228/P741009_Macular Cube 512x128_2-28-2011_14-42-18_OS_sn44088_cube_raw.img'
    img = OpenRawImageOCT(filename, (1024, 128, 512))

    plt.imshow(img[:, 60, :].T)
    plt.show()

if __name__ == "__main__":
    test_raw_reading()
