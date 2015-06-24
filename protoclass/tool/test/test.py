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
    
    from ..dicom_manip import OpenRawImageOCT

    filename = 'data/oct/PMS15336OD.img'
    img = OpenRawImageOCT(filename, (512, 128, 1024))

    plt.imshow(img[:, 20, :].T)
    plt.show()

if __name__ == "__main__":
    test_raw_reading()
