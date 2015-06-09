#title           :texture.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/06/07
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

import matplotlib.pyplot as plt

def test_2d_lbp():
    
    from protoclass.tool.dicom_manip import OpenRawImageOCT

    # Read the first volume
    filename = '/DATA/OCT/data/DR with DME cases/741009_OS cystoid spaces with HE causing central retinal thickening/P741009 20110228/P741009_Macular Cube 512x128_2-28-2011_14-42-18_OS_sn44088_cube_raw.img'
    img = OpenRawImageOCT(filename, (512, 128, 1024))

    # Extract lbp map for the first slice
    from protoclass.extraction.texture_analysis import LBPMapExtraction

    radius = 16
    n_points = 8 * radius
    lbp_im = LBPMapExtraction(img[:, 0, :], radius=radius, n_points=n_points)

    plt.figure()
    plt.imshow(lbp_im)
    plt.show()

def test_25d_lbp():
    
    from protoclass.tool.dicom_manip import OpenRawImageOCT

    # Read the first volume
    filename = '/DATA/OCT/data/DR with DME cases/741009_OS cystoid spaces with HE causing central retinal thickening/P741009 20110228/P741009_Macular Cube 512x128_2-28-2011_14-42-18_OS_sn44088_cube_raw.img'
    img = OpenRawImageOCT(filename, (512, 128, 1024))

    # Extract lbp map for the first slice
    from protoclass.extraction.texture_analysis import LBPMapExtraction

    radius = 16
    n_points = 8 * radius
    extr_3d = '2.5D'
    extr_axis = 'y'
    lbp_vol = LBPMapExtraction(img, radius=radius, n_points=n_points, 
                               extr_3d=extr_3d, extr_axis=extr_axis)
    
    plt.figure()
    plt.imshow(lbp_vol[:, 0, :])
    plt.show()

def test_2d_lbp_hist():

    from protoclass.tool.dicom_manip import OpenVolumeNumpy

    # Read the first volume
    #filename = '/work/le2i/gu5306le/OCT/lbp_r_1_data_npz/P741009OS_nlm_lbp_1.npz'
    filename = '/DATA/OCT/P741009OS_nlm_lbp_1.npz'

    # Read the data
    lbp_vol = OpenVolumeNumpy(filename, name_var_extract='vol_lbp')

    # Compute the histogram for one frame
    from protoclass.extraction.texture_analysis import LBPpdfExtraction
    hist = LBPpdfExtraction(lbp_vol[:, 0, :], strategy_win='sliding_win')

    print hist.shape
    
    # plt.figure()
    # plt.plot(hist)
    # plt.show()

def test_25d_lbp_hist():

    from protoclass.tool.dicom_manip import OpenVolumeNumpy

    # Read the first volume
    #filename = '/work/le2i/gu5306le/OCT/lbp_r_1_data_npz/P741009OS_nlm_lbp_1.npz'
    filename = '/DATA/OCT/P741009OS_nlm_lbp_1.npz'

    # Read the data
    lbp_vol = OpenVolumeNumpy(filename, name_var_extract='vol_lbp')

    # Compute the histogram for one frame
    from protoclass.extraction.texture_analysis import LBPpdfExtraction

    extr_3d = '2.5D'
    extr_axis = 'y'
    hist = LBPpdfExtraction(lbp_vol, extr_3d=extr_3d, extr_axis=extr_axis,
                            strategy_win='sliding_win')

    print hist.shape
    
    # plt.figure()
    # plt.plot(hist[:, 0, :])
    # plt.show()
    
if __name__ == "__main__":
    #test_2d_lbp()
    #test_25d_lbp()
    #test_2d_lbp_hist()
    test_25d_lbp_hist()
