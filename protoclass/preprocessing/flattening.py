#title           :flattening.py
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
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing

def Flatten3D(volume, flattening_method='morph-mat', crop = False, **kwargs):
    # GOAL: flatten a complete 3D volume

    # Check that the type of image is float and in the range (0., 1.)
    ### Check the type
    if not ( (volume.dtype == 'float32') or 
             (volume.dtype != 'float64')    ):
        raise ValueError('protoclass.preprocessing.flattening: The volume data should be float data.')

    ### Check the range
    if ( (np.min(volume) < 0.0) and
         (np.max(volume) > 1.0)     ):
        raise ValueError('protoclass.preprocessing.flattening: The volume data should range between 0.0 and 1.0.')

    if (flattening_method == 'morph-mat'):

        # The dimension which will vary will be Y. 
        # We need to swap Y in first position
        volume_swaped = np.swapaxes(volume, 0, 1)
        
        # Compute the Haralick statistic in parallel
        num_cores = kwargs.pop('num_cores', multiprocessing.cpu_count())
        print num_cores
        data = Parallel(n_jobs=num_cores)(delayed(Flatten2DMphMath)(im.T, **kwargs) for im in volume_swaped)

        # Unpack the data
        volume_flatten = []
        baseline_volume = []
        for sl in data:
            baseline_volume.append(sl[0])
            volume_flatten.append(sl[1])

        # We need now to align the volume
        center_line = np.mean(baseline_volume)
        volume_aligned = Parallel(n_jobs=num_cores)(delayed(Align2DIm)(im.T, baseline_volume[idx_im], center_line)
                                                    for idx_im, im in enumerate(volume_flatten))

        # Do we want to crop an image
        if (crop == True):
            crop_size = kwargs.pop('crop_size', 400)
            volume_cropped = Parallel(n_jobs=num_cores)(delayed(Cropping2DIm)(im, center_line, crop_size)
                                                        for im in volume_aligned)

            print np.array(volume_cropped).shape

            # We need to swap back
            return np.swapaxes(volume_cropped, 0, 1)

        # We need to swap back
        return np.swapaxes(volume_aligned, 0, 1)
        #return np.swapaxes(np.swapaxes(volume_aligned, 1, 2), 1, 0)
    
    else:
        raise ValueError('protoclass.preprocessing.flattening: Unrecognise type of flattening.')

def Flatten2DMphMath(image, **kwargs):
    # GOAL: flatten the 2d image using some morpho mat and warping
    # Parameters - check kawrgs.pop

    ### Original pipeline proposed by:
    ### Liu et al. "Automated macular pathology diagnosis in retinal OCT images using multi-scale spatial pyramid and
    ### local binary patterns in texture and shape encoding", Medical Image Anlysis 15 (2011)
    ### 1. Thresholding
    ### 2. Median filtering
    ### 3. Closing + opening
    ### 4. Fitting via 2nd order polynomial
    ### 5. Warping the entire image to get a straight line

    # ----- 1. THRESHOLDING ----- #
    # PARAMETERS
    ### Get the type of threshold needed to segment the image
    thres_type = kwargs.pop('thres_type', 'static')
    if (thres_type == 'static'):
        # Get the threshold if specified
        thres_val = kwargs.pop('thres_val', .2) # By default the static threshold will be .2
    elif (thres_type == 'otsu'):
        # Import the threshold otsu from skimage
        from skimage.filters import threshold_otsu
        thres_val = threshold_otsu(image)
    else:
        raise ValueError('protoclass.preprocessing.flattening.Flatten2DMphMath: Unrecognise type of thresholding.')

    # PROCESSING
    ### Apply the thresholding
    bin_img = (image > thres_val)

    # ----- 2. MEDIAN FILTERING ----- #
    # PARAMETERS
    ### Get the morphological operator for the median filtering
    median_kernel_type = kwargs.pop('median_kernel_type', 'square') # Default type of kernel square
    median_kernel_size = kwargs.pop('median_kernel_size', 5) # Default kernel size 5

    if (median_kernel_type == 'square'):
        from skimage.morphology import square
        median_kernel = square(median_kernel_size)
    elif (median_kernel_type == 'disk'):
        from skimage.morphology import disk
        median_kernel = disk(median_kernel_size)
    else:
        raise ValueError('protoclass.preprocessing.flattening.Flatten2DMphMath: Median kernel type unrecognized')

    # CONVERSION INTO UBYTE
    from skimage import img_as_ubyte
    bin_img_uint8 = img_as_ubyte(bin_img)

    # PROCESSING
    from skimage.filters import median
    bin_filt_img_uint8 = median(bin_img_uint8, median_kernel)

    # CONVERSION INTO BOOL
    from skimage import img_as_bool
    bin_filt_img = img_as_bool(bin_filt_img_uint8)
    
    # ----- 3. MORPHO MATH ----- #
    # PARAMETERS
    ### Get the morphological operator for the opening operation
    opening_kernel_type = kwargs.pop('opening_kernel_type', 'disk') # Default type of kernel disk
    opening_kernel_size = kwargs.pop('opening_kernel_size', 5) # Default kernel size 5
    if (opening_kernel_type == 'square'):
        from skimage.morphology import square
        opening_kernel = square(opening_kernel_size)
    elif (opening_kernel_type == 'disk'):
        from skimage.morphology import disk
        opening_kernel = disk(opening_kernel_size)
    else:
        raise ValueError('protoclass.preprocessing.flattening.Flatten2DMphMath: Opening kernel type unrecognized')
        
    ### Get the morphological operator for the closing operation
    closing_kernel_type = kwargs.pop('closing_kernel_type', 'disk') # Default type of kernel disk
    closing_kernel_size = kwargs.pop('closing_kernel_size', 35) # Default kernel size 35
    if (closing_kernel_type == 'square'):
        from skimage.morphology import square
        closing_kernel = square(closing_kernel_size)
    elif (closing_kernel_type == 'disk'):
        from skimage.morphology import disk
        closing_kernel = disk(closing_kernel_size)
    else:
        raise ValueError('protoclass.preprocessing.flattening.Flatten2DMphMath: Closing kernel type unrecognized')

    # PROCESSING
    ### Apply a closing operation
    from skimage.morphology import binary_closing
    bin_closing = binary_closing(bin_filt_img, closing_kernel)

    ### Apply an opening operation
    from skimage.morphology import binary_opening
    bin_opening = binary_opening(bin_closing, opening_kernel)

    # ----- 4. POLYNOMIAL FITTING ----- #
    # PROCESSING
    ### Get the x,y position of True value in the binary image
    y_img, x_img = np.nonzero(bin_opening)

    ### Fit a second order polynomial
    x_interp = np.arange(bin_opening.shape[1])
    p_fitted = np.poly1d(np.polyfit(x_img, y_img, 2))
    y_interp = p_fitted(x_interp)
    
    # ----- 5. WARPING OF THE IMAGE ----- #
    ### Allocate the memory for the image
    warped_img = np.zeros(image.shape)
    
    ### Get each column of the original image
    ### Get the minimum y of the interpolated value in order to align the value to this baseline
    baseline_y = np.min(y_interp)
    for col_ind, col_img in enumerate(image.T):
        
        ### Compute the distance to apply the rolling
        dist_roll = int(np.round(baseline_y - y_interp[col_ind]))

        ### Assign the new column to the warped image
        warped_img[:, col_ind] = np.roll(col_img, dist_roll)

    # Finally return the unflatten image
    return (np.round(baseline_y), warped_img)

def Align2DIm(image, baseline_im, center_line):
    # GOAL: All the different images need to be aligned to the same reference

    ### Compute the distance to apply the rolling
    dist_roll = int(np.round(center_line - baseline_im))

    ### Roll each column of the image
    return np.roll(image, dist_roll, axis=1)
    
def Cropping2DIm(image, center_line, crop_size, **kwargs):
    # GOAL: Crop each 2D image

    ### Compute the top and bottom limits
    top_limit = center_line - int(crop_size / 2.0)
    bottom_limit = center_line + int(crop_size / 2.0)

    return image[:, top_limit:bottom_limit]

