""" Test the class T2W modality. """

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from protoclass.data_management import T2WModality


def test_read_t2w_dicom_no_dir():
    """ Test if an error is raised when the directory does not exist. """

    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)
    # Check that an error is risen
    assert_raises(ValueError, t2w_mod.read_data_from_path)


def test_read_t2w_dicom_more_2_serie():
    """ Test if an error is raised if there is more than 2 series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)
    # Check the assert
    assert_raises(ValueError, t2w_mod.read_data_from_path)


def test_read_dce_data():
    """ Test if we can read t2w series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)

    t2w_mod.read_data_from_path()

    # Check the type of the data
    assert_equal(t2w_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(t2w_mod.data_.shape, (360, 448, 64))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(t2w_mod.min_, 0.)
    assert_equal(t2w_mod.max_, 1014.)

    # Check that the data correspond to the one save inside the the test
    data = np.load(os.path.join(currdir, 'data', 'bin_t2w_data.npy'))
    assert_array_equal(t2w_mod.bin_, data)
    data = np.load(os.path.join(currdir, 'data', 'pdf_t2w_data.npy'))
    assert_array_equal(t2w_mod.pdf_, data)
    data = np.load(os.path.join(currdir, 'data', 'data_t2w_data.npy'))
    assert_array_equal(t2w_mod.data_, data)

def test_update_histogram():
    """ Test that the function properly update the value of the histogram. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)

    t2w_mod.read_data_from_path()

    # Change something in the data to check that the computation
    # is working
    t2w_mod.data_[20:40, :, :] = 1050.
    t2w_mod._update_histogram()

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(t2w_mod.min_, 0.)
    assert_equal(t2w_mod.max_, 1050.)

    # Check the pdf and bins
    data = np.load(os.path.join(currdir, 'data', 'bin_t2w_data_update.npy'))
    assert_array_equal(t2w_mod.bin_, data)
    data = np.load(os.path.join(currdir, 'data', 'pdf_t2w_data_update.npy'))
    assert_array_equal(t2w_mod.pdf_, data)


def test_update_histogram_wt_data():
    """ Test whether an error is raised if the histogram function is called
    before to read the data. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)

    assert_raises(ValueError, t2w_mod._update_histogram)