""" Test the class DCE modality. """

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from protoclass.data_management import DCEModality


def test_read_dce_dicom_no_dir():
    """ Test if an error is raised when the directory does not exist. """

    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)
    # Check that an error is risen
    assert_raises(ValueError, dce_mod.read_data_from_path)


def test_read_dce_dicom_less_2_serie():
    """ Test if an error is raised if there is not at least 2 series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)
    # Check the assert
    assert_raises(ValueError, dce_mod.read_data_from_path)


def test_read_dce_data():
    """ Test if we can read 2 dce series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    dce_mod.read_data_from_path()

    # Check the type of the data
    assert_equal(dce_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dce_mod.data_.shape, (2, 368, 448, 64))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dce_mod.min_series_, 0.)
    assert_equal(dce_mod.max_series_, 676.)

    # Check that bin is what we expect
    data = np.load(os.path.join(currdir, 'data', 'bin_dce_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dce_mod.bin_series_, data):
        assert_array_equal(exp, gt)

    # Check that pdf is what we expect
    data = np.load(os.path.join(currdir, 'data', 'pdf_dce_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dce_mod.pdf_series_, data):
        assert_array_equal(exp, gt)


def test_update_histogram():
    """ Test that the function properly update the value of the histogram. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    dce_mod.read_data_from_path()

    # Change something in the data to check that the computation
    # is working
    dce_mod.data_[0, 20:40, :, :] = 1000.
    dce_mod._update_histogram()

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dce_mod.min_series_, 0.)
    assert_equal(dce_mod.max_series_, 1000.)

    # Check that bin is what we expect
    data = np.load(os.path.join(currdir, 'data', 'bin_dce_data_update.npy'))
    # Check that each array are the same
    for exp, gt in zip(dce_mod.bin_series_, data):
        assert_array_equal(exp, gt)

    # Check that pdf is what we expect
    data = np.load(os.path.join(currdir, 'data', 'pdf_dce_data_update.npy'))
    # Check that each array are the same
    for exp, gt in zip(dce_mod.pdf_series_, data):
        assert_array_equal(exp, gt)


def test_update_histogram_wt_data():
    """ Test whether an error is raised if the histogram function is called
    before to read the data. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    assert_raises(ValueError, dce_mod._update_histogram)


def test_build_heatmap_wt_data():
    """ Test whether an error is raised if heatmap build function is called
    before to read the data. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    assert_raises(ValueError, dce_mod.build_heatmap)


def test_build_heatmap():
    """ Test if the heatmap is built properly. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    # Read the data
    dce_mod.read_data_from_path()

    # Build the heatmap
    heatmap = dce_mod.build_heatmap()

    # Check that heatmap is what we expect
    data = np.load(os.path.join(currdir, 'data', 'heatmap.npy'))
    assert_array_equal(heatmap, data)
