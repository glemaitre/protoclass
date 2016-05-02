""" Test the class DWI modality. """

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

from nose.tools import assert_true

from protoclass.data_management import DWIModality


def test_path_list_no_dir():
    """ Test either if an error is raised when the directory does not
    exist. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    dwi_mod = DWIModality()

    # We can pass a list of unknown path
    path_data_list = [path_data, path_data]
    assert_raises(ValueError, dwi_mod.read_data_from_path, path_data_list)


def test_path_list_wrong_type():
    """ Test either an error is raised if the type in the list is
    not string. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    dwi_mod = DWIModality()

    # We can a list we incorrect type
    path_data_list = [1, path_data, path_data]
    assert_raises(ValueError, dwi_mod.read_data_from_path, path_data_list)


def test_path_no_dir():
    """ Test either if an error is raised when no path is given at
    any point. """
    # Create an object to handle the data
    dwi_mod = DWIModality()

    # Check that an error is risen
    assert_raises(ValueError, dwi_mod.read_data_from_path)


def test_path_wrong_type():
    """ Test either if an error is raised when the type of the path is not a
    string. """
    # Create a dummy type
    path_data = 1
    # Create an object to handle the data
    dwi_mod = DWIModality()
    # Check that an error is risen
    assert_raises(ValueError, dwi_mod.read_data_from_path, path_data)


def test_path_wrong_dir():
    """ Test if an error is raised when the directory does not exist. """

    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    dwi_mod = DWIModality()
    # Check that an error is risen
    assert_raises(ValueError, dwi_mod.read_data_from_path, path_data)


def test_read_dwi_dicom_less_2_serie():
    """ Test if an error is raised if there is not at least 2 series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    dwi_mod = DWIModality()
    # Check the assert
    assert_raises(ValueError, dwi_mod.read_data_from_path, path_data)


def test_read_dwi_data():
    """ Test if we can read 2 dwi series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality()

    # Check that the data have been read
    assert_true(not dwi_mod.is_read())

    dwi_mod.read_data_from_path(path_data)

    # Check that the data have been read
    assert_true(dwi_mod.is_read())

    # Check the type of the data
    assert_equal(dwi_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dwi_mod.data_.shape, (2, 256, 256, 5))
    # Check that the data are identical
    data = np.load(os.path.join(currdir, 'data', 'data_dwi_data.npy'))
    assert_array_equal(dwi_mod.data_, data)

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dwi_mod.min_series_, 0.)
    assert_equal(dwi_mod.max_series_, 1692.)

    # Check that bin is what we expect
    data = np.load(os.path.join(currdir, 'data', 'bin_dwi_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.bin_series_, data):
        assert_array_equal(exp, gt)

    # Check that pdf is what we expect
    data = np.load(os.path.join(currdir, 'data', 'pdf_dwi_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.pdf_series_, data):
        assert_array_equal(exp, gt)


def test_update_histogram():
    """ Test that the function properly update the value of the histogram. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality()

    dwi_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    dwi_mod.data_[0, 20:40, :, :] = 2000.
    dwi_mod.update_histogram('auto')

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dwi_mod.min_series_, 0.)
    assert_equal(dwi_mod.max_series_, 2000.)

    # Check that bin is what we expect
    data = np.load(os.path.join(currdir, 'data', 'bin_dwi_data_update.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.bin_series_, data):
        assert_array_equal(exp, gt)

    # Check that pdf is what we expect
    data = np.load(os.path.join(currdir, 'data', 'pdf_dwi_data_update.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.pdf_series_, data):
        assert_array_equal(exp, gt)


def test_update_histogram_wt_data():
    """ Test whether an error is raised if the histogram function is called
    before to read the data. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality()

    assert_raises(ValueError, dwi_mod.update_histogram)


def test_dwi_path_data_warning():
    """ Test either if a warning is raised if the path will be overriden. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality(path_data)

    # Check that a warning is raised when reading the data with a data path
    # after specifying one previously.
    assert_warns(UserWarning, dwi_mod.read_data_from_path, path_data)


def test_dwi_path_data_constructor():
    """ Test if the dwi function is working when passing the path data to the
    constructor. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality(path_data)

    # Check that the data have been read
    assert_true(not dwi_mod.is_read())

    dwi_mod.read_data_from_path()

    # Check that the data have been read
    assert_true(dwi_mod.is_read())

    # Check the type of the data
    assert_equal(dwi_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dwi_mod.data_.shape, (2, 256, 256, 5))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dwi_mod.min_series_, 0.)
    assert_equal(dwi_mod.max_series_, 1692.)

    # Check that the data are identical
    data = np.load(os.path.join(currdir, 'data', 'data_dwi_data.npy'))
    assert_array_equal(dwi_mod.data_, data)

    # Check that bin is what we expect
    data = np.load(os.path.join(currdir, 'data', 'bin_dwi_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.bin_series_, data):
        assert_array_equal(exp, gt)

    # Check that pdf is what we expect
    data = np.load(os.path.join(currdir, 'data', 'pdf_dwi_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.pdf_series_, data):
        assert_array_equal(exp, gt)


def test_dwi_path_data_list():
    """ Test if the dwi function is working when passing the path data
    as list. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_1'),
                      os.path.join(path_data, 's_2')]
    # Create an object to handle the data
    dwi_mod = DWIModality(path_data_list)

    dwi_mod.read_data_from_path()

    # Check the type of the data
    assert_equal(dwi_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dwi_mod.data_.shape, (2, 256, 256, 5))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dwi_mod.min_series_, 0.)
    assert_equal(dwi_mod.max_series_, 1692.)

    # Check that the data are identical
    data = np.load(os.path.join(currdir, 'data', 'data_dwi_data.npy'))
    assert_array_equal(dwi_mod.data_, data)

    # Check that bin is what we expect
    data = np.load(os.path.join(currdir, 'data', 'bin_dwi_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.bin_series_, data):
        assert_array_equal(exp, gt)

    # Check that pdf is what we expect
    data = np.load(os.path.join(currdir, 'data', 'pdf_dwi_data.npy'))
    # Check that each array are the same
    for exp, gt in zip(dwi_mod.pdf_series_, data):
        assert_array_equal(exp, gt)


def test_path_data_list_2_series():
    """ Test either if an error is raised when there is several series when
    data are read from several folders. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_2'),
                      os.path.join(path_data, 's_1'),
                      os.path.join(currdir, 'data', 'dwi')]
    # Create an object to handle the data
    dwi_mod = DWIModality(path_data_list)

    # Check that an error is raised due to 2 series in dwi
    assert_raises(ValueError, dwi_mod.read_data_from_path)


def test_path_data_list_1_serie():
    """ Test either if an error is raised when there is only on serie to
    be read. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_2')]
    # Create an object to handle the data
    dwi_mod = DWIModality(path_data_list)

    # Check that an error is raised due to 2 series in dwi
    assert_raises(ValueError, dwi_mod.read_data_from_path)


def test_update_histogram_wrong_nb_bins():
    """Test either if an error is raised when the wrong arguments is passed."""

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality()

    dwi_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    dwi_mod.data_[0, 20:40, :, :] = 2000.
    assert_raises(ValueError, dwi_mod.update_histogram, 'rnd')


def test_update_histogram_wrong_list_bins():
    """Test either if an error is raised when the list of number of bins
    is not consistent."""

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality()

    dwi_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    dwi_mod.data_[0, 20:40, :, :] = 2000.

    # There is two series, let's give a list with three elements
    list_bins = [100, 100, 100]
    assert_raises(ValueError, dwi_mod.update_histogram, nb_bins=list_bins)


def test_update_histogram_wrong_type_list():
    """Test either if an error is raised when the type in the list are
    not consistent."""

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality()

    dwi_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    dwi_mod.data_[0, 20:40, :, :] = 2000.

    # There is two series, let's give a list with three elements
    list_bins = [100, 100, 'a']
    assert_raises(ValueError, dwi_mod.update_histogram, nb_bins=list_bins)


def test_update_histogram_unkown_nb_bins():
    """Test either if an error is raised when the arguments `nb_bins`
    is unknown."""

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dwi')
    # Create an object to handle the data
    dwi_mod = DWIModality()

    dwi_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    dwi_mod.data_[0, 20:40, :, :] = 2000.

    # There is two series, let's give a list with three elements
    assert_raises(ValueError, dwi_mod.update_histogram, nb_bins=10)

