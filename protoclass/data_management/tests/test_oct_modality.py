""" Test the class OCT modality. """

import numpy as np
import os

from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns
from numpy.testing import assert_almost_equal

from nose.tools import assert_true

from protoclass.data_management import OCTModality

PRECISION_DECIMAL = 2


def test_path_list_no_dir():
    """ Test either if an error is raised when the directory does not
    exist. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    oct_mod = OCTModality()

    # We can pass a list of unknown path
    path_data_list = [path_data, path_data]
    sz_data = (512, 128, 1024)
    assert_raises(ValueError, oct_mod.read_data_from_path, sz_data,
                  path_data_list)


def test_path_list_wrong_type():
    """ Test either an error is raised if the type in the list is
    not string. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    oct_mod = OCTModality()

    # We can a list we incorrect type
    path_data_list = [1, path_data, path_data]
    sz_data = (512, 128, 1024)
    assert_raises(ValueError, oct_mod.read_data_from_path, sz_data,
                  path_data_list)


def test_path_no_dir():
    """ Test either if an error is raised when no path is given at
    any point. """
    # Create an object to handle the data
    oct_mod = OCTModality()

    # Check that an error is risen
    sz_data = (512, 128, 1024)
    assert_raises(ValueError, oct_mod.read_data_from_path, sz_data)


def test_path_wrong_type():
    """ Test either if an error is raised when the type of the path is not a
    string. """
    # Create a dummy type
    path_data = 1
    # Create an object to handle the data
    oct_mod = OCTModality()
    # Check that an error is risen
    sz_data = (512, 128, 1024)
    assert_raises(ValueError, oct_mod.read_data_from_path, sz_data, path_data)


def test_path_wrong_dir():
    """ Test if an error is raised when the directory does not exist. """

    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    oct_mod = OCTModality()
    # Check that an error is risen
    sz_data = (512, 128, 1024)
    assert_raises(ValueError, oct_mod.read_data_from_path, sz_data, path_data)


def test_updade_no_data():
    """Test either if an error is raised when no data have been loaded."""

    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    oct_mod = OCTModality()
    # Check that an error is risen
    sz_data = (512, 128, 1024)

    assert_raises(ValueError, oct_mod.update_histogram)


def test_read_from_path():
    """Test the routine to read the OCT data."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'oct', 'oct_image.img')
    # Create an object to handle the data
    oct_mod = OCTModality()

    # Check that the data have been read
    assert_true(not oct_mod.is_read())

    # Read the data
    sz_data = (4, 1024, 512)
    oct_mod.read_data_from_path(sz_data, path_data)


def test_update_histogram():
    """ Test that the function properly update the value of the histogram. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'oct', 'oct_image.img')
    # Create an object to handle the data
    oct_mod = OCTModality()

    # Read the data
    sz_data = (4, 1024, 512)
    oct_mod.read_data_from_path(sz_data, path_data)

    # Change something in the data to check that the computation
    # is working
    oct_mod.data_[:, :, 20:40] = 1.
    oct_mod.update_histogram()

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(oct_mod.min_, 0.)
    assert_equal(oct_mod.max_, 1.)

    # Check the pdf and bins
    data = np.load(os.path.join(currdir, 'data', 'bin_oct_data_update.npy'))
    assert_array_equal(oct_mod.bin_, data)
    data = np.load(os.path.join(currdir, 'data', 'pdf_oct_data_update.npy'))
    assert_array_equal(oct_mod.pdf_, data)


def test_update_histogram_wt_data():
    """ Test whether an error is raised if the histogram function is called
    before to read the data. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'oct', 'oct_image.img')
    # Create an object to handle the data
    oct_mod = OCTModality()

    assert_raises(ValueError, oct_mod.update_histogram)


def test_update_histogram_wrong_string_nb_bins():
    """Test either if an error is raised when an unknown string is passed."""

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'oct', 'oct_image.img')
    # Create an object to handle the data
    oct_mod = OCTModality()

    # Read the data
    sz_data = (4, 1024, 512)
    oct_mod.read_data_from_path(sz_data, path_data)

    # Change something in the data to check that the computation
    # is working
    oct_mod.data_[:, :, 20:40] = 1.
    assert_raises(ValueError, oct_mod.update_histogram, 'rnd')


def test_oct_path_data_warning():
    """ Test either if a warning is raised if the path will be overriden. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'oct', 'oct_image.img')
    # Create an object to handle the data
    oct_mod = OCTModality(path_data)

    # Check that a warning is raised when reading the data with a data path
    # after specifying one previously.
    sz_data = (4, 1024, 512)
    assert_warns(UserWarning, oct_mod.read_data_from_path, sz_data, path_data)


def test_update_histogram_wrong_type_nb_bins():
    """Test either if an error is raised when an unknown type is passed."""

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'oct', 'oct_image.img')
    # Create an object to handle the data
    oct_mod = OCTModality()

    # Read the data
    sz_data = (4, 1024, 512)
    oct_mod.read_data_from_path(sz_data, path_data)

    # Change something in the data to check that the computation
    # is working
    oct_mod.data_[:, :, 20:40] = 1.
    assert_raises(ValueError, oct_mod.update_histogram, 2.5)


def test_update_histogram_fix_nb_bins():
    """Test the histogram update with a fix number of bin."""

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'oct', 'oct_image.img')
    # Create an object to handle the data
    oct_mod = OCTModality()

    # Read the data
    sz_data = (4, 1024, 512)
    oct_mod.read_data_from_path(sz_data, path_data)

    # Change something in the data to check that the computation
    # is working
    oct_mod.data_[:, :, 20:40] = 1.
    oct_mod.update_histogram(nb_bins=100)

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(oct_mod.min_, 0.)
    assert_equal(oct_mod.max_, 1.)

    # Check the pdf and bins
    data = np.load(os.path.join(currdir, 'data',
                                'bin_oct_data_update_100_bin.npy'))
    assert_array_equal(oct_mod.bin_, data)
    data = np.load(os.path.join(currdir, 'data',
                                'pdf_oct_data_update_100_bin.npy'))
    assert_array_equal(oct_mod.pdf_, data)
