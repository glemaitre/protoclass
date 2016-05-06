""" Test the class T2W modality. """

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

from nose.tools import assert_true

from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality


def test_path_list_no_dir():
    """ Test either if an error is raised when the directory does not
    exist. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    t2w_mod = T2WModality()

    # We can pass a list of unknown path
    path_data_list = [path_data, path_data]
    assert_raises(ValueError, t2w_mod.read_data_from_path, path_data_list)


def test_path_list_wrong_type():
    """ Test either an error is raised if the type in the list is
    not string. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    t2w_mod = T2WModality()

    # We can a list we incorrect type
    path_data_list = [1, path_data, path_data]
    assert_raises(ValueError, t2w_mod.read_data_from_path, path_data_list)


def test_path_no_dir():
    """ Test either if an error is raised when no path is given at
    any point. """
    # Create an object to handle the data
    t2w_mod = T2WModality()

    # Check that an error is risen
    assert_raises(ValueError, t2w_mod.read_data_from_path)


def test_path_wrong_type():
    """ Test either if an error is raised when the type of the path is not a
    string. """
    # Create a dummy type
    path_data = 1
    # Create an object to handle the data
    t2w_mod = T2WModality()
    # Check that an error is risen
    assert_raises(ValueError, t2w_mod.read_data_from_path, path_data)


def test_path_wrong_dir():
    """ Test if an error is raised when the directory does not exist. """

    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    t2w_mod = T2WModality()
    # Check that an error is risen
    assert_raises(ValueError, t2w_mod.read_data_from_path, path_data)


def test_read_t2w_dicom_more_2_serie():
    """ Test if an error is raised if there is more than 2 series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    # Check the assert
    assert_raises(ValueError, t2w_mod.read_data_from_path, path_data)


def test_read_t2w_data():
    """ Test if we can read t2w series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    # Check that the data have been read
    assert_true(not t2w_mod.is_read())

    t2w_mod.read_data_from_path(path_data)

    # Check that the data have been read
    assert_true(t2w_mod.is_read())

    # Check the type of the data
    assert_equal(t2w_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(t2w_mod.data_.shape, (360, 448, 5))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(t2w_mod.min_, 0.)
    assert_equal(t2w_mod.max_, 959.)

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
    t2w_mod = T2WModality()

    t2w_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    t2w_mod.data_[20:40, :, :] = 1050.
    t2w_mod.update_histogram('auto')

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
    t2w_mod = T2WModality()

    assert_raises(ValueError, t2w_mod.update_histogram)


def test_t2w_path_data_warning():
    """ Test either if a warning is raised if the path will be overriden. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)

    # Check that a warning is raised when reading the data with a data path
    # after specifying one previously.
    assert_warns(UserWarning, t2w_mod.read_data_from_path, path_data)


def test_t2w_path_data_constructor():
    """ Test if the dce function is working when passing the path data to the
    constructor. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)

    t2w_mod.read_data_from_path()

    # Check that the data have been read
    assert_true(t2w_mod.is_read())

    # Check the type of the data
    assert_equal(t2w_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(t2w_mod.data_.shape, (360, 448, 5))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(t2w_mod.min_, 0.)
    assert_equal(t2w_mod.max_, 959.)

    # Check that the data correspond to the one save inside the the test
    data = np.load(os.path.join(currdir, 'data', 'bin_t2w_data.npy'))
    assert_array_equal(t2w_mod.bin_, data)
    data = np.load(os.path.join(currdir, 'data', 'pdf_t2w_data.npy'))
    assert_array_equal(t2w_mod.pdf_, data)
    data = np.load(os.path.join(currdir, 'data', 'data_t2w_data.npy'))
    assert_array_equal(t2w_mod.data_, data)


def test_t2w_histogram_wrong_nb_bins_param():
    """Test if either an error is raised when the parameters for the `nb_bins`
    is unknown."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)

    t2w_mod.read_data_from_path()

    # Update the histogram with the unknown parameters
    assert_raises(ValueError, t2w_mod.update_histogram, nb_bins='rnd')


def test_t2w_update_histo_force_bins():
    """Test to check the routine to update the histogram with a given number
    of bins."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality(path_data)

    t2w_mod.read_data_from_path()

    # Recompute the number of histogram with a given number of bins
    t2w_mod.update_histogram(nb_bins=10)

    # Check that the data correspond to the one save inside the the test
    data = np.load(os.path.join(currdir, 'data',
                                'bin_t2w_data_forced_bin.npy'))
    assert_array_equal(t2w_mod.bin_, data)
    data = np.load(os.path.join(currdir, 'data',
                                'pdf_t2w_data_forced_bin.npy'))
    assert_array_equal(t2w_mod.pdf_, data)


def test_get_pdf_no_data():
    """Test either if an error is raised when a pdf is asked with not
    data opened."""

    # Create the object
    t2w_mod = T2WModality()
    assert_raises(ValueError, t2w_mod.get_pdf)


def test_get_pdf_wrong_string():
    """Test either if an error is raised when the string is unknown."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    t2w_mod.read_data_from_path(path_data)

    # Compute the hisogram with a wrong argument as string
    assert_raises(t2w_mod.get_pdf, nb_bins='rnd')


def test_get_pdf_wrong_type():
    """Test either if an error is raised when the type is unknown."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    t2w_mod.read_data_from_path(path_data)

    # Compute the hisogram with a wrong argument as string
    assert_raises(t2w_mod.get_pdf, nb_bins=[100, 100])


def test_get_pdf_auto():
    """Test the routine with the automatic number of bins"""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    t2w_mod.read_data_from_path(path_data)

    # Compute the hisogram with a wrong argument as string
    pdf_data, bin_data = t2w_mod.get_pdf()

    # Check that the data correspond to the one save inside the the test
    data = np.load(os.path.join(currdir, 'data',
                                'bin_t2w_get_pdf_auto.npy'))
    assert_array_equal(bin_data, data)
    data = np.load(os.path.join(currdir, 'data',
                                'pdf_t2w_get_pdf_auto.npy'))
    assert_array_equal(pdf_data, data)


def test_get_pdf_none():
    """Test the routine using the inital number of bins."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    t2w_mod.read_data_from_path(path_data)

    # Compute the hisogram with a wrong argument as string
    pdf_data, bin_data = t2w_mod.get_pdf(nb_bins=None)

    # Check that the data correspond to the one save inside the the test
    data = np.load(os.path.join(currdir, 'data',
                                'bin_t2w_get_pdf_none.npy'))
    assert_array_equal(bin_data, data)
    data = np.load(os.path.join(currdir, 'data',
                                'pdf_t2w_get_pdf_none.npy'))
    assert_array_equal(pdf_data, data)


def test_get_pdf_int():
    """Test the routine using a given number of bins."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    t2w_mod.read_data_from_path(path_data)

    # Compute the hisogram with a wrong argument as string
    pdf_data, bin_data = t2w_mod.get_pdf(nb_bins=100)

    # Check that the data correspond to the one save inside the the test
    data = np.load(os.path.join(currdir, 'data',
                                'bin_t2w_get_pdf_int.npy'))
    assert_array_equal(bin_data, data)
    data = np.load(os.path.join(currdir, 'data',
                                'pdf_t2w_get_pdf_int.npy'))
    assert_array_equal(pdf_data, data)


def test_get_pdf_roi():
    """Test the routine to get pdf and bins with a given roi."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    t2w_mod.read_data_from_path(path_data)

    path_data = os.path.join(currdir, 'data', 'gt_folders')
    path_data_list = [os.path.join(path_data, 'prostate'),
                      os.path.join(path_data, 'cg'),
                      os.path.join(path_data, 'pz'),
                      os.path.join(path_data, 'cap')]
    # Give the list for the ground_truth
    label = ['prostate', 'cg', 'pz', 'cap']
    # Create an object to handle the data
    gt_mod = GTModality()

    # Read the data
    gt_mod.read_data_from_path(label, path_data=path_data_list)

    # Extract the prostate indexes
    label_extr = 'prostate'
    data_prostate = gt_mod.extract_gt_data(label_extr, 'index')

    # Compute the hisogram with a wrong argument as string
    pdf_data, bin_data = t2w_mod.get_pdf(roi_data=data_prostate)

    # Check that the data correspond to the one save inside the the test
    data = np.load(os.path.join(currdir, 'data',
                                'bin_t2w_get_pdf_roi.npy'))
    assert_array_equal(bin_data, data)
    data = np.load(os.path.join(currdir, 'data',
                                'pdf_t2w_get_pdf_roi.npy'))
    assert_array_equal(pdf_data, data)
