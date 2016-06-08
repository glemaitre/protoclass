""" Test the class DCE modality. """

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

from nose.tools import assert_true

from protoclass.data_management import DCEModality


def test_path_list_no_dir():
    """ Test either if an error is raised when the directory does not
    exist. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    dce_mod = DCEModality()

    # We can pass a list of unknown path
    path_data_list = [path_data, path_data]
    assert_raises(ValueError, dce_mod.read_data_from_path, path_data_list)


def test_path_list_wrong_type():
    """ Test either an error is raised if the type in the list is
    not string. """
    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    dce_mod = DCEModality()

    # We can a list we incorrect type
    path_data_list = [1, path_data, path_data]
    assert_raises(ValueError, dce_mod.read_data_from_path, path_data_list)


def test_path_no_dir():
    """ Test either if an error is raised when no path is given at
    any point. """
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Check that an error is risen
    assert_raises(ValueError, dce_mod.read_data_from_path)


def test_path_wrong_type():
    """ Test either if an error is raised when the type of the path is not a
    string. """
    # Create a dummy type
    path_data = 1
    # Create an object to handle the data
    dce_mod = DCEModality()
    # Check that an error is risen
    assert_raises(ValueError, dce_mod.read_data_from_path, path_data)


def test_path_wrong_dir():
    """ Test if an error is raised when the directory does not exist. """

    # Create a dummy named directory
    path_data = 'None'
    # Create an object to handle the data
    dce_mod = DCEModality()
    # Check that an error is risen
    assert_raises(ValueError, dce_mod.read_data_from_path, path_data)


def test_read_dce_dicom_less_2_serie():
    """ Test if an error is raised if there is not at least 2 series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    dce_mod = DCEModality()
    # Check the assert
    assert_raises(ValueError, dce_mod.read_data_from_path, path_data)


def test_read_dce_data():
    """ Test if we can read 2 dce series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Check that the data have not been read
    assert_true(not dce_mod.is_read())

    dce_mod.read_data_from_path(path_data)

    # Check that the data have been read
    assert_true(dce_mod.is_read())

    # Check the type of the data
    assert_equal(dce_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dce_mod.data_.shape, (2, 368, 448, 5))
    # Check that the data are identical
    data = np.load(os.path.join(currdir, 'data', 'data_dce_data.npy'))
    assert_array_equal(dce_mod.data_, data)

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dce_mod.min_series_, 0.)
    assert_equal(dce_mod.max_series_, 616.)

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

    # Check the value present in the dictionary
    assert_equal(dce_mod.metadata_['size'], (448, 368, 5))
    assert_equal(dce_mod.metadata_['origin'], (-139.197998046875,
                                               -125.99199676513672,
                                               -6.814799785614014))
    assert_equal(dce_mod.metadata_['direction'], (0.9999267096574318,
                                                  -1.3680395668288353e-08,
                                                  -0.012106829215863141,
                                                  0.0016226550415489812,
                                                  0.9909776862500729,
                                                  0.13401713452043554,
                                                  0.011997595770753412,
                                                  -0.1340269575662008,
                                                  0.9909050571781686))
    assert_equal(dce_mod.metadata_['spacing'], (0.67633926868439,
                                                0.67633926868439,
                                                1.249927043914795))
    assert_equal(dce_mod.metadata_['TR'], 2350.0)
    assert_equal(dce_mod.metadata_['TE'], 101.0)
    assert_equal(dce_mod.metadata_['flip-angle'], 140.0)
    assert_array_equal(dce_mod.time_info_, np.array([0., 0.]))


def test_update_histogram():
    """ Test that the function properly update the value of the histogram. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    dce_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    dce_mod.data_[0, 20:40, :, :] = 1000.
    dce_mod.update_histogram('auto')

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
    dce_mod = DCEModality()

    assert_raises(ValueError, dce_mod.update_histogram)


def test_build_heatmap_wt_data():
    """ Test whether an error is raised if heatmap build function is called
    before to read the data. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    assert_raises(ValueError, dce_mod.build_heatmap)


def test_build_heatmap():
    """ Test if the heatmap is built properly. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Build the heatmap
    heatmap, bins_heatmap = dce_mod.build_heatmap()

    # Check that heatmap is what we expect
    data = np.load(os.path.join(currdir, 'data', 'heatmap.npy'))
    assert_array_equal(heatmap, data)
    data = np.load(os.path.join(currdir, 'data', 'bins_heatmap.npy'))
    assert_array_equal(bins_heatmap, data)


def test_dce_path_data_warning():
    """ Test either if a warning is raised if the path will be overriden. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    # Check that a warning is raised when reading the data with a data path
    # after specifying one previously.
    assert_warns(UserWarning, dce_mod.read_data_from_path, path_data)


def test_dce_path_data_constructor():
    """ Test if the dce function is working when passing the path data to the
    constructor. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    # Check that the data have not been read
    assert_true(not dce_mod.is_read())

    dce_mod.read_data_from_path()

    # Check that the data have been read
    assert_true(dce_mod.is_read())

    # Check the type of the data
    assert_equal(dce_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dce_mod.data_.shape, (2, 368, 448, 5))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dce_mod.min_series_, 0.)
    assert_equal(dce_mod.max_series_, 616.)

    # Check that the data are identical
    data = np.load(os.path.join(currdir, 'data', 'data_dce_data.npy'))
    assert_array_equal(dce_mod.data_, data)

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


def test_dce_path_data_list():
    """ Test if the dce function is working when passing the path data
    as list. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_2'),
                      os.path.join(path_data, 's_1')]
    # Create an object to handle the data
    dce_mod = DCEModality(path_data_list)

    dce_mod.read_data_from_path()

    # Check the type of the data
    assert_equal(dce_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dce_mod.data_.shape, (2, 368, 448, 5))

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dce_mod.min_series_, 0.)
    assert_equal(dce_mod.max_series_, 616.)

    # Check that the data are identical
    data = np.load(os.path.join(currdir, 'data', 'data_dce_data.npy'))
    assert_array_equal(dce_mod.data_, data)

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


def test_path_data_list_2_series():
    """ Test either if an error is raised when there is several series when
    data are read from several folders. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_2'),
                      os.path.join(path_data, 's_1'),
                      os.path.join(currdir, 'data', 'dce')]
    # Create an object to handle the data
    dce_mod = DCEModality(path_data_list)

    # Check that an error is raised due to 2 series in dce
    assert_raises(ValueError, dce_mod.read_data_from_path)


def test_path_data_list_1_serie():
    """ Test either if an error is raised when there is only on serie to
    be read. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_2')]
    # Create an object to handle the data
    dce_mod = DCEModality(path_data_list)

    # Check that an error is raised due to 2 series in dce
    assert_raises(ValueError, dce_mod.read_data_from_path)


def test_dce_get_pdf_roi():
    """Test the function to get a pdf from ROI."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_2'),
                      os.path.join(path_data, 's_1')]
    # Create an object to handle the data
    dce_mod = DCEModality()

    dce_mod.read_data_from_path(path_data_list)

    # Create ground truth array
    pos = np.ones((368, 448), dtype=bool)
    neg = np.zeros((368, 448), dtype=bool)
    gt_index = np.rollaxis(np.array([neg, pos, pos, pos, neg]), 0, 3)

    # Compute the histgram for the required data
    pdf, bins = dce_mod.get_pdf_list(roi_data=(gt_index))

    pdf_roi = np.load(os.path.join(currdir, 'data', 'pdf_roi.npy'))
    bins_roi = np.load(os.path.join(currdir, 'data', 'bins_roi.npy'))

    for pdf_s, bins_s, pdf_gt, bins_gt in zip(pdf, bins,
                                              pdf_roi, bins_roi):
        assert_array_equal(pdf_s, pdf_gt)
        assert_array_equal(bins_s, bins_gt)


def test_dce_get_pdf_wt_roi():
    """Test the function to get a pdf wihtout ROI."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce_folders')
    # Create the list of path
    path_data_list = [os.path.join(path_data, 's_2'),
                      os.path.join(path_data, 's_1')]
    # Create an object to handle the data
    dce_mod = DCEModality()

    dce_mod.read_data_from_path(path_data_list)

    # Compute the histgram for the required data
    pdf, bins = dce_mod.get_pdf_list()

    for pdf_s, bins_s, pdf_gt, bins_gt in zip(pdf, bins,
                                              dce_mod.pdf_series_,
                                              dce_mod.bin_series_):
        assert_array_equal(pdf_s, pdf_gt)
        assert_array_equal(bins_s, bins_gt)


def test_build_heatmap_roi():
    """ Test if the heatmap is built properly when providing a ROI. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)
    dce_mod.data_ /= 2.
    dce_mod.update_histogram()

    # Create some ground-truth
    pos = np.ones((368, 448), dtype=bool)
    neg = np.zeros((368, 448), dtype=bool)
    gt_index = np.rollaxis(np.array([neg, pos, pos, pos, neg]), 0, 3)

    # Build the heatmap
    heatmap, bins_heatmap = dce_mod.build_heatmap(roi_data=(gt_index))

    # Check that heatmap is what we expect
    data = np.load(os.path.join(currdir, 'data', 'heatmap_roi_mod.npy'))
    assert_array_equal(heatmap, data)
    data = np.load(os.path.join(currdir, 'data', 'bins_heatmap_roi_mod.npy'))
    assert_array_equal(bins_heatmap, data)


def test_build_heatmap_none_bins():
    """ Test if the heatmap is built properly with None value for bins."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Build the heatmap
    heatmap, bins_heatmap = dce_mod.build_heatmap(nb_bins=None)

    # Check that heatmap is what we expect
    data = np.load(os.path.join(currdir, 'data', 'heatmap.npy'))
    assert_array_equal(heatmap, data)
    data = np.load(os.path.join(currdir, 'data', 'bins_heatmap.npy'))
    assert_array_equal(bins_heatmap, data)


def test_get_pdf_no_data():
    """Test either if an error is raised when a pdf is asked with not
    data opened."""

    # Create the object
    dce_mod = DCEModality()
    assert_raises(ValueError, dce_mod.get_pdf_list)


def test_get_pdf_wrong_bins():
    """Test either if an error is raised with inconsistent number of bins."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Get the pdf with the wrong number of series in nb_bins
    # There is only two series
    nb_bins = [100, 100, 100]
    assert_raises(ValueError, dce_mod.get_pdf_list, nb_bins=nb_bins)


def test_update_histogram_wrong_bins():
    """Test either if an error is raised with wrong number of bins when
    updating the histogram."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Get the pdf with the wrong number of series in nb_bins
    # There is only two series
    nb_bins = [100, 100, 100]
    assert_raises(ValueError, dce_mod.update_histogram, nb_bins=nb_bins)


def test_get_pdf_nb_bins_str_unknown():
    """ Test either if an error is raised when the string for `nb_bins`
    is unknown."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Pass an unknown string for `nb_bins`
    assert_raises(ValueError, dce_mod.get_pdf_list,
                  roi_data=None, nb_bins='rnd')


def test_get_pdf_nb_bins_wrong_type():
    """ Test either if an error is raised when an unknown parameter type
    is passed."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Pass a single integer which is an unknown type
    assert_raises(ValueError, dce_mod.get_pdf_list, roi_data=None, nb_bins=10)


def test_update_histogram_wrong_string():
    """Test either if an error is raised when an unknown string is provided
    for the arguments `nb_bins`."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Get the pdf with the wrong number of series in nb_bins
    # There is only two series
    assert_raises(ValueError, dce_mod.update_histogram, nb_bins='rnd')


def test_update_histogram_wrong_bins_type():
    """Test either if an error is raised with an inconsistent type of data
    in a list."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Get the pdf with the wrong number of series in nb_bins
    # There is only two series
    nb_bins = [100, 100, 'a']
    assert_raises(ValueError, dce_mod.update_histogram, nb_bins=nb_bins)


def test_update_histogram_wrong_bins_type_2():
    """Test either if an error is raised with an inconsistent type of data
    in a list."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Get the pdf with the wrong number of series in nb_bins
    # There is only two series
    nb_bins = [100, 'a']
    assert_raises(ValueError, dce_mod.update_histogram, nb_bins=nb_bins)


def test_update_histogram_wrong_instance():
    """Test either if an error is raised with an type for `nb_bins`
    argument."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Get the pdf with the wrong number of series in nb_bins
    # There is only two series
    assert_raises(ValueError, dce_mod.update_histogram, nb_bins=10)


def test_update_histogram_fix_bins():
    """ Test that the function properly update the value of the histogram
    and fixing the number of bins. """

    # Load the data and then call the function independently
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    dce_mod.read_data_from_path(path_data)

    # Change something in the data to check that the computation
    # is working
    dce_mod.data_[0, 20:40, :, :] = 1000.
    nb_bins = [100, 100]
    dce_mod.update_histogram(nb_bins=nb_bins)

    # We need to check that the minimum and maximum were proprely computed
    assert_equal(dce_mod.min_series_, 0.)
    assert_equal(dce_mod.max_series_, 1000.)

    # Check that bin is what we expect
    data = np.load(os.path.join(currdir, 'data',
                                'bin_dce_data_update_100_bins.npy'))
    # Check that each array are the same
    for exp, gt in zip(dce_mod.bin_series_, data):
        assert_array_equal(exp, gt)

    # Check that pdf is what we expect
    data = np.load(os.path.join(currdir, 'data',
                                'pdf_dce_data_update_100_bins.npy'))
    # Check that each array are the same
    for exp, gt in zip(dce_mod.pdf_series_, data):
        assert_array_equal(exp, gt)


# def test_dce_compute_aif():
#     """Test the function compute the AIF"""

#     # Load the data with only a single serie
#     currdir = os.path.dirname(os.path.abspath(__file__))
#     path_data = os.path.join(currdir,
#                              '../../preprocessing/tests/data/full_dce')

#     # Create an object to handle the data
#     dce_mod = DCEModality()

#     dce_mod.read_data_from_path(path_data)

#     # Compute the AIF
#     print dce_mod.compute_aif()
