""" Test the function allowing input/output of DICOM data. """

import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from ..io_dicom import read_dce_serie_dicom


def test_read_dce_dicom_no_dir():
    """ Test if an error is raised when the directory does not exist. """

    # Create a dummy named directory
    path_data = 'None'
    # Check that an error is risen
    assert_raises(ValueError, read_dce_serie_dicom, path_data)


def test_read_dce_dicom_less_2_serie():
    """ Test if an error is raised if there is not at least 2 series. """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Check the assert
    assert_raises(ValueError, read_dce_serie_dicom, path_data)


def test_read_dce_data():
    """ Test if we can read 2 dce series """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')

    serie_data_dce = read_dce_serie_dicom(path_data)

    # Check the type of the data
    assert_equal(serie_data_dce.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(serie_data_dce.shape, (364, 512, 64, 2))
