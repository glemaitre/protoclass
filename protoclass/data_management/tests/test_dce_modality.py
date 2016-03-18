""" Test the class DCE modality. """

import numpy as np
import os

from numpy.testing import assert_equal
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
    """ Test if we can read 2 dce series """

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality(path_data)

    dce_mod.read_data_from_path()

    # Check the type of the data
    assert_equal(dce_mod.data_.dtype, np.float64)
    # Check that the dimension are the one that we expect
    assert_equal(dce_mod.data_.shape, (368, 448, 64, 2))
