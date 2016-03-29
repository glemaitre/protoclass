""" Test the short path normalization. """

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from testfixtures import Comparison

from protoclass.preprocessing import ShortPathNormalization

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality


def test_spn_bad_mod():
    """ Test either if an error is raised when the base modality does not
    inherate from TemporalModality. """

    # Try to create the normalization object with the wrong class object
    assert_raises(ValueError, ShortPathNormalization, T2WModality())


def test_snp_bad_mod_fit():
    """ Test either if an error is raised when a modality to fit does not
    correspond to the template modality given at the construction. """

    # Create the normalization object with the right modality
    dce_norm = ShortPathNormalization(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)
    # Fit and raise the error
    assert_raises(ValueError, dce_norm.fit, t2w_mod)


def test_spn_right_fitting():
    """ Test if the construction of the normalization object is correct. """

    # Create the object and check that it contains the same modality
    dce_norm = ShortPathNormalization(DCEModality())

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Fit the modality
    dce_norm.fit(dce_mod)
