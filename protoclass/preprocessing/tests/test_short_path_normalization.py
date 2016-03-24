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

    # Read some T2W image which is not inheriting from TemporalModality to
    # create the object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data)
    t2w_mod.read_data_from_path()

    # Try to create the normalization object
    assert_raises(ValueError, ShortPathNormalization, t2w_mod)


def test_spn_right_mod():
    """ Test if the construction of the normalization object is correct. """

    # Read some DCE images that can be normalized using the
    # TemporalNormalization class
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    dce_mod = DCEModality(path_data)
    dce_mod.read_data_from_path()

    # Create the object and check that it contains the same modality
    dce_norm = ShortPathNormalization(dce_mod)
    assert_equal(dce_norm.base_modality_, dce_mod)
