""" Test the different method allowing to export the results. """

import numpy as np
import os

from numpy.testing import assert_raises
from numpy.testing import assert_equal

from nose.tools import assert_true

import filecmp

from protoclass.utils.export import make_table

data = np.array([[.805, .372, .712, .407, .529],
                 [.296, .051, .292, .227, .173],
                 [.697, .383, .165, .719, .760],
                 [.995, .721, .329, .035, .995],
                 [.686, .589, .869, .859, .697]])

feature_list = ['a', 'b', 'c', 'd', 'e']


def test_make_table_wrong_ext():
    """ Test either an error is raised when the extension is unknown. """
    filename = 'random.rnd'

    assert_raises(ValueError, make_table, data, feature_list, filename)


def test_make_table_wrong_dimension():
    """ Test either if an error is raised if the dimension of the data are
    different from the feature list and vice-versa. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data')
    filename = os.path.join(path_data, 'test.tex')

    feature_corr = ['a', 'b', 'c']
    assert_raises(ValueError, make_table, data, feature_corr, filename)

    data_corr = np.random.random((3, 5))
    data_corr += np.min(data_corr)
    assert_raises(ValueError, make_table, data_corr, feature_list, filename)


def test_make_table_neg():
    """ Test either if an error is raised when negative value are
    present in data. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data')
    filename = os.path.join(path_data, 'test.tex')

    data_corr = np.ones((5, 5)) * -1.
    assert_raises(ValueError, make_table, data_corr, feature_list, filename)

    data_corr = np.ones((5, 5)) * 1000.
    assert_raises(ValueError, make_table, data_corr, feature_list, filename)


def test_make_table_wrong_file():
    """ Test either if an error is raised if the file cannot be created. """
    filename = '/none/random.tex'

    assert_raises(IOError, make_table, data, feature_list, filename)


def test_make_table():
    """ Test the routine to create a table. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data')
    filename = os.path.join(path_data, 'test.tex')

    # Check that the function is going through
    assert_equal(make_table(data, feature_list, filename), None)
    # Check that the tex file created is correct
    assert_true(filecmp.cmp(filename,
                            os.path.join(path_data, 'make_table.tex'),
                            shallow=False))

    # Check that the function is going through by scaling under 100.
    assert_equal(make_table(data * 100, feature_list, filename), None)
    # Check that the tex file created is correct
    assert_true(filecmp.cmp(filename,
                            os.path.join(path_data, 'make_table.tex'),
                            shallow=False))
