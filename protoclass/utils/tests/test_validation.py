""" Test different validation methods. """

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_raises

from protoclass.utils.validation import check_path_data
from protoclass.utils.validation import check_modality
from protoclass.utils.validation import check_img_filename

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality


def test_check_path_data_str_exist():
    """ Test that the path is return properly when it exists. """
    # Build an existing path
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = check_path_data(os.path.join(currdir, 'data', 'path_data'))

    # Check if path_data is correct
    assert_equal(path_data, os.path.join(currdir, 'data', 'path_data'))


def test_check_path_data_str_unknown():
    """ Test either if an error is raised in the case that the path does
    not extist. """
    # Create a random path
    currdir = 'None'

    # Check that the error is raised
    assert_raises(ValueError, check_path_data, currdir)


def test_check_path_data_str_wrong_type():
    """ Test either an error is raised when a wrong type is given as path. """
    # Build a list of path
    path_data = 1

    assert_raises(ValueError, check_path_data, path_data)


def test_check_path_data_lst_str_exist():
    """ Test that the paths is return properly when they are existing. """
    # Build a list of path
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_list = [os.path.join(currdir, 'data', 'path_list_data', 's_1'),
                      os.path.join(currdir, 'data', 'path_list_data', 's_2')]
    path_data_list_res = check_path_data(path_data_list)

    # Check that each element of the list is equal
    for path_gt, path_res in zip(path_data_list, path_data_list_res):
        assert_equal(path_gt, path_res)


def test_check_path_data_lst_str_wrong_type():
    """ Test either an error is raised when a wrong type is inserted in the
    list of path. """
    # Build a list of path
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_list = [1,
                      os.path.join(currdir, 'data', 'path_list_data', 's_1'),
                      os.path.join(currdir, 'data', 'path_list_data', 's_2')]

    assert_raises(ValueError, check_path_data, path_data_list)


def test_check_path_data_lst_str_unknown():
    """ Test either an error is raised when there is some unknown path
    inside the list. """
    # Build a list of unknown path
    path_data_list = ['None', 'None']

    assert_raises(ValueError, check_path_data, path_data_list)


def test_check_modality_wrong_modality():
    """ Test that an error is raised when two modalities are not the same when
    checking their types. """
    assert_raises(ValueError, check_modality, DCEModality(), T2WModality())


def test_check_modality():
    """ Test that everything is fine when the modalities are the same. """
    assert_equal(check_modality(T2WModality(), T2WModality()), None)


def test_check_img_filename_no_file():
    """ Test if an error is raised when the file does not exist. """
    assert_raises(ValueError, check_img_filename, 'random.rnd')


def test_check_img_filename_not_img():
    """ Test if an error is raised when the file is not of img extension. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'path_list_data',
                            's_2', 'README.md')
    assert_raises(ValueError, check_img_filename, filename)


def test_check_img_filename():
    """ Test the routine to check if the file is of type img. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'path_list_data',
                            's_2', 'README.img')

    assert_equal(ValueError, check_img_filename(filename), filename)
