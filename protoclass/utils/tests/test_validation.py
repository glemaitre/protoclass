"""Test different validation methods."""

import os

import numpy as np

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from protoclass.utils.validation import check_path_data
from protoclass.utils.validation import check_modality
from protoclass.utils.validation import check_img_filename
from protoclass.utils.validation import check_npy_filename
from protoclass.utils.validation import check_filename_pickle_load
from protoclass.utils.validation import check_filename_pickle_save
from protoclass.utils.validation import check_modality_inherit
from protoclass.utils.validation import check_modality_gt

from protoclass.data_management import StandaloneModality
from protoclass.data_management import TemporalModality
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
    filename = os.path.join(currdir, 'data', 'README.img')

    assert_equal(check_img_filename(filename), filename)


def test_check_img_filename_wrong_type():
    """ Test if an error is raised when the type is wrong. """
    assert_raises(ValueError, check_img_filename, 1)


def test_check_npy_filename_no_file():
    """ Test if an error is raised when the file does not exist. """
    assert_raises(ValueError, check_npy_filename, 'random.rnd')


def test_check_npy_filename_not_npy():
    """ Test if an error is raised when the file is not of npy extension. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'path_list_data',
                            's_2', 'README.md')
    assert_raises(ValueError, check_npy_filename, filename)


def test_check_npy_filename():
    """ Test the routine to check if the file is of type npy. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'README.npy')

    assert_equal(check_npy_filename(filename), filename)


def test_check_npy_filename_wrong_type():
    """ Test if an error is raised when the type is wrong. """
    assert_raises(ValueError, check_npy_filename, 1)


def test_check_filename_save_wrong_type():
    """ Test either if an error is raised when the type is the wrong one. """
    assert_raises(ValueError, check_filename_pickle_save, 1)


def test_check_filename_save_wrong_ext():
    """ Test either if an error is raised when the extension of the filename
    is wrong. """
    assert_raises(ValueError, check_filename_pickle_save, 'random.rnd')


def test_check_filename_pickle_save():
    """ Test the routine to check the pickle filename is working. """
    filename = 'random.p'
    out_filename = check_filename_pickle_save(filename)

    assert_equal(out_filename, out_filename)


def test_check_filename_pickle_load_wrong_type():
    """ Test either if an error is raised when the wrong type is given. """
    assert_raises(ValueError, check_filename_pickle_load, 1)


def test_check_filename_pickle_load_not_fit():
    """ Test either an error is raised when the file is not with npy
    extension. """
    assert_raises(ValueError, check_filename_pickle_load, 'file.rnd')


def test_check_filename_pickle_load_not_exist():
    """ Test either if an error is raised when the file is not existing. """
    assert_raises(ValueError, check_filename_pickle_load, 'file.p')


def test_check_filename_pickle_load():
    """ Test the routine to check the filename is pickle. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'random.p')
    my_filename = check_filename_pickle_load(filename)

    assert_equal(my_filename, filename)


def test_check_modality_inherit_not_subclass():
    """Test if an error is raised when a modality is not inheriting
    from another class."""

    # Create an object
    dce_mod = DCEModality()

    # Check that an error is raised with Standalone classinfo
    assert_raises(ValueError, check_modality_inherit, dce_mod,
                  StandaloneModality)


def test_check_modality_inherit():
    """Test the routine to check class inherency."""

    # Create an object
    dce_mod = DCEModality()
    dce_mod_2 = check_modality_inherit(dce_mod, TemporalModality)

    assert_equal(dce_mod, dce_mod_2)


def test_check_modality_gt_not_gt():
    """Test if an error is raised when the gt is not a GTModality object."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)

    assert_raises(ValueError, check_modality_gt, t2w_mod, t2w_mod, 'prostate')


def test_check_modality_gt_not_read():
    """Test if an error is raised when the gt is not read."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()

    assert_raises(ValueError, check_modality_gt, t2w_mod, gt_mod, label_gt[0])


def test_check_modality_gt_mod_not_read():
    """Test if an error is raised when the modality is not read."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    assert_raises(ValueError, check_modality_gt, t2w_mod, gt_mod, label_gt[0])


def test_check_modality_gt_not_read():
    """Test if an error is raised when the gt is not read."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    assert_raises(ValueError, check_modality_gt, t2w_mod, gt_mod, label_gt[0])


def test_check_modality_gt_mod_not_read():
    """Test if an error is raised when the modality is not read."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    roi_data = check_modality_gt(t2w_mod, gt_mod, label_gt[0])
    data = np.load(os.path.join(currdir, 'data', 'roi_data.npy'))

    # Check each array of the tuple
    for i in range(len(roi_data)):
        assert_array_equal(roi_data[i], data[i])
