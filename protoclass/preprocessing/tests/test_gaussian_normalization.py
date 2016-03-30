""" Test the Gaussian normalization. """

import os

from numpy.testing import assert_raises
from numpy.testing import assert_equal

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import GaussianNormalization


def test_gn_init_wrong_base_modality():
    """ Test either if an error is raised when a wrong base modality
    is given. """

    # Check that an error is raised with incompatible type of
    # base modality
    assert_raises(ValueError, GaussianNormalization, DCEModality())


def test_gn_init_wrong_type_params():
    """ Test either if an error is raised when the type of params
    is wrong. """

    # Check that an error is raised with incompatible type of
    # params
    assert_raises(ValueError, GaussianNormalization, T2WModality(), 1)


def test_gn_init_string_unknown():
    """ Test either if an error is raised when the string in the params
    is unknown. """

    # Force the params parameter to be unknown with a string type
    assert_raises(ValueError, GaussianNormalization, T2WModality(), 'None')


def test_gn_init_dict_wrong_dict():
    """ Test either if an error is raised when mu and sigma are not
    present in the dictionary. """

    # Create a dictionary with unknown variable
    params = {'None': 1., 'Stupid': 2.}
    assert_raises(ValueError, GaussianNormalization, T2WModality(), params)


def test_gn_init_dict_not_float():
    """ Test either if an error is raised if mu and sigma are not of
    type float. """

    # Create a dictionary with unknown variable
    params = {'mu': 'bbbb', 'sigma': 'aaaa'}
    assert_raises(ValueError, GaussianNormalization, T2WModality(), params)


def test_gn_init_dict_extra_param():
    """ Test either if an error is raised when an unknown key is given
    additionaly inside the dictionary. """

    # Create a dictionary with unknown variable
    params = {'mu': 1., 'sigma': 3., 'None': 1.}
    assert_raises(ValueError, GaussianNormalization, T2WModality(), params)


def test_gn_init_auto():
    """ Test the constructor with params equal to 'auto'. """

    # Create the object
    params = 'auto'
    obj = GaussianNormalization(T2WModality(), params)
    # Check some members variable from the object
    assert_equal(obj.is_fitted_, False)
    assert_equal(obj.params, params)
    assert_equal(obj.fit_params_, None)


def test_gn_init_dict():
    """ Test the constructor with params equal to a given dict. """

    # Create the object
    params = {'mu': 1., 'sigma': 3.}
    obj = GaussianNormalization(T2WModality(), params)
    # Check some members variable from the object
    assert_equal(obj.is_fitted_, False)
    assert_equal(cmp(obj.fit_params_, params), 0)



# def test_gn_init():
#     """ Test the routine to initialize the Gaussian normalization. """

# def test_gn_fit():
#     """ Test either if an error is raised when the object given as
#     ground-truth is not a GTModality object. """

#     # Create a T2WModality object
#     currdir = os.path.dirname(os.path.abspath(__file__))
#     path_data_t2w = os.path.join(currdir, 'data', 't2w')
#     t2w_mod = T2WModality()
#     t2w_mod.read_data_from_path(path_data=path_data_t2w)

#     # Create the GTModality object
#     path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
#     path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
#                          os.path.join(path_data_gt, 'pz'),
#                          os.path.join(path_data_gt, 'cg'),
#                          os.path.join(path_data_gt, 'cap')]
#     label_gt = ['prostate', 'pz', 'cg', 'cap']
#     gt_mod = GTModality()
#     gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

#     # Create the object for the standalone normalization
#     standalone_norm = StandaloneNormalization(T2WModality())
#     standalone_norm.fit(t2w_mod, gt_mod, 'prostate')
