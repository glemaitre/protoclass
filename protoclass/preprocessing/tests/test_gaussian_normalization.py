""" Test the Gaussian normalization. """

import os

from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_warns
from numpy.testing import assert_almost_equal

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import GaussianNormalization

DECIMAL_PRECISON = 3

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


def test_gn_fit_wrong_modality():
    """ Test either if an error is raised in case that a wrong
    modality is provided for fitting. """

    # Create a DCEModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_dce = os.path.join(currdir, 'data', 'dce')
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data=path_data_dce)

    # Create the Gaussian normalization object
    gaussian_norm = GaussianNormalization(T2WModality())

    # Try to make the fitting with another based modality
    assert_raises(ValueError, gaussian_norm.fit, dce_mod)


def test_gn_fit_modality_not_read():
    """ Test either if an error is raised when the data from the
    modality where not read. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)

    # Create the Gaussian normalization object
    gaussian_norm = GaussianNormalization(T2WModality())

    # Call the fitting before to have read the data
    assert_raises(ValueError, gaussian_norm.fit, t2w_mod)


def test_gn_fit_no_gt_1_cat():
    """ Test either if a warning is raised to inform that no
    ground-truth has been provided. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the Gaussian normalization object
    gaussian_norm = GaussianNormalization(T2WModality())

    # Call the fitting before to have read the data
    assert_warns(UserWarning, gaussian_norm.fit, t2w_mod, None, 'prostate')
    assert_almost_equal(gaussian_norm.mu_, 91.3885267391,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(gaussian_norm.sigma_, 156.874802646,
                        decimal=DECIMAL_PRECISON)


def test_gn_fit_no_cat():
    """ Test either if an error is raised when the category of the
    ground-truth is not provided and ground-truth is. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

    # Try to normalize without saying which label to use
    gaussian_norm = GaussianNormalization(T2WModality())
    assert_raises(ValueError, gaussian_norm.fit, t2w_mod, gt_mod)


def test_gn_fit_gt_not_read():
    """ Test either if the ground-truth is read or not. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()

    # Try to normalize without saying which label to use
    gaussian_norm = GaussianNormalization(T2WModality())
    assert_raises(ValueError, gaussian_norm.fit, t2w_mod, gt_mod, label_gt[0])


def test_gn_fit_wrong_gt():
    """ Test either if an error is raised when the wrong class is provided for
    the ground-truth. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

    gaussian_norm = GaussianNormalization(T2WModality())
    assert_raises(ValueError, gaussian_norm.fit, t2w_mod, t2w_mod, label_gt[0])


def test_gn_wrong_label_gt():
    """ Test either if an error is raised when the category label asked was
    not load by the GTModality. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

    gaussian_norm = GaussianNormalization(T2WModality())
    assert_raises(ValueError, gaussian_norm.fit, t2w_mod, gt_mod, 'None')


def test_gn_wrong_size_gt():
    """ Test either if an error is raised when the size of the ground-truth
    is different from the size of the base modality. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

    # Change the size of the data of the modality
    t2w_mod.data_ = t2w_mod.data_[:-1, :, :]

    gaussian_norm = GaussianNormalization(T2WModality())
    assert_raises(ValueError, gaussian_norm.fit, t2w_mod, gt_mod, label_gt[0])


def test_gn_fit_wrong_params():
    """ Test either if an error is raised when the params change just before
    fitting. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

    gaussian_norm = GaussianNormalization(T2WModality())
    gaussian_norm.params = 'None'
    assert_raises(ValueError, gaussian_norm.fit, t2w_mod, gt_mod, label_gt[0])


def test_gn_fit_wt_gt():
    """ Test the fitting routine without ground-truth. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    gaussian_norm = GaussianNormalization(T2WModality())
    gaussian_norm.fit(t2w_mod)
    assert_almost_equal(gaussian_norm.mu_, 91.3885267391,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(gaussian_norm.sigma_, 156.874802646,
                        decimal=DECIMAL_PRECISON)

def test_gn_fit_auto():
    """ Test the fitting routine with auto parameters. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

    gaussian_norm = GaussianNormalization(T2WModality())
    gaussian_norm.fit(t2w_mod, gt_mod, label_gt[0])
    assert_almost_equal(gaussian_norm.mu_, 250.20018737,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(gaussian_norm.sigma_, 65.9800837658,
                        decimal=DECIMAL_PRECISON)

def test_gn_fit_fix_mu_sigma():
    """ Test the fitting routine with fixed mean and std. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the GTModality object
    path_data_gt = os.path.join(currdir, 'data', 'gt_folders')
    path_data_gt_list = [os.path.join(path_data_gt, 'prostate'),
                         os.path.join(path_data_gt, 'pz'),
                         os.path.join(path_data_gt, 'cg'),
                         os.path.join(path_data_gt, 'cap')]
    label_gt = ['prostate', 'pz', 'cg', 'cap']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(cat_gt=label_gt, path_data=path_data_gt_list)

    params = {'mu' : 1., 'sigma': 3.}
    gaussian_norm = GaussianNormalization(T2WModality(), params=params)
    gaussian_norm.fit(t2w_mod, gt_mod, label_gt[0])
    assert_almost_equal(gaussian_norm.mu_, 250.20018737,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(gaussian_norm.sigma_, 65.9800837658,
                        decimal=DECIMAL_PRECISON)
