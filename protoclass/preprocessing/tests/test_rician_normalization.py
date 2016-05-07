""" Test the Rician normalization. """

import os

import numpy as np

from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_warns
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import RicianNormalization

DECIMAL_PRECISON = 1


def test_rn_init_wrong_base_modality():
    """ Test either if an error is raised when a wrong base modality
    is given. """

    # Check that an error is raised with incompatible type of
    # base modality
    assert_raises(ValueError, RicianNormalization, DCEModality())


def test_rn_init_wrong_type_params():
    """ Test either if an error is raised when the type of params
    is wrong. """

    # Check that an error is raised with incompatible type of
    # params
    assert_raises(ValueError, RicianNormalization, T2WModality(), 1)


def test_rn_init_string_unknown():
    """ Test either if an error is raised when the string in the params
    is unknown. """

    # Force the params parameter to be unknown with a string type
    assert_raises(ValueError, RicianNormalization, T2WModality(), 'None')


def test_rn_init_dict_wrong_dict():
    """ Test either if an error is raised when mu and sigma are not
    present in the dictionary. """

    # Create a dictionary with unknown variable
    params = {'None': 1., 'Stupid': 2.}
    assert_raises(ValueError, RicianNormalization, T2WModality(), params)


def test_rn_init_dict_not_float():
    """ Test either if an error is raised if mu and sigma are not of
    type float. """

    # Create a dictionary with unknown variable
    params = {'b': 'cccc', 'off': 'bbbb', 'sigma': 'aaaa'}
    assert_raises(ValueError, RicianNormalization, T2WModality(), params)


def test_rn_init_dict_extra_param():
    """ Test either if an error is raised when an unknown key is given
    additionaly inside the dictionary. """

    # Create a dictionary with unknown variable
    params = {'b': 10., 'off': 1., 'sigma': 3., 'None': 1.}
    assert_raises(ValueError, RicianNormalization, T2WModality(), params)


def test_rn_init_auto():
    """ Test the constructor with params equal to 'auto'. """

    # Create the object
    params = 'auto'
    obj = RicianNormalization(T2WModality(), params)
    # Check some members variable from the object
    assert_equal(obj.is_fitted_, False)
    assert_equal(obj.params, params)
    assert_equal(obj.fit_params_, None)


def test_rn_init_dict():
    """ Test the constructor with params equal to a given dict. """

    # Create the object
    params = {'b': .1, 'off': 1., 'sigma': 3.}
    obj = RicianNormalization(T2WModality(), params)
    # Check some members variable from the object
    assert_equal(obj.is_fitted_, False)
    assert_equal(cmp(obj.fit_params_, params), 0)


def test_rn_fit_wrong_modality():
    """ Test either if an error is raised in case that a wrong
    modality is provided for fitting. """

    # Create a DCEModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_dce = os.path.join(currdir, 'data', 'dce')
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data=path_data_dce)

    # Create the Rician normalization object
    rician_norm = RicianNormalization(T2WModality())

    # Try to make the fitting with another based modality
    assert_raises(ValueError, rician_norm.fit, dce_mod)


def test_rn_fit_modality_not_read():
    """ Test either if an error is raised when the data from the
    modality where not read. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)

    # Create the Rician normalization object
    rician_norm = RicianNormalization(T2WModality())

    # Call the fitting before to have read the data
    assert_raises(ValueError, rician_norm.fit, t2w_mod)


def test_rn_fit_no_gt_1_cat():
    """ Test either if a warning is raised to inform that no
    ground-truth has been provided. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    # Create the Rician normalization object
    rician_norm = RicianNormalization(T2WModality())

    # Call the fitting before to have read the data
    assert_warns(UserWarning, rician_norm.fit, t2w_mod, None, 'prostate')
    assert_almost_equal(rician_norm.fit_params_['b'], 162.71,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['off'], 0.0,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['sigma'], 148.84,
                        decimal=DECIMAL_PRECISON)


def test_rn_fit_no_cat():
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
    rician_norm = RicianNormalization(T2WModality())
    assert_raises(ValueError, rician_norm.fit, t2w_mod, gt_mod)


def test_rn_fit_gt_not_read():
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
    rician_norm = RicianNormalization(T2WModality())
    assert_raises(ValueError, rician_norm.fit, t2w_mod, gt_mod, label_gt[0])


def test_rn_fit_wrong_gt():
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

    rician_norm = RicianNormalization(T2WModality())
    assert_raises(ValueError, rician_norm.fit, t2w_mod, t2w_mod, label_gt[0])


def test_rn_wrong_label_gt():
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

    rician_norm = RicianNormalization(T2WModality())
    assert_raises(ValueError, rician_norm.fit, t2w_mod, gt_mod, 'None')


def test_rn_wrong_size_gt():
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

    rician_norm = RicianNormalization(T2WModality())
    assert_raises(ValueError, rician_norm.fit, t2w_mod, gt_mod, label_gt[0])


def test_rn_fit_wrong_params():
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

    rician_norm = RicianNormalization(T2WModality())
    rician_norm.params = 'None'
    assert_raises(ValueError, rician_norm.fit, t2w_mod, gt_mod, label_gt[0])


def test_rn_fit_wt_gt():
    """ Test the fitting routine without ground-truth. """

    # Create a T2WModality object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    t2w_mod.read_data_from_path()

    rician_norm = RicianNormalization(T2WModality())
    rician_norm.fit(t2w_mod)
    assert_almost_equal(rician_norm.fit_params_['b'], 162.71,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['off'], 0.0,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['sigma'], 148.84,
                        decimal=DECIMAL_PRECISON)


def test_rn_fit_auto():
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

    rician_norm = RicianNormalization(T2WModality())
    rician_norm.fit(t2w_mod, gt_mod, label_gt[0])
    assert_almost_equal(rician_norm.fit_params_['b'], 259.37,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['off'], 7.00,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['sigma'], 83.45,
                        decimal=DECIMAL_PRECISON)

def test_rn_fit_fix_params():
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

    params = {'b': 200., 'off': 7., 'sigma': 80.}
    rician_norm = RicianNormalization(T2WModality(), params=params)
    rician_norm.fit(t2w_mod, gt_mod, label_gt[0])
    assert_almost_equal(rician_norm.fit_params_['b'], 200.,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['off'], 7.00,
                        decimal=DECIMAL_PRECISON)
    assert_almost_equal(rician_norm.fit_params_['sigma'], 80.00,
                        decimal=DECIMAL_PRECISON)

def test_rn_normalize():
    """ Test the normalize function. """

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

    # Store the data before the normalization
    pdf_copy = t2w_mod.pdf_.copy()
    data_copy = t2w_mod.data_.copy()

    # Normalize the data
    rician_norm = RicianNormalization(T2WModality())
    rician_norm.fit(t2w_mod, gt_mod, 'prostate')

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import rice

    plt.figure()
    pdf, bins = t2w_mod.get_pdf(gt_mod.extract_gt_data('prostate'))
    center = (bins[:-1] + bins[1:]) / 2.
    plt.bar(center, pdf, align='center')
    print rician_norm.fit_params_['b']
    print rician_norm.fit_params_['off']
    print rician_norm.fit_params_['sigma']
    print rice.pdf(center,
                   rician_norm.fit_params_['b'],
                   rician_norm.fit_params_['off'],
                   rician_norm.fit_params_['sigma'])
    print center.shape
    plt.plot(center, rice.pdf(center,
                              rician_norm.fit_params_['b'],
                              rician_norm.fit_params_['off'],
                              rician_norm.fit_params_['sigma']))
    plt.savefig('histogram.png')

    t2w_mod = rician_norm.normalize(t2w_mod)

    # Check that the data are equal to what they should be
    assert_array_almost_equal(t2w_mod.data_, (data_copy - 245.90) / 74.31,
                              decimal=DECIMAL_PRECISON)

    # Denormalize the data
    t2w_mod = rician_norm.denormalize(t2w_mod)

    # Check that the data are equal to the original data
    data = np.load(os.path.join(currdir, 'data', 'data_denormalize.npy'))
    assert_array_equal(t2w_mod.data_, data)
    data = np.load(os.path.join(currdir, 'data', 'pdf_denormalize.npy'))
    assert_array_equal(t2w_mod.pdf_, data)


def test_normalize_wt_fitting():
    """Test either an error is raised if the data are not fitted first."""
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

    # Store the data before the normalization
    pdf_copy = t2w_mod.pdf_.copy()
    data_copy = t2w_mod.data_.copy()

    # Normalize the data
    rician_norm = RicianNormalization(T2WModality())
    assert_raises(ValueError, rician_norm.normalize, t2w_mod)


def test_denormalize_wt_fitting():
    """Test either an error is raised if the data are not fitted first."""
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

    # Store the data before the normalization
    pdf_copy = t2w_mod.pdf_.copy()
    data_copy = t2w_mod.data_.copy()

    # Normalize the data
    rician_norm = RicianNormalization(T2WModality())
    assert_raises(ValueError, rician_norm.denormalize, t2w_mod)


def test_rn_save_load():
    """ Test the save and load function. """

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

    # Normalize the data
    rician_norm = RicianNormalization(T2WModality())
    rician_norm.fit(t2w_mod, gt_mod, 'prostate')

    # Store the normalization object
    filename = os.path.join(currdir, 'data', 'rn_obj.p')
    rician_norm.save_to_pickles(filename)

    # Load the object
    rn_2 = RicianNormalization.load_from_pickles(filename)

    # Check that the different variables are the same
    assert_equal(type(rn_2.base_modality_), type(rician_norm.base_modality_))
    assert_equal(rn_2.fit_params_['b'], rician_norm.fit_params_['b'])
    assert_equal(rn_2.fit_params_['off'], rician_norm.fit_params_['off'])
    assert_equal(rn_2.fit_params_['sigma'], rician_norm.fit_params_['sigma'])
    assert_equal(rn_2.is_fitted_, rician_norm.is_fitted_)
    assert_array_equal(rn_2.roi_data_, rician_norm.roi_data_)
