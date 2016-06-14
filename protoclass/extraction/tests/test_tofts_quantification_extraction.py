"""Test the Tofts qunatification extraction."""

import os

import numpy as np

from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_warns

from protoclass.extraction import ToftsQuantificationExtraction

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

DECIMAL_PRECISION = 5
RND_SEED = 0

# Define the variables which will be shared across test
T10 = 1.6
CA = 3.5


def test_tqe_bad_mod():
    """Test either if an error is raised when the base modality does not
    inherate from TemporalModality."""

    # Try to create the normalization object with the wrong class object
    assert_raises(ValueError, ToftsQuantificationExtraction, T2WModality(),
                  T10, CA)


def test_tqe_bad_mod_fit():
    """Test either if an error is raised when a modality to fit does not
    correspond to the template modality given at the construction."""

    # Create the normalization object with the right modality
    dce_tqe = ToftsQuantificationExtraction(DCEModality(), T10, CA)

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)
    # Fit and raise the error
    assert_raises(ValueError, dce_tqe.fit, t2w_mod)


def test_tqe_not_read_mod_fit():
    """Test either if an error is raised when the modality has not been
    read before fitting."""

    # Create the normalization object with the right modality
    dce_tqe = ToftsQuantificationExtraction(DCEModality(), T10, CA)

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Fit and raise the error
    assert_raises(ValueError, dce_tqe.fit, dce_mod)


def test_tqe_bad_mod_transform():
    """Test either if an error is raised when a modality to tranform does not
    correspond to the template modality given at the construction."""

    # Create the normalization object with the right modality
    dce_tqe = ToftsQuantificationExtraction(DCEModality(), T10, CA)

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)
    # Fit and raise the error
    assert_raises(RuntimeError, dce_tqe.transform, dce_mod)


def test_tqe_compute_aif_bad_exccentricity():
    """Test either if an error is raised when a wrong eccentricity is given to
    compute the AIF."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Define the eccentricity with to large number
    eccentricity = 100.
    assert_raises(ValueError, ToftsQuantificationExtraction.compute_aif,
                  dce_mod, eccentricity=eccentricity)
    eccentricity = -1.
    assert_raises(ValueError, ToftsQuantificationExtraction.compute_aif,
                  dce_mod, eccentricity=eccentricity)


def test_tqe_compute_aif_bad_threshold():
    """Test either if an error is raised when a wrong selection threshold is
    given to compute the AIF."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Define the eccentricity with to large number
    thres_sel = 100.
    assert_raises(ValueError, ToftsQuantificationExtraction.compute_aif,
                  dce_mod, thres_sel=thres_sel)
    thres_sel = -1.
    assert_raises(ValueError, ToftsQuantificationExtraction.compute_aif,
                  dce_mod, thres_sel=thres_sel)


def test_tqe_compute_aif_bad_estimator():
    """Test either if an error is raised when a wrong estimator is
    given to compute the AIF."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Define the eccentricity with to large number
    estimator = 'rnd'
    assert_raises(ValueError, ToftsQuantificationExtraction.compute_aif,
                  dce_mod, estimator=estimator)


def test_tqe_compute_aif_dce_not_read():
    """Test either if an error is raised when the DCE are not read
    before to compute the AIF."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Define the eccentricity with to large number
    assert_raises(RuntimeError, ToftsQuantificationExtraction.compute_aif,
                  dce_mod)


def test_tqe_compute_aif_default():
    """Test the AIF computation when the default parameters are used."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Compute the AIF
    signal_aif = ToftsQuantificationExtraction.compute_aif(
        dce_mod, random_state=RND_SEED)
    aif_gt = np.array([379., 366., 343., 355., 367., 470., 613., 628., 604.,
                       575.])
    assert_array_equal(signal_aif, aif_gt)


def test_tqe_compute_aif_mean():
    """Test the AIF computation when the mean esatimator is used."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Compute the AIF
    signal_aif = ToftsQuantificationExtraction.compute_aif(
        dce_mod, estimator='mean', random_state=RND_SEED)
    aif_gt = np.array([347.29533, 332.32211, 317.53709, 322.32994, 336.03532,
                       441.30315, 586.89144, 598.05404, 585.32235, 562.42261])
    assert_array_almost_equal(signal_aif, aif_gt, decimal=DECIMAL_PRECISION)


def test_tqe_compute_aif_max():
    """Test the AIF computation when the max esatimator is used."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Compute the AIF
    signal_aif = ToftsQuantificationExtraction.compute_aif(
        dce_mod, estimator='max', random_state=RND_SEED)
    aif_gt = np.array([503., 482., 493., 467., 504., 648., 816., 850., 827.,
                       787.])
    assert_array_equal(signal_aif, aif_gt)


def test_tqe_compute_fit_no_aif():
    """Test the fit function."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Create the object for Tofts quantification extraction
    tqe = ToftsQuantificationExtraction(DCEModality(), 1.6, 3.5,
                                        random_state=RND_SEED)

    # Perform the fitting
    tqe.fit(dce_mod, fit_aif=False)

    # Check the value fitted
    assert_almost_equal(tqe.TR_, 0.00324, decimal=DECIMAL_PRECISION)
    assert_almost_equal(tqe.flip_angle_, 10., decimal=DECIMAL_PRECISION)
    assert_equal(tqe.start_enh_, 3)
    cp_r_gt = np.array([0., 0., 0., 0.13859428, 6.23675492, 6.90344512,
                        1.80619315, 2.22619032, 3.69060743, 3.32021637])
    assert_array_almost_equal(tqe.cp_t_, cp_r_gt, decimal=DECIMAL_PRECISION)


def test_tqe_compute_fit_aif():
    """Test the fit function."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Create the object for Tofts quantification extraction
    tqe = ToftsQuantificationExtraction(DCEModality(), 1.6, 3.5,
                                        random_state=RND_SEED)

    # Perform the fitting
    tqe.fit(dce_mod, fit_aif=True)

    # Check the value fitted
    assert_almost_equal(tqe.TR_, 0.00324, decimal=DECIMAL_PRECISION)
    assert_almost_equal(tqe.flip_angle_, 10., decimal=DECIMAL_PRECISION)
    assert_equal(tqe.start_enh_, 3)
    cp_r_gt = np.array([3.71038e-02, 2.35853e-02, 4.21997e-13, 1.22529e-02,
                        2.46203e-02, 1.35724e-01, 3.06310e-01, 3.25429e-01,
                        2.94957e-01, 2.58964e-01])
    assert_array_almost_equal(tqe.cp_t_, cp_r_gt, decimal=DECIMAL_PRECISION)


def test_tqe_pop_aif():
    """Test the function to generate high-resolution AIF from
    population-based estimate."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Compute the AIF
    signal_aif = ToftsQuantificationExtraction.population_based_aif(dce_mod)

    signal_aif_gt = np.array([0.08038468, 3.61731785, 4.00399817, 1.04759202,
                              1.29119039, 2.14055231, 1.9257255, 1.84094174,
                              1.80151533, 1.76710282])

    assert_array_almost_equal(signal_aif, signal_aif_gt,
                              decimal=DECIMAL_PRECISION)


def test_tqe_conv_signal_conc_wt_fitting():
    """Test either if nan error is raised when a conversion is attended without
    fitting the data first."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Create the object for the Tofts extraction
    tqe = ToftsQuantificationExtraction(DCEModality(), 1.6, 3.5)

    # Try to perform a conversion
    signal = np.array([379., 366., 343., 355., 367., 470., 613., 628., 604.,
                       575.])
    assert_raises(RuntimeError, tqe.signal_to_conc, signal, 343.)
    conc = np.array([2.15201846e-02, 1.36794587e-02, 2.44758162e-13,
                        7.10669089e-03, 1.42797742e-02, 7.87199894e-02,
                        1.77659845e-01, 1.88748637e-01, 1.71075044e-01,
                        1.50198853e-01])
    assert_raises(RuntimeError, tqe.conc_to_signal, conc, 343.)


def test_tqe_conv_signal_conc():
    """Test the conversion from signal to concentration."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Create the object for the Tofts extraction
    tqe = ToftsQuantificationExtraction(DCEModality(), 1.6, 3.5)
    tqe.fit(dce_mod, fit_aif=False)

    # Try to perform a conversion
    signal = np.array([379., 366., 343., 355., 367., 470., 613., 628., 604.,
                       575.])

    conc = tqe.signal_to_conc(signal, 343.)
    conc_gt = np.array([2.15201846e-02, 1.36794587e-02, 2.44758162e-13,
                        7.10669089e-03, 1.42797742e-02, 7.87199894e-02,
                        1.77659845e-01, 1.88748637e-01, 1.71075044e-01,
                        1.50198853e-01])
    assert_almost_equal(conc, conc_gt)

    # Apply the back conversion
    signal_back = tqe.conc_to_signal(conc, 343.)
    assert_almost_equal(signal_back, signal, decimal=DECIMAL_PRECISION)


def test_qte_transform_no_fit():
    """Test either if an error is raised when the wrong kind of model is passed
    as argument during the transform."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)
    # Create the gt data
    gt_mod = GTModality()
    gt_cat = ['cap']
    path_data = [os.path.join(
        currdir,
        '../../preprocessing/tests/data/full_gt/cap')]
    gt_mod.read_data_from_path(gt_cat, path_data)

    # Create the object for the Tofts extraction
    tqe = ToftsQuantificationExtraction(DCEModality(), 1.6, 3.5,
                                        random_state=RND_SEED)
    assert_raises(ValueError, tqe.transform, dce_mod, gt_mod, gt_cat[0],
                  kind='rnd')


def test_qte_transform_extended():
    """Test the transform function for extended model."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)
    # Create the gt data
    gt_mod = GTModality()
    gt_cat = ['cap']
    path_data = [os.path.join(
        currdir,
        '../../preprocessing/tests/data/full_gt/cap')]
    gt_mod.read_data_from_path(gt_cat, path_data)

    # Create the object for the Tofts extraction
    tqe = ToftsQuantificationExtraction(DCEModality(), 1.6, 3.5,
                                        random_state=RND_SEED)
    tqe.fit(dce_mod)
    data = tqe.transform(dce_mod, gt_mod, gt_cat[0], kind='extended')

    data_gt = np.load(os.path.join(currdir, 'data/tofts_ext_data.npy'))
    assert_array_almost_equal(data, data_gt, decimal=DECIMAL_PRECISION)


def test_qte_transform_regular():
    """Test the transform function for regular model."""

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir,
                             '../../preprocessing/tests/data/full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)
    # Create the gt data
    gt_mod = GTModality()
    gt_cat = ['cap']
    path_data = [os.path.join(
        currdir,
        '../../preprocessing/tests/data/full_gt/cap')]
    gt_mod.read_data_from_path(gt_cat, path_data)

    # Create the object for the Tofts extraction
    tqe = ToftsQuantificationExtraction(DCEModality(), 1.6, 3.5,
                                        random_state=RND_SEED)
    tqe.fit(dce_mod)
    data = tqe.transform(dce_mod, gt_mod, gt_cat[0], kind='regular')

    data_gt = np.load(os.path.join(currdir, 'data/tofts_reg_data.npy'))
    assert_array_almost_equal(data, data_gt, decimal=DECIMAL_PRECISION)
