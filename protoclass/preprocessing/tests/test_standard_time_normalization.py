"""Test the standard time normalization."""

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

from nose.tools import assert_true

from skimage import img_as_float

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

PRECISION_DECIMAL = 2


def test_stn_bad_mod():
    """Test either if an error is raised when the base modality does not
    inherate from TemporalModality."""

    # Try to create the normalization object with the wrong class object
    assert_raises(ValueError, StandardTimeNormalization, T2WModality())


def test_stn_bad_mod_fit():
    """Test either if an error is raised when a modality to fit does not
    correspond to the template modality given at the construction."""

    # Create the normalization object with the right modality
    dce_norm = StandardTimeNormalization(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)
    # Fit and raise the error
    assert_raises(ValueError, dce_norm.fit, t2w_mod)


def test_build_graph():
    """Test the method to build a graph from the heatmap."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Build a heatmap from the dce data
    # Reduce the number of bins to enforce low memory consumption
    nb_bins = [100] * dce_mod.n_serie_
    heatmap, bins_heatmap = dce_mod.build_heatmap(gt_mod.extract_gt_data(
        label_gt[0]), nb_bins=nb_bins)

    # Build the graph by taking the inverse exponential of the heatmap
    graph = StandardTimeNormalization._build_graph(heatmap, .5)
    graph_dense = graph.toarray()

    data = np.load(os.path.join(currdir, 'data', 'graph.npy'))
    assert_array_equal(graph_dense, data)


def test_walk_through_graph_shortest_path():
    """Test the routine to go through the graph using shortest path."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Build a heatmap from the dce data
    # Reduce the number of bins to enforce low memory consumption
    nb_bins = [10] * dce_mod.n_serie_
    heatmap, bins_heatmap = dce_mod.build_heatmap(gt_mod.extract_gt_data(
        label_gt[0]), nb_bins=nb_bins)

    # Build the graph by taking the inverse exponential of the heatmap
    heatmap_inv_exp = np.exp(img_as_float(1. - (heatmap / np.max(heatmap))))
    graph = StandardTimeNormalization._build_graph(heatmap_inv_exp, .99)

    start_end_tuple = ((0, 6), (3, 6))

    # Call the algorithm to walk through the graph
    path = StandardTimeNormalization._walk_through_graph(graph,
                                                         heatmap_inv_exp,
                                                         start_end_tuple,
                                                         'shortest-path')

    gt_path = np.array([[0, 6], [1, 6], [2, 6], [3, 6]])
    assert_array_equal(path, gt_path)


def test_walk_through_graph_route_through():
    """Test the routine to go through the graph using route through."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Build a heatmap from the dce data
    # Reduce the number of bins to enforce low memory consumption
    nb_bins = [100] * dce_mod.n_serie_
    heatmap, bins_heatmap = dce_mod.build_heatmap(gt_mod.extract_gt_data(
        label_gt[0]), nb_bins=nb_bins)

    # Build the graph by taking the inverse exponential of the heatmap
    heatmap_inv_exp = np.exp(img_as_float(1. - (heatmap / np.max(heatmap))))
    graph = StandardTimeNormalization._build_graph(heatmap_inv_exp, .99)

    start_end_tuple = ((0, 131), (3, 135))

    # Call the algorithm to walk through the graph
    path = StandardTimeNormalization._walk_through_graph(graph,
                                                         heatmap_inv_exp,
                                                         start_end_tuple,
                                                         'route-through-graph')

    gt_path = np.array([[0, 131], [1, 132], [2, 134], [3, 135]])
    assert_array_equal(path, gt_path)


def test_shift_heatmap_wrong_shift():
    """Test if an error is raised when the shidt provided is not consistent."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Build a heatmap from the dce data
    # Reduce the number of bins to enforce low memory consumption
    nb_bins = [100] * dce_mod.n_serie_
    heatmap, bins_heatmap = dce_mod.build_heatmap(gt_mod.extract_gt_data(
        label_gt[0]), nb_bins=nb_bins)

    # Create a list of shift which do not have the same number of entries
    # than the heatmap - There is 4 series, let's create only 2
    shift_arr = np.array([10] * 2)

    assert_raises(ValueError, StandardTimeNormalization._shift_heatmap,
                  heatmap, shift_arr)


def test_shift_heatmap():
    """Test the routine which shift the heatmap."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Build a heatmap from the dce data
    # Reduce the number of bins to enforce low memory consumption
    nb_bins = [100] * dce_mod.n_serie_
    heatmap, bins_heatmap = dce_mod.build_heatmap(gt_mod.extract_gt_data(
        label_gt[0]), nb_bins=nb_bins)

    # Create a list of shift which do not have the same number of entries
    # than the heatmap - There is 4 series, let's create only 2
    shift_arr = np.array([10] * 4)

    heatmap_shifted = StandardTimeNormalization._shift_heatmap(heatmap,
                                                               shift_arr)

    data = np.load(os.path.join(currdir, 'data', 'heatmap_shifted.npy'))
    assert_array_equal(heatmap_shifted, data)


def test_partial_fit_model_wrong_params_type():
    """Test either if an error is raised when the parameters do not have
    the right types."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  ground_truth=gt_mod, cat=label_gt[0], params=1)


def test_partial_fit_model_wrong_string():
    """Test either if an error is raised when the string is unknown."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  ground_truth=gt_mod, cat=label_gt[0], params='rnd')


def test_partial_fit_model_dict_missing_params():
    """Test either if an error is raised when a parameters is missing
    in the dictionary."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    params = {'std': 50}
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  ground_truth=gt_mod, cat=label_gt[0], params=params)


def test_partial_fit_model_dict_wrong_params():
    """Test either if an error is raised when a parameters is wrong
    in the dictionary."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    params = {'std': 50., 'exp': 25., 'alpha': .9, 'max_iter': 5, 'rnd': 50}
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  ground_truth=gt_mod, cat=label_gt[0], params=params)


def test_partial_fit_model_dict_wrong_type():
    """Test either if an error is raised when a parameters is a wrong
     type in the dictionary."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    params = {'std': 50., 'exp': 25., 'alpha': .9, 'max_iter': 5.}
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  ground_truth=gt_mod, cat=label_gt[0], params=params)

    params = {'std': 50., 'exp': 25, 'alpha': .9, 'max_iter': 5}
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  ground_truth=gt_mod, cat=label_gt[0], params=params)


def test_partial_fit_model():
    """Test the routine to fit the model."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    stn.partial_fit_model(dce_mod, gt_mod, label_gt[0])

    # Check the model computed
    model_gt = np.array([22.26479174, 22.51070962, 24.66027277, 23.43488237,
                         23.75601817, 22.56173871, 26.86244505, 45.06227804,
                         62.34273874, 71.35327656])
    assert_array_almost_equal(stn.model_, model_gt, decimal=PRECISION_DECIMAL)
    assert_true(stn.is_model_fitted_)


def test_partial_fit_model_2():
    """Test the routine to fit two models."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    stn.partial_fit_model(dce_mod, gt_mod, label_gt[0])
    stn.partial_fit_model(dce_mod, gt_mod, label_gt[0])

    # Check the model computed
    model_gt = np.array([22.26479174, 22.51070962, 24.66027277, 23.43488237,
                         23.75601817, 22.56173871, 26.86244505, 45.06227804,
                         62.34273874, 71.35327656])
    assert_array_almost_equal(stn.model_, model_gt, decimal=PRECISION_DECIMAL)
    assert_true(stn.is_model_fitted_)


def test_save_model_not_fitted():
    """Test either if an error is raised if the model is not fitted and
    requested to be stored."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.save_model, os.path.join(currdir, 'data',
                                                           'model.npy'))


def test_save_model_wrong_ext():
    """Test either if an error is raised if the filename as a wrong
    extension while storing the model."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    stn.partial_fit_model(dce_mod, gt_mod, label_gt[0])

    # Try to store the file not with an npy file
    assert_raises(ValueError, stn.save_model, os.path.join(currdir, 'data',
                                                           'model.rnd'))


def test_save_load_model():
    """Test the routine to store and load the model."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    stn.partial_fit_model(dce_mod, gt_mod, label_gt[0])

    # Store the file
    filename = os.path.join(currdir, 'data', 'model.npy')
    stn.save_model(filename)

    # Load the model in another object
    stn2 = StandardTimeNormalization(dce_mod)
    stn2.load_model(filename)

    # Check the different variable
    model_gt = np.array([22.26479174, 22.51070962, 24.66027277, 23.43488237,
                         23.75601817, 22.56173871, 26.86244505, 45.06227804,
                         62.34273874, 71.35327656])
    assert_array_almost_equal(stn.model_, model_gt, decimal=PRECISION_DECIMAL)
    assert_true(stn.is_model_fitted_)


def test_shift_serie():
    """Test the routine for shifting."""

    # Create a synthetic signal
    signal = np.arange(5)

    # Create a shift of 0
    shift = 0
    signal_shift = StandardTimeNormalization._shift_serie(signal, shift)

    # Check the signal
    assert_array_equal(signal_shift, signal)

    # Create a shift of 2
    shift = 2
    signal_shift = StandardTimeNormalization._shift_serie(signal, shift)

    # Check the signal
    gt_signal = np.array([0, 0, 0, 1, 2])
    assert_array_equal(signal_shift, gt_signal)

    # Create a shift of -2
    shift = -2
    signal_shift = StandardTimeNormalization._shift_serie(signal, shift)

    # Check the signal
    gt_signal = np.array([2, 3, 4, 4, 4])
    assert_array_equal(signal_shift, gt_signal)


def test_fit():
    """Test the routine to fit the parameters of the dce normalization."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)

    # Create a synthetic model to fit on
    stn.model_ = np.array([30., 30., 32., 31., 31., 30., 35., 55., 70., 80.])
    stn.is_model_fitted_ = True

    # Fit the parameters on the model
    stn.fit(dce_mod, gt_mod, label_gt[0])

    assert_almost_equal(stn.fit_params_['scale-int'], 1.2296657327848537,
                        decimal=PRECISION_DECIMAL)
    assert_equal(stn.fit_params_['shift-time'], 0.0)
    data = np.array([191.29, 193.28, 195.28, 195.28, 195.28, 197.28, 213.25,
                     249.18, 283.12, 298.10])
    assert_array_almost_equal(stn.fit_params_['shift-int'], data,
                              decimal=PRECISION_DECIMAL)


def test_normalize_denormalize():
    """Test the data normalization and denormalization."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)

    # Simulate that we fitted the data
    stn.model_ = np.array([30., 30., 32., 31., 31., 30., 35., 55., 70., 80.])
    stn.is_model_fitted_ = True
    stn.fit_params_ = {'scale-int': 1.2296657327848537,
                       'shift-time': 0.0,
                       'shift-int': np.array([191.29, 193.28, 195.28, 195.28,
                                              195.28, 197.28, 213.25, 249.18,
                                              283.12, 298.10])}
    stn.is_fitted_ = True

    # Store the data somewhere
    data_gt_cp = dce_mod.data_.copy()

    # Normalize the data
    dce_mod_norm = stn.normalize(dce_mod)

    # Check if the data are properly normalized
    dce_mod_norm.data_.flags.writeable = False
    data = np.load(os.path.join(currdir, 'data', 'data_normalized_dce.npy'))
    assert_equal(hash(dce_mod_norm.data_.data), data)

    dce_mod_norm.data_.flags.writeable = True

    dce_mod_2 = stn.denormalize(dce_mod_norm)
    dce_mod_2.data_.flags.writeable = False
    assert_equal(hash(dce_mod_2.data_.data), 5894673046233470809)


def test_normalize_denormalize_2():
    """Test the data normalization and denormalization with shift < 0."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)

    # Simulate that we fitted the data
    stn.model_ = np.array([30., 30., 32., 31., 31., 30., 35., 55., 70., 80.])
    stn.is_model_fitted_ = True
    stn.fit_params_ = {'scale-int': 1.2296657327848537,
                       'shift-time': -3.0,
                       'shift-int': np.array([191.29, 193.28, 195.28, 195.28,
                                              195.28, 197.28, 213.25, 249.18,
                                              283.12, 298.10])}
    stn.is_fitted_ = True

    # Store the data somewhere
    data_gt_cp = dce_mod.data_.copy()

    # Normalize the data
    dce_mod_norm = stn.normalize(dce_mod)

    # Check if the data are properly normalized
    dce_mod_norm.data_.flags.writeable = False
    data = np.load(os.path.join(currdir, 'data', 'data_normalized_dce_2.npy'))
    assert_equal(hash(dce_mod_norm.data_.data), data)

    dce_mod_norm.data_.flags.writeable = True

    dce_mod_2 = stn.denormalize(dce_mod_norm)
    dce_mod_2.data_.flags.writeable = False
    assert_equal(hash(dce_mod_2.data_.data), 4280768911343777554)


def test_normalize_denormalize_3():
    """Test the data normalization and denormalization with shift > 0."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)

    # Simulate that we fitted the data
    stn.model_ = np.array([30., 30., 32., 31., 31., 30., 35., 55., 70., 80.])
    stn.is_model_fitted_ = True
    stn.fit_params_ = {'scale-int': 1.2296657327848537,
                       'shift-time': 3.0,
                       'shift-int': np.array([191.29, 193.28, 195.28, 195.28,
                                              195.28, 197.28, 213.25, 249.18,
                                              283.12, 298.10])}
    stn.is_fitted_ = True

    # Store the data somewhere
    data_gt_cp = dce_mod.data_.copy()

    # Normalize the data
    dce_mod_norm = stn.normalize(dce_mod)

    # Check if the data are properly normalized
    dce_mod_norm.data_.flags.writeable = False
    data = np.load(os.path.join(currdir, 'data', 'data_normalized_dce_3.npy'))
    assert_equal(hash(dce_mod_norm.data_.data), data)

    dce_mod_norm.data_.flags.writeable = True

    dce_mod_2 = stn.denormalize(dce_mod_norm)
    dce_mod_2.data_.flags.writeable = False
    assert_equal(hash(dce_mod_2.data_.data), -3781160829709175881)


def test_normalize_no_fitting():
    """Test either if an error is raised when the data have not been fitted
    and the normalization is attempted."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)

    # Attempt the normalization before any fitting
    assert_raises(ValueError, stn.normalize, dce_mod)


def test_denormalize_no_fitting():
    """Test either if an error is raised when the data have not been fitted
    and the denormalization is attempted."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)

    # Attempt the normalization before any fitting
    assert_raises(ValueError, stn.denormalize, dce_mod)


def test_fit_wrong_mod_gt():
    """Test either if an error is raised when the modality provided is not
    the good class."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.partial_fit_model, dce_mod, 1, label_gt[0])


def test_fit_not_read_mod_gt():
    """Test either if an error is raised when the ground-truth was
    not opened."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  gt_mod, label_gt[0])


def test_fit_wrong_gt_size():
    """Test either if an error is raised when the ground-truth have an
    inconsistent size."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  gt_mod, label_gt[0])


def test_fit_not_read_mod():
    """Test either if an error is raised when the modality was
    not opened."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()

    # Read the ground-truth
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.partial_fit_model, dce_mod,
                  gt_mod, label_gt[0])


def test_partial_fit_model_wt_gt_and_cat():
    """Test either if a warning is raised when a gt is not provided
    and a cat is."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_warns(UserWarning, stn.partial_fit_model, dce_mod, cat='prostate')


def test_partial_fit_model_wt_label_gt():
    """Test either if a warning is raised when a gt is not provided
    and a cat is."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    assert_raises(ValueError, stn.partial_fit_model, dce_mod, gt_mod)


def test_partial_fit_without_gt():
    """Test the partial routine without any gt provided."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'full_dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Create the object to make the normalization
    stn = StandardTimeNormalization(dce_mod)
    stn.partial_fit_model(dce_mod)

    # Check the data of the model
    data = np.array([89.90, 90.78, 89.38, 90.45, 91.62, 90.51,
                     93.79, 98.52, 101.79, 103.56])
    assert_array_almost_equal(stn.model_, data, decimal=PRECISION_DECIMAL)
    assert_true(stn.is_model_fitted_)


# This test is commented since that it take more than 3 GB of memory
# It is already tested with Gaussian normalization
# def test_save_load():
#     """Test the save and load routine."""

#     # Load the data with only a single serie
#     currdir = os.path.dirname(os.path.abspath(__file__))
#     path_data = os.path.join(currdir, 'data', 'full_dce')
#     # Create an object to handle the data
#     dce_mod = DCEModality()

#     # Read the data
#     dce_mod.read_data_from_path(path_data)

#     # Load the GT data
#     path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
#     label_gt = ['prostate']
#     gt_mod = GTModality()
#     gt_mod.read_data_from_path(label_gt, path_gt)

#     # Create the object to make the normalization
#     stn = StandardTimeNormalization(dce_mod)

#     # Create a synthetic model to fit on
#     stn.model_ = np.array([30., 30., 32., 31., 31., 30., 35., 55., 70., 80.])
#     stn.is_model_fitted_ = True

#     # Fit the parameters on the model
#     stn.fit(dce_mod, gt_mod, label_gt[0])

#     # Store the normalization object
#     filename = os.path.join(currdir, 'data', 'stn_obj.p')
#     stn.save_to_pickles(filename)

#     # Load the object
#     stn_2 = StandardTimeNormalization.load_from_pickles(filename)

#     # Check that the different variables are the same
#     assert_equal(type(stn_2.base_modality_), type(stn.base_modality_))
#     assert_equal(stn_2.fit_params_['shift-int'], stn.fit_params_['shift-int'])
#     assert_equal(stn_2.fit_params_['shift-time'],
#                  stn.fit_params_['shift-time'])
#     assert_equal(stn_2.fit_params_['scale-int'], stn.fit_params_['scale-int'])
#     assert_equal(stn_2.is_fitted_, stn.is_fitted_)
#     assert_array_equal(stn_2.roi_data_, stn.roi_data_)
