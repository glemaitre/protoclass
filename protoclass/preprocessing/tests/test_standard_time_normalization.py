"""Test the standard time normalization."""

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_raises

from nose.tools import assert_true

from skimage import img_as_float
from scipy.ndimage.filters import gaussian_filter1d

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

PRECISION_DECIMAL = 0


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
        label_gt[0]),  nb_bins=nb_bins)

    # Build the graph by taking the inverse exponential of the heatmap
    heatmap_inv_exp = np.exp(img_as_float(1. - (heatmap / np.max(heatmap))))
    graph = StandardTimeNormalization._build_graph(heatmap_inv_exp, .99)

    start_end_tuple = ((0, 6), (3, 6))
    
    # Call the algorithm to walk through the graph
    path = StandardTimeNormalization._walk_through_graph(graph,
                                                         heatmap_inv_exp,
                                                         start_end_tuple,
                                                         'shortest-path')

    gt_path = np.array([[0, 6],[1, 6],[2, 6],[3, 6]])
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
        label_gt[0]),  nb_bins=nb_bins)

    # Build the graph by taking the inverse exponential of the heatmap
    heatmap_inv_exp = np.exp(img_as_float(1. - (heatmap / np.max(heatmap))))
    graph = StandardTimeNormalization._build_graph(heatmap_inv_exp, .99)

    start_end_tuple = ((0, 131), (3, 135))
    
    # Call the algorithm to walk through the graph
    path = StandardTimeNormalization._walk_through_graph(graph,
                                                         heatmap_inv_exp,
                                                         start_end_tuple,
                                                         'route-through-graph')

    gt_path = np.array([[0, 131],[1, 132],[2, 134],[3, 135]])
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
        label_gt[0]),  nb_bins=nb_bins)

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
        label_gt[0]),  nb_bins=nb_bins)

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


def test_fit_model():
    """Test the routine to fit the parameters."""

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
    stn.fit(dce_mod, gt_mod, label_gt[0])
