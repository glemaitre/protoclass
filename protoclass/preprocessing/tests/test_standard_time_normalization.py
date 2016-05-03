"""Test the standard time normalization."""

import numpy as np
import os

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from testfixtures import Comparison

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality


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

    # Build the graph
    graph = StandardTimeNormalization._build_graph(heatmap, .99)

    data = np.load(os.path.join(currdir, 'data', 'graph.npy'))
    assert_array_equal(graph.todense(), data)
