"""Test the enhancement signal extraction."""

import os

from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from protoclass.extraction import EnhancementSignalExtraction

from protoclass.data_management import DCEModality
from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality


def test_ese_bad_mod():
    """Test either if an error is raised when the base modality does not
    inherate from TemporalModality."""

    # Try to create the normalization object with the wrong class object
    assert_raises(ValueError, EnhancementSignalExtraction, T2WModality())


def test_ese_bad_mod_fit():
    """Test either if an error is raised when a modality to fit does not
    correspond to the template modality given at the construction."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)
    # Fit and raise the error
    assert_raises(ValueError, dce_ese.fit, t2w_mod)


def test_ese_not_read_mod_fit():
    """Test either if an error is raised when the modality has not been
    read before fitting."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Fit and raise the error
    assert_raises(ValueError, dce_ese.fit, dce_mod)


def test_ese_fit():
    """Test either if an error is raised since that the function
    is not implemented."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Open the DCE data
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)
    # Fit and raise the error
    assert_raises(NotImplementedError, dce_ese.fit, dce_mod)


def test_ese_bad_mod_transform():
    """Test either if an error is raised when a modality to tranform does not
    correspond to the template modality given at the construction."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 't2w')
    # Create an object to handle the data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(path_data)
    # Fit and raise the error
    assert_raises(ValueError, dce_ese.transform, t2w_mod)


def test_ese_transform_gt_cat():
    """Test the transform routine with a given ground-truth."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Fit and raise the error
    data = dce_ese.transform(dce_mod, gt_mod, label_gt[0])

    # Check the size of the data
    assert_equal(data.shape, (12899, 4))
    # Check the hash of the data
    data.flags.writeable = False
    assert_equal(hash(data.data), -3808597525488161265)


def test_ese_wrong_gt_mod():
    """Test either if an error is raised when a wrong modality is given
    as ground-truth."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Fit and raise the error
    assert_raises(ValueError, dce_ese.transform, dce_mod, dce_mod, 'prostate')


def test_ese_wrong_gt_size():
    """Test if an error is raised when the size of gt is not consistent
    with the modality."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'full_gt', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Fit and raise the error
    assert_raises(ValueError, dce_ese.transform, dce_mod, gt_mod, label_gt[0])


def test_ese_gt_not_read():
    """Test if an error is raised when the GT is not read."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()

    # Fit and raise the error
    assert_raises(ValueError, dce_ese.transform, dce_mod, gt_mod, label_gt[0])


def test_ese_transform_wt_gt_and_cat():
    """Test either if a warning is raised when a gt is not provided
    and a cat is."""

    # Load the data with only a single serie
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Read the data
    dce_mod.read_data_from_path(path_data)

    # Create the object to make the normalization
    dce_ese = EnhancementSignalExtraction(dce_mod)
    assert_warns(UserWarning, dce_ese.transform, dce_mod, cat='prostate')


def test_ese_transform_gt_no_cat():
    """Test eihter if an error is raised when no category for GT is
    provided."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_data)

    # Load the GT data
    path_gt = [os.path.join(currdir, 'data', 'gt_folders', 'prostate')]
    label_gt = ['prostate']
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, path_gt)

    # Fit and raise the error
    assert_raises(ValueError, dce_ese.transform, dce_mod, gt_mod)


def test_ese_not_read_mod_transform():
    """Test either if an error is raised when the modality has not been
    read before transforming the data."""

    # Create the normalization object with the right modality
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Try to fit an object with another modality
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(currdir, 'data', 'dce')
    # Create an object to handle the data
    dce_mod = DCEModality()

    # Fit and raise the error
    assert_raises(ValueError, dce_ese.transform, dce_mod)
