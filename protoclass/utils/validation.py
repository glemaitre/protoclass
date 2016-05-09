"""Utilities for input validation."""

import os
import numpy as np

def check_path_data(path_data):
    """Check if the path data exist.

    Parameters
    ----------
    path_data : str or list of str
        Path to the temporal data.

    Returns
    -------
    path_data : str or list of str
        Path to the temporal data.

    """

    # Check if we have a list or a single string object
    if isinstance(path_data, list):
        # We have to check that each item is a basestring and if the directory
        # is existing
        for s in path_data:
            if isinstance(s, basestring):
                # Check that the directory exist
                if not os.path.isdir(s):
                    raise ValueError('The directory specified does not exist.')
            else:
                raise ValueError('One of the item in path data is not of the'
                                 ' correct type str.')
    elif isinstance(path_data, basestring):
        # Check that the directory exist
        if not os.path.isdir(path_data):
            raise ValueError('The directory specified does not exist.')
    else:
        raise ValueError('The input path_data is of unknown type.')

    return path_data


def check_modality(modality, template_modality):
    """Check the modality class is the same than a template modality.

    Parameters
    ----------
    modality : object
        The modality object of interest.

    template_modality : object
        The template modality object of interest.

    Returns
    -------
    None

    """

    # Check that the two modality classes are coherent
    if type(modality) is not type(template_modality):
        raise ValueError('The input modality is different from the template'
                         ' modality given during the construction of the'
                         ' object.')
    else:
        pass

    return None


def check_img_filename(filename):
    """Method to check that the filename is an `img` file.

    Parameters
    ----------
    filename : str
        The filename to check.

    Returns
    -------
        The filename checked.

    """
    # Check that filename is of type basetring
    if isinstance(filename, basestring):
        # Check that filename point to a file
        if os.path.isfile(filename):
            if filename.endswith('.img'):
                return filename
            else:
                raise ValueError('The file needs to be with an img extension.')
        else:
            raise ValueError('The filename provided does not point to a file.')
    else:
        raise ValueError('Wrong type for filename variable.')


def check_npy_filename(filename):
    """Method to check that the filename is an `npy` file.

    Parameters
    ----------
    filename : str
        The filename to check.

    Returns
    -------
        The filename checked.

    """
    # Checkt that filename is of type string
    if isinstance(filename, basestring):
        # Check that filename point to a file
        if os.path.isfile(filename):
            if filename.endswith('.npy'):
                return filename
            else:
                raise ValueError('The file needs to be with an npy extension.')
        else:
            raise ValueError('The filename provided does not point to a file.')
    else:
        raise ValueError('Wrong type for filename variable.')


def check_filename_pickle_load(filename):
    """Method to check if the filename corresponds to a pickle file.

    Parameters
    ----------
    filename : str
        The pickle file to check.

    Returns
    -------
    filename : str
        The checked filename.

    """

    # Check that filename is of string type
    if isinstance(filename, basestring):
        # Check that this is a fit file
        if filename.endswith('.p'):
            # Check that the file is existing
            if os.path.isfile(filename):
                return filename
            else:
                raise ValueError('The file does not exist.')
        else:
            raise ValueError('The file is not an pickle `.p` file.')
    else:
        raise ValueError('The filename needs to be a string.')


def check_filename_pickle_save(filename):
    """Function to check the extension of the pickle file.

    Parameters
    ----------
    filename : str
        The filename which needs to be checked.

    Returns
    -------
    filename : str
        The filename which has been checked.

    """
    # Check that the filename is a string
    if isinstance(filename, basestring):
        # Check the extension
        if filename.endswith('.p'):
            return filename
        else:
            raise ValueError('The filename should have a `.p` extension.')
    else:
        raise ValueError('The filename needs to be of type string.')


def check_modality_inherit(child_modality, parent_modality):
    """Check if a modality inherit from another one.

    Parameters
    ----------
    child_modality : object
        The child object.

    parent_modality_info : classinfo
        The class info of the parent modality.

    Returns
    -------
    child_modality : object
        If no error raise, return the child_object.

    """

    if not issubclass(type(child_modality), parent_modality):
        raise ValueError('The base modality provided in the constructor is'
                         ' not a TemporalModality.')
    else:
        return child_modality


def check_modality_gt(modality, ground_truth, cat):
    """Check the consistency of the modality with the ground-truth.

    Parameters
    ----------
    modality : object
        The modality object of interest.

    ground-truth : object of type GTModality
        The ground-truth of GTModality.

    cat : str
        String corresponding at the ground-truth of interest.

    Returns
    -------
    roi_data : ndarray, shape (non_zero_samples, 3)
        Corresponds to the indexes of the data of insterest
        extracted from the ground-truth.

    """

    from ..data_management import GTModality

    # Check that the ground-truth is from GTModality
    if not isinstance(ground_truth, GTModality):
        raise ValueError('The ground-truth should be an object of'
                         ' class GTModality.')

    # Check that the ground truth has been read
    if not ground_truth.is_read():
        raise ValueError('No data have been read during the'
                         'construction of the GT modality object.')

    # Check that the ground truth has been read
    if not modality.is_read():
        raise ValueError('No data have been read during the'
                         'construction of the modality object.')

    # Check that the size of the ground-truth and the modality
    # are consistant
    # In this case check only the last three dimension
    if ((np.size(modality.data_, -3),
         np.size(modality.data_, -2),
         np.size(modality.data_, -1)) !=
            ground_truth.extract_gt_data(cat, 'data').shape):
        raise ValueError('The ground-truth does not correspond to the'
                         ' given modality volume.')

    # Find the element which are not zero
    return ground_truth.extract_gt_data(cat, 'index')
