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
