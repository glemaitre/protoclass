"""Utilities for input validation"""

import os


def check_path_data(path_data):
    """ Method to check if the path data exist.

    Parameters
    ----------
    path_data : str or list of str
        Path to the temporal data.

    Return
    ------
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
    """ Method to check the modality class is the same than a template
    modality.

    Parameters
    ----------
    modality : object
        The modality object of interest.

    template_modality : object
        The template modality object of interest.

    Return
    ------
    None
    """

    # Check that the two modality classes are coherent
    if type(modality) is not type(template_modality):
        raise ValueError('The input modality is different from the template'
                         ' modality given during the construction of the'
                         ' object.')
    else:
        pass


def check_modality_gt(modality, ground_truth):
    """ Method to check the consistency of the modality with the ground-truth. """
