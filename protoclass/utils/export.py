"""Utilities to export the classification results"""

import numpy as np


def make_table(data, feature_list, filename, decimals=1):
    """Function to build a latex table from a numpy matrix.

    Parameters
    ----------
    data : ndarray, shape (M, N)
        The data to save into a latex array. M corresponds to the
        feature category.

    feature_list: list of str, length (M,)
        The name of the features in data.

    filename: str
        Name of the file where to save the table.
        The format supported is latex ('.tex')

    decimal: int
        The precision to report the results.

    Returns
    -------
    None
    """
    # Check the extension of filename
    if not filename.endswith('.tex'):
        raise ValueError('The format to export is unknown.')

    # Check that we have the dimension of data and feature list are coherent
    if len(feature_list) != data.shape[0]:
        raise ValueError('The number of of feature in the list should be'
                         ' identical to the number of raw of data')

    # Check that there is no negative value inside data or too large value
    if np.min(data) < 0. or np.max(data) > 100.:
        raise ValueError('Some values in data are negative or too large.')

    # Check if the value in data are between 0 and 1 or between 0 and 100
    if np.max(data) < 1.:
        # Rescale the data between 0 and 100
        data = np.around(data * 100., decimals=decimals)
    else:
        # Round only
        data = np.around(data, decimals=decimals)

    # Open the file if possible
    fi = open(filename, 'w+')
    # Get the raw of data and the feature category in the same time
    for feat, r_data in zip(feature_list, data):
        line = feat
        for c_data in r_data:
            line += ' & ' + str(c_data)
        line += ' \\' + '\\ ' + '\n'
        fi.write(line)
    fi.close()

    return None
