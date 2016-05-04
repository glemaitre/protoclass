"""Additional utilities for ndarray used repetetively."""

import numpy as np


def find_nearest(array, value):
    """Find the nearest value and index inside an ndarray.

    Parameters
    ----------
    array : ndarray
        Array in which the value has to be search.

    value : array.dtype
        Targetted value.

    Returns
    -------
    nn_value : array.dtype
        The nearest value found in the array.

    idx : int
        The associated index of `nn_value`

    """
    idx = (np.abs(array-value)).argmin()

    return array[idx], idx
