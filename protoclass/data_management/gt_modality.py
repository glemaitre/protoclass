""" GT modality class.
"""

import numpy as np
import SimpleITK as sitk
import os

from .multisequence_modality import MultisequenceModality


class GTModality(MultisequenceModality):
    """ Class to handle GT modality.

    Parameters
    ----------
    path_data : str, list of str, or None, optional (default=None)
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : str or list of str
        Location of the data.

    data_ : array-like, shape (GT, Y, X, Z)
        The different volume of the GT volume. The data are saved in
        GT, Y, X, Z ordered.

    n_serie_ : int
        The number of ground-truth volume. Refer to cat_gt_ to know the
        information associated to each volume.

    cat_gt_ : list of str
        Categorical labels associated with each ground-truth volume read.
    """

    def __init__(self, path_data=None):
        super(GTModality, self).__init__(path_data=path_data)
        self.data_ = None

    def _update_histogram(self):
        """ Method to compute histogram and statistics. """
        raise NotImplementedError

    def read_data_from_path(self, cat_gt, path_data=None):
        """Function to read GT images which correspond to a 3D volume,
        a volume for different information.

        Parameters
        ----------
        cat_gt : list of str
            Categorical label affected to each ground-truth volume read.

        path_data : str, list or None, optional (default=None)
            Path to the temporal data. It will overrides the path given
            in the constructor.

        Return
        ------
        self : object
           Returns self.
        """
        # Called the parent function to read the data
        super(GTModality, self).read_data_from_path(path_data=path_data)

        # Check that we have the same number of category than the number
        # of serie
        if len(cat_gt) != self.n_serie_:
            raise ValueError('The number of categorie should be the same than'
                             ' the number of round-truth volume read.')
        else:
            self.cat_gt_ = cat_gt

        return self
