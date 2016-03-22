""" T2W modality class.
"""

import numpy as np
import SimpleITK as sitk
import os

from .standalone_modality import StandaloneModality


class T2WModality(StandaloneModality):
    """ Class to handle T2W-MRI modality.

    Parameters
    ----------
    path_data : string
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : string
        Location of the data

    data_ : array-like, shape (Y, X, Z)
        The different volume of the T2W volume. The data are saved in
        Y, X, Z ordered.

    pdf_ : list, length (n_serie)
        List of the PDF for each serie.

    bin_ : list of ndarray, length (n_serie)
        List of the bins used to plot the pdfs.

    max_ : float
        Maximum intensity of the T2W-MRI volume.

    min_ : float
        Minimum intensity of the T2W-MRI volume.
    """

    def __init__(self, path_data):
        super(T2WModality, self).__init__(path_data=path_data)
        self.data_ = None

    def _update_histogram(self):
        """Function to compute histogram of each serie and store it
        The min and max of the series are also stored

        Parameters
        ----------

        Return:
        -------
        self : object
            Returns self.
        """
        # Check if the data have been read
        if self.data_ is None:
            raise ValueError('You need to read the data first. Call the'
                             'function read_data_from_path()')

        # Compute the min and max from the T2W volume
        self.max_ = np.ndarray.max(self.data_)
        self.min_ = np.ndarray.min(self.data_)

        # Build the histogram corresponding to the current volume
        bins = int(np.round(self.max_ - self.min_))
        self.pdf_, self.bin_ = np.histogram(self.data_, bins=bins, density=True)

        return self

    def read_data_from_path(self):
        """Function to read T2W images which correspond to a 3D volume.

        Parameters
        ----------
        path_data : str
            Path to the DCE data.

        Return
        ------
        self : object
           Returns self.
        """
        # Called the parent function to read the data
        super(T2WModality, self).read_data_from_path()

        # Compute the information regarding the T2W images
        self._update_histogram()

        return self
