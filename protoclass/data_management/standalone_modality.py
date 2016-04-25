"""Basic class for standalone modality (T1, T2, etc.)."""

import numpy as np
import SimpleITK as sitk
import warnings

from abc import ABCMeta, abstractmethod

from .base_modality import BaseModality
from ..utils.validation import check_path_data


class StandaloneModality(BaseModality):
    """Basic class for medical modality (T1, T2, etc.).

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data=None):
        """Constructor."""
        super(StandaloneModality, self).__init__(path_data=path_data)

    @abstractmethod
    def update_histogram(self):
        """Method to compute histogram and statistics."""
        raise NotImplementedError

    @abstractmethod
    def read_data_from_path(self, path_data=None):
        """Read standalone images which represent a single 3D volume.

        Parameters
        ----------
        path_data : str or None, optional (default=None)
            Path to the standalone modality data.

        Returns
        -------
        self : object
           Returns self.

        """
        # Check the consistency of the path data
        if self.path_data_ is not None and path_data is not None:
            # We will overide the path and raise a warning
            warnings.warn('The data path will be overriden using the path'
                          ' given in the function.')
            self.path_data_ = check_path_data(path_data)
        elif self.path_data_ is None and path_data is not None:
            self.path_data_ = check_path_data(path_data)
        elif self.path_data_ is None and path_data is None:
            raise ValueError('You need to give a path_data from where to read'
                             ' the data.')
        # Create a reader object
        reader = sitk.ImageSeriesReader()

        # Find the different series present inside the folder
        series = np.array(reader.GetGDCMSeriesIDs(self.path_data_))

        # Check that you have more than one serie
        if len(series) > 1:
            raise ValueError('The number of series should not be larger than'
                             ' 1 with standalone modality.')

        # The data can be read
        dicom_names_serie = reader.GetGDCMSeriesFileNames(self.path_data_)
        # Set the list of files to read the volume
        reader.SetFileNames(dicom_names_serie)

        # Read the data for the current volume
        vol = reader.Execute()

        # Get a numpy volume
        vol_numpy = sitk.GetArrayFromImage(vol)

        # The Matlab convention is (Y, X, Z)
        # The Numpy convention is (Z, Y, X)
        # We have to swap these axis
        # Swap Z and X
        vol_numpy = np.swapaxes(vol_numpy, 0, 2)
        vol_numpy = np.swapaxes(vol_numpy, 0, 1)

        # Convert the volume to float
        vol_numpy = vol_numpy.astype(np.float64)

        # We can create a numpy array
        self.data_ = vol_numpy

        return self
