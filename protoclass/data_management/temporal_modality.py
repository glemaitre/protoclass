""" Base class for temporal modality (DCE).
"""

import numpy as np
import SimpleITK as sitk
import os

from abc import ABCMeta, abstractmethod

from .base_modality import BaseModality


class TemporalModality(BaseModality):
    """ Basic class for temporal medical modality (DCE).

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data):
        super(TemporalModality, self).__init__(path_data=path_data)

    @abstractmethod
    def _update_histogram(self):
        """ Method to compute histogram and statistics. """
        raise NotImplementedError

    @abstractmethod
    def read_data_from_path(self):
        """Function to read temporal images which represent a 3D volume
        over time.

        Parameters
        ----------

        path_data : str
            Path to the DCE data.

        Return
        ------
        self : object
           Returns self.
        """
        # Check that the directory exist
        if os.path.isdir(self.path_data_) is not True:
            raise ValueError('The directory specified does not exist.')

        # Create a reader object
        reader = sitk.ImageSeriesReader()

        # Find the different series present inside the folder
        series_time = np.array(reader.GetGDCMSeriesIDs(self.path_data_))

        # Check that you have more than one serie
        if len(series_time) < 2:
            raise ValueError('The time serie should at least contain'
                             '2 series.')

        # The IDs need to be re-ordered in an incremental manner
        # Create a list by converting to integer the number after
        # the last full stop
        id_series_time_int = np.array([int(s[s.rfind('.')+1:])
                                      for s in series_time])
        # Sort and get the corresponding index
        idx_series_sorted = np.argsort(id_series_time_int)

        # Open the volume in the sorted order
        list_volume = []
        for id_time in series_time[idx_series_sorted]:
            # Get the filenames corresponding to the current ID
            dicom_names_serie = reader.GetGDCMSeriesFileNames(self.path_data_,
                                                              id_time)
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

            # Concatenate the different volume
            list_volume.append(vol_numpy)

        # We can create a numpy array
        # The first dimension corresponds to the time dimension
        # When processing the data, we need to slice the data
        # considering this dimension emphasizing the decision to let
        # it at the first position.
        self.data_ = np.array(list_volume)
        self.n_serie_ = self.data_.shape[0]

        return self
