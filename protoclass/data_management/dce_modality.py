""" DCE modality class.
"""

import numpy as np
import SimpleITK as sitk
import os

from .base_modality import BaseModality


class DCEModality(BaseModality):
    """ Class to handle DCE-MRI modality.

    Parameters
    ----------
    path_data : string
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : string
        Location of the data

    data_ : array-like, shape (Y, X, Z, T)
            The different volume of the DCE serie. The data are saved in
            Y, X, Z, T ordered.
    """

    def __init__(self,
                 path_data):
        super(DCEModality, self).__init__(
            path_data=path_data)

    def read_data_from_path(self):
        """
        Function to read DCE images which is of 3D volume over time.

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
        series_dce = np.array(reader.GetGDCMSeriesIDs(self.path_data_))

        # Check that you have more than one serie
        if len(series_dce) < 2:
            raise ValueError('The DCE serie should at least contain 2 series.')

        # The IDs need to be re-ordered in an incremental manner
        # Create a list by converting to integer the number after
        # the last full stop
        id_series_dce_int = np.array([int(s[s.rfind('.')+1:])
                                      for s in series_dce])
        # Sort and get the corresponding index
        idx_series_sorted = np.argsort(id_series_dce_int)

        # Open the volume in the sorted order
        list_volume = []
        for id_dce in series_dce[idx_series_sorted]:
            # Get the filenames corresponding to the current ID
            dicom_names_serie = reader.GetGDCMSeriesFileNames(self.path_data_,
                                                              id_dce)
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
        # Additionaly, we need to roll the first dimension to the
        # last one since this is the index of each DCE serie
        self.data_ = np.rollaxis(np.array(list_volume), 0, 4)
