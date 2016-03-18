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

    data_ : array-like, shape (T, Y, X, Z)
            The different volume of the DCE serie. The data are saved in
            T, Y, X, Z ordered.
    """

    def __init__(self,
                 path_data):
        super(DCEModality, self).__init__(
            path_data=path_data)
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

        # Compute the min and max from all DCE series
        self.max_series_ = np.ndarray.max(self.data_)
        self.min_series_ = np.ndarray.min(self.data_)

        # For each serie compute the pdfs and store them
        pdf_series = []
        bin_series = []

        for data_serie in self.data_:
            bins = int(np.round(np.ndarray.max(data_serie) -
                                np.ndarray.min(data_serie)))

            pdf_s, bin_s = np.histogram(data_serie,
                                        bins=bins,
                                        density=True)
            pdf_series.append(pdf_s)
            bin_series.append(bin_s)

        # Keep these data in the object
        self.pdf_series_ = pdf_series
        self.bin_series_ = bin_series

        return self

    def read_data_from_path(self):
        """Function to read DCE images which is of 3D volume over time.

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
        # The first dimension correspond to the time dimension
        # When processing the data, we need to slice the data
        # considering this dimension emphasizing the decision to let
        # it at the first position.
        self.data_ = np.array(list_volume)

        # Compute the information regarding the pdf of the DCE series
        self._update_histogram()

        return self
