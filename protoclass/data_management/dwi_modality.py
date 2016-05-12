"""DWI modality class."""

import warnings

import numpy as np
import SimpleITK as sitk

from .multisequence_modality import MultisequenceModality

from ..utils.validation import check_path_data


class DWIModality(MultisequenceModality):
    """Class to handle DWI modality.

    Parameters
    ----------
    path_data : str, list of str, or None, optional (default=None)
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : str or list of str
        Location of the data.

    data_ : ndarray, shape (B_seq, Y, X, Z)
        The different volume from different B sequuences. The data are
        saved in B_seq, Y, X, Z ordered.

    pdf_series_ : list of ndarray, length (n_serie)
        List of the PDF for each serie.

    bin_series_ : list of ndarray, length (n_serie)
        List of the bins used to plot the pdfs.

    max_series_ : float
        Maximum intensity of all the DCE series.

    min_series_ : float
        Minimum intensity of all the DCE series.

    n_serie_ : int
        Number of serie in this DCE sequence.

    max_series_list_ : list of float
        List of the maximum intensity for each DCE serie.

    min_series_list_ : list of float
        List of the minimum intensity for each DCE serie.

    """

    def __init__(self, path_data=None):
        super(DWIModality, self).__init__(path_data=path_data)

    def update_histogram(self, nb_bins=None):
        """Update the histogram of each serie and first-order statistics.

        Parameters
        ----------
        nb_bins : list of int, str, or None, optional (default=None)
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If None, the number of bins found at reading will be used.
            - If 'auto', the number of bins is found at fitting time.
            - Otherwise, a list of integer needs to be given.

        Returns
        -------
        self : object
            Returns self.

        """
        # Check if the data have been read
        if self.data_ is None:
            raise ValueError('You need to read the data first. Call the'
                             ' function read_data_from_path()')

        # Compute the min and max from all DCE series
        self.max_series_ = np.ndarray.max(self.data_)
        self.min_series_ = np.ndarray.min(self.data_)

        # For each serie compute the pdfs and store them
        pdf_series = []
        bin_series = []
        min_series_list = []
        max_series_list = []

        # Check that we have a proper list of bins
        if isinstance(nb_bins, basestring):
            if nb_bins == 'auto':
                nb_bins = []
                for data_serie in self.data_:
                    nb_bins.append(int(np.round(np.ndarray.max(data_serie) -
                                                np.ndarray.min(data_serie))))
            else:
                raise ValueError('Unknown arguments for `nb_bins`.')
        elif isinstance(nb_bins, list):
            if (len(nb_bins) != len(self.data_) or
                    not all(isinstance(x, int) for x in nb_bins)):
                raise ValueError('Provide a list of integer of bins'
                                 ' with the same size as the number'
                                 ' of serie in the data.')
        elif nb_bins is None:
            nb_bins = self.nb_bins_
        else:
            raise ValueError('Unknown arguments for `nb_bins`.')

        for data_serie, bins in zip(self.data_, nb_bins):
            pdf_s, bin_s = np.histogram(data_serie,
                                        bins=bins,
                                        density=True)
            pdf_series.append(pdf_s)
            bin_series.append(bin_s)
            min_series_list.append(np.ndarray.min(data_serie))
            max_series_list.append(np.ndarray.max(data_serie))

        # Keep these data in the object
        self.pdf_series_ = pdf_series
        self.bin_series_ = bin_series
        self.min_series_list_ = min_series_list
        self.max_series_list_ = max_series_list

        return self

    def _build_dict_metadata(self, vol, filename, refit=False):
        """Build the dictionary from the dicom header.

        Parameters
        ----------
        vol : SimpleITK.Image
            A SimpleITK image from which we can get some metadata information.

        filename : string
            The filename to use to open a single image to get additional
        information from the DICOM header.

        Returns
        -------
        refit : bool
            Update the refit variable

        """
        if not refit:
            # Store the DICOM metadata
            self.metadata_ = {}
            # Get the information that have been created by SimpleITK
            # Information about data reconstruction
            self.metadata_['size'] = vol.GetSize()
            self.metadata_['origin'] = vol.GetOrigin()
            self.metadata_['direction'] = vol.GetDirection()
            self.metadata_['spacing'] = vol.GetSpacing()
            # Information about the MRI sequence
            # Read the first image for the sequence
            im = sitk.ReadImage(filename)
            self.metadata_['TR'] = float(im.GetMetaData('0018|0080'))
            self.metadata_['TE'] = float(im.GetMetaData('0018|0081'))
            self.metadata_['flip-angle'] = float(
                im.GetMetaData('0018|1314'))
            # If the data are provided by a GE scanner
            if im.GetMetaData('0008|0070') == 'GE MEDICAL SYSTEMS':
                try:
                    self.metadata_['b-value'] = [im.GetMetaData('0043|0081')]
                except RuntimeError:
                    warnings.warn('Impossible to get the b-values for'
                                  ' this sequence.')
            refit = True
        else:
            im = sitk.ReadImage(filename)
            if im.GetMetaData('0008|0070') == 'GE MEDICAL SYSTEMS':
                try:
                    self.metadata_['b-value'] = [im.GetMetaData('0043|0081')]
                except RuntimeError:
                    warnings.warn('Impossible to get the b-values for'
                                  ' this sequence.')

        return refit

    def read_data_from_path(self, path_data=None):
        """Function to read DWI images which represent a 3D volume
        over time.

        Parameters
        ----------
        path_data : str, list of str, or None, optional (default=None)
            Path to the DWI data. It will overrides the path given
            in the constructor.

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

        # There is two possibilities to open the data. If path_data is a list,
        # each path will contain only one serie and we can open the data as
        # in standalone. If path_data is a single string, the folder contain
        # several series and we can go through each of them.

        # Case that we have a single string
        if isinstance(self.path_data_, basestring):
            # Create a reader object
            reader = sitk.ImageSeriesReader()

            # Find the different series present inside the folder
            series_seq = np.array(reader.GetGDCMSeriesIDs(self.path_data_))

            # Check that you have more than one serie
            if len(series_seq) < 2:
                raise ValueError('The multisequence should at least contain'
                                 ' 2 sequences.')

            # The IDs need to be re-ordered in an incremental manner
            # Create a list by converting to integer the number after
            # the last full stop
            id_series_seq_int = np.array([int(s[s.rfind('.')+1:])
                                          for s in series_seq])
            # Sort and get the corresponding index
            idx_series_sorted = np.argsort(id_series_seq_int)

            # Open the volume in the sorted order
            list_volume = []
            is_dict_built = False
            for id_seq in series_seq[idx_series_sorted]:
                # Get the filenames corresponding to the current ID
                dicom_names_serie = reader.GetGDCMSeriesFileNames(self.path_data_,
                                                                  id_seq)
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

                # Build the dictionary
                is_dict_built = self._build_dict_metadata(vol,
                                                          dicom_names_serie[0],
                                                          is_dict_built)

            # We can create a numpy array
            self.data_ = np.array(list_volume)
            self.n_serie_ = self.data_.shape[0]

        # Case that we have a list of string
        else:
            # We have to iterate through each folder and check that we have
            # only one serie
            # Create a reader object

            # Check that you have more than one serie
            if len(self.path_data_) < 2:
                raise ValueError('The multisequence should at least contain'
                                 ' 2 sequences.')

            list_volume = []
            is_dict_built = False
            for path_serie in self.path_data_:

                reader = sitk.ImageSeriesReader()

                # Find the different series present inside the folder
                series = np.array(reader.GetGDCMSeriesIDs(path_serie))

                # Check that you have more than one serie
                if len(series) > 1:
                    raise ValueError('The number of series should not be'
                                     ' larger than 1 when a list of path is'
                                     ' given.')

                # The data can be read
                dicom_names_serie = reader.GetGDCMSeriesFileNames(path_serie)
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

                # Append inside the volume list
                list_volume.append(vol_numpy)

                # Build the dictionary
                is_dict_built = self._build_dict_metadata(vol,
                                                          dicom_names_serie[0],
                                                          is_dict_built)

            # We can create a numpy array
            self.data_ = np.array(list_volume)
            self.n_serie_ = self.data_.shape[0]

        # Create the list of number of bins to compute the histogram
        self.nb_bins_ = []
        for data_serie in self.data_:
            self.nb_bins_.append(int(np.round(np.ndarray.max(data_serie) -
                                              np.ndarray.min(data_serie))))

        # Compute the information regarding the pdf of the DCE series
        self.update_histogram()

        return self
