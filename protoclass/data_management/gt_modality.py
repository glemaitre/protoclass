"""GT modality class.
"""

import numpy as np
import SimpleITK as sitk
import warnings

from .multisequence_modality import MultisequenceModality
from ..utils.validation import check_path_data


class GTModality(MultisequenceModality):
    """Class to handle GT modality.

    Parameters
    ----------
    path_data : str, list of str, or None, optional (default=None)
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : str or list of str
        Location of the data.

    data_ : ndarray, shape (GT, Y, X, Z)
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

    def update_histogram(self):
        """Method to compute histogram and statistics."""
        raise NotImplementedError

    def read_data_from_path(self, cat_gt, path_data=None):
        """Read GT images which correspond to a 3D volume, a volume
        for different information.

        Parameters
        ----------
        cat_gt : list of str
            Categorical label affected to each ground-truth volume read.

        path_data : str, list or None, optional (default=None)
            Path to the temporal data. It will overrides the path given
            in the constructor.

        Returns
        -------
        self : object
           Returns self.

        Notes
        -----
        This function overwrite the function of MultisequenceModality since
        that a GT can contain only one sequence as well.

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

            # The IDs need to be re-ordered in an incremental manner
            # Create a list by converting to integer the number after
            # the last full stop
            id_series_seq_int = np.array([int(s[s.rfind('.')+1:])
                                          for s in series_seq])
            # Sort and get the corresponding index
            idx_series_sorted = np.argsort(id_series_seq_int)

            # Open the volume in the sorted order
            list_volume = []
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

            # We can create a numpy array
            self.data_ = np.array(list_volume)
            self.n_serie_ = self.data_.shape[0]

        # Case that we have a list of string
        else:
            # We have to iterate through each folder and check that we have
            # only one serie
            # Create a reader object

            list_volume = []
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

            # We can create a numpy array
            self.data_ = np.array(list_volume)
            self.n_serie_ = self.data_.shape[0]

        # Check that we have the same number of category than the number
        # of serie
        if len(cat_gt) != self.n_serie_:
            raise ValueError('The number of categorie should be the same than'
                             ' the number of ground-truth volume read.')
        else:
            self.cat_gt_ = cat_gt

        return self

    def extract_gt_data(self, label, output_type='index'):
        """Extract data corresponding to a given label.

        Parameters
        ----------
        label : str
            Label of the ground-truth to extract.

        output_type : str, optional (default='index')
            Type of the desired output. The choices are.

            - If 'index', the index corresponding to the positive class will
            be returned.
            - If 'data', the full ground-truth data will be returned.

        Returns
        -------
        output : ndarray
            The output data. Depends of the arguments `output_type`

        """
        # Check that data have been read
        if not self.is_read():
            raise ValueError('Read the data before to extract them.')

        # Check that the label was part of the category given at opening
        if not any([label == x for x in self.cat_gt_]):
            raise ValueError('The provided label was not part of the category'
                             ' at the opening time.')

        # Get the index corresponding to the ground-truth
        idx_label = self.cat_gt_.index(label)

        # Return the data corresponding to the ground-truth
        if output_type == 'index':
            return np.nonzero(self.data_[idx_label, :, :, :])
        elif output_type == 'data':
            return self.data_[idx_label, :, :, :]
        else:
            raise ValueError('Invalid output descriptor.')
