"""T2W modality class."""

import warnings

import numpy as np
import SimpleITK as sitk

from .standalone_modality import StandaloneModality

from ..utils.validation import check_path_data


class T2WModality(StandaloneModality):
    """Class to handle T2W-MRI modality.

    Parameters
    ----------
    path_data : str, optional (default=None)
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : string
        Location of the data.

    data_ : ndarray, shape (Y, X, Z)
        The different volume of the T2W volume. The data are saved in
        Y, X, Z ordered.

    metadata_ : dict
        Dictionnary which contain the MRI sequence information. Note that the
        information are given in the original ordering (X, Y, Z), which is
        different from the organisation of `data_` which is (Y, X, Z).

    pdf_ : list, length (n_serie)
        List of the PDF for each serie.

    bin_ : list of ndarray, length (n_serie)
        List of the bins used to plot the pdfs.

    max_ : float
        Maximum intensity of the T2W-MRI volume.

    min_ : float
        Minimum intensity of the T2W-MRI volume.

    """

    def __init__(self, path_data=None):
        super(T2WModality, self).__init__(path_data=path_data)

    def get_pdf(self, roi_data=None, nb_bins='auto'):
        """ Extract the a list of pdf related with the data.

        Parameters
        ----------
        roi_data : tuple
            Indices of elements to consider while computing the histogram.
            The ROI is a 3D volume which will be used for each time serie.

        nb_bins : list of int or str, optional (default='auto')
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If 'auto', the number of bins is found at fitting time.
            - If None, the number of bins used is the one at the last
            call of update histogram.
            - Otherwise, a list of integer needs to be given.

        Returns
        -------
        pdf_data : ndarray, length (n_serie)
            List of the pdf with the associated series.

        bin_data : list of ndarray, length (n_series + 1)
            List of the bins associated with the list of pdf.

        """
        # Check that the data have been read
        if self.data_ is None:
            raise ValueError('You need to load the data first. Refer to the'
                             ' function read_data_from_path().')

        # Build the histogram corresponding to the current volume
        # Find how many bins do we need
        if isinstance(nb_bins, basestring):
            if nb_bins == 'auto':
                nb_bins = int(np.round(self.max_ - self.min_))
            else:
                raise ValueError('Unknown parameters for `nb_bins.`')
        elif isinstance(nb_bins, int):
            pass
        elif nb_bins is None:
            nb_bins = self.nb_bins_
        else:
            raise ValueError('Unknown type for the parameters `nb_bins`.')

        if roi_data is None:
            pdf_data, bin_data = np.histogram(self.data_,
                                              bins=nb_bins,
                                              density=True)
        else:
            pdf_data, bin_data = np.histogram(self.data_[roi_data],
                                              bins=nb_bins,
                                              density=True)

        return pdf_data, bin_data

    def update_histogram(self, nb_bins=None):
        """Update the PDF and the first-order statistics.

        Parameters
        ----------
        nb_bins : int or None, optional (default=None)
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If None, the number of bins found at reading will be used.
            - If 'auto', the number of bins is found at fitting time.
            - Otherwise, an integer needs to be given.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        There is the possibility to redifine the number of bins to use for
        the histogram since it can be tricky to play with normalized data.

        """
        # Check if the data have been read
        if self.data_ is None:
            raise ValueError('You need to read the data first. Call the'
                             ' function read_data_from_path()')

        # Compute the min and max from the T2W volume
        self.max_ = np.ndarray.max(self.data_)
        self.min_ = np.ndarray.min(self.data_)

        # Build the histogram corresponding to the current volume
        # Find how many bins do we need
        if isinstance(nb_bins, basestring):
            if nb_bins == 'auto':
                nb_bins = int(np.round(self.max_ - self.min_))
            else:
                raise ValueError('Unknown parameters for `nb_bins.`')
        elif nb_bins is None:
            nb_bins = self.nb_bins_

        self.pdf_, self.bin_ = np.histogram(self.data_,
                                            bins=nb_bins,
                                            density=True)

        return self

    def read_data_from_path(self, path_data=None):
        """Read T2W images which represent a single 3D volume.

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

        # Compute the information regarding the T2W images
        # Set the number of bins that will be later used to compute
        # the histogram
        self.nb_bins_ = int(np.round(np.ndarray.max(self.data_) -
                                     np.ndarray.min(self.data_)))
        self.update_histogram()

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
        im = sitk.ReadImage(dicom_names_serie[0])
        self.metadata_['TR'] = float(im.GetMetaData('0018|0080'))
        self.metadata_['TE'] = float(im.GetMetaData('0018|0081'))
        self.metadata_['flip-angle'] = float(im.GetMetaData('0018|1314'))

        return self
