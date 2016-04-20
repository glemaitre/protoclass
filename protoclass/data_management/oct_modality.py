""" OCT modality class.
"""

import warnings
import numpy as np

from skimage import img_as_float

from .standalone_modality import StandaloneModality
from ..utils.validation import check_img_filename


class OCTModality(StandaloneModality):
    """ Class to handle OCT modality.

    Parameters
    ----------
    path_data : str, optional (default=None)
         The `img` file where the data are stored.

    Attributes
    ----------
    path_data_ : string
        Location of the `img` file.

    data_ : array-like, shape (Y, X, Z)
        The different volume of the SD-OCT volume. The data are saved in
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

    def __init__(self, path_data=None):
        super(OCTModality, self).__init__(path_data=path_data)

    def _update_histogram(self):
        """Function to compute histogram of each serie and store it
        The min and max of the series are also stored

        Returns
        -------
        self : object
            Returns self.
        """
        # Check if the data have been read
        if self.data_ is None:
            raise ValueError('You need to read the data first. Call the'
                             ' function read_data_from_path()')

        # Compute the min and max from the T2W volume
        self.max_ = np.ndarray.max(self.data_)
        self.min_ = np.ndarray.min(self.data_)

        # Build the histogram corresponding to the current volume
        bins = int(np.round(self.max_ - self.min_))
        self.pdf_, self.bin_ = np.histogram(self.data_,
                                            bins=bins,
                                            density=True)

        return self

    def read_data_from_path(self, sz_data, path_data=None, dtype='uint8'):
        """Function to read T2W images which correspond to a 3D volume.

        Parameters
        ----------
        sz_data : tuple of int, shape (X, Y, Z)
            Tuple with 3 values specifying the dimension of the volume.

        path_data : str or None, optional (default=None)
            Path to the `img` file. It will overrides the path given
            in the constructor.

        dtype : str, optional (default='uint8')
            Type of the raw data.

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
            self.path_data_ = check_img_filename(path_data)
        elif self.path_data_ is None and path_data is not None:
            self.path_data_ = check_img_filename(path_data)
        elif self.path_data_ is None and path_data is None:
            raise ValueError('You need to give a path_data from where to read'
                             ' the data.')

        

        # Compute the information regarding the T2W images
        self._update_histogram()

        return self
