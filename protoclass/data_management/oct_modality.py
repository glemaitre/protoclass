"""OCT modality class."""

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

    data_ : array-like, shape (Y, Z, X)
        The different volume of the SD-OCT volume. The data are saved in
        Y, Z, X ordered. Generally, the data are coded on 8 bits.

    pdf_ : list, length (n_serie)
        List of the PDF for each serie.

    bin_ : list of ndarray, length (n_serie)
        List of the bins used to plot the pdfs.

    max_ : float
        Maximum intensity of the OCT volume.

    min_ : float
        Minimum intensity of the OCT volume.

    """

    def __init__(self, path_data=None):
        if path_data is not None:
            if path_data.endswith('.img'):
                self.path_data_ = check_img_filename(path_data)
            else:
                super(OCTModality, self).__init__(path_data=path_data)
        else:
            self.path_data_ = None
        self.data_ = None

    def update_histogram(self, nb_bins=None):
        """Update the PDF and the first-order statistics.

        Parameters
        ----------
        nb_bins : int or None, optional (default=None)
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If None, the number of bins found at reading will be used.
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
        if nb_bins is None:
            nb_bins = self.nb_bins_
        elif isinstance(nb_bins, int):
            pass
        else:
            raise ValueError('Unknown type for parameters nb_bins.')


        # Build the histogram corresponding to the current volume
        self.pdf_, self.bin_ = np.histogram(self.data_,
                                            bins=nb_bins,
                                            density=True)

        return self

    def read_data_from_path(self, sz_data, path_data=None, dtype='uint8'):
        """Function to read OCT images which correspond to a 3D volume.

        Parameters
        ----------
        sz_data : tuple of int, shape (Y, Z, X)
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

        # Data are stored as (Y, Z, X)
        vol_oct = np.fromfile(self.path_data_, 
                              dtype=dtype, sep="").reshape(sz_data)
        vol_oct = img_as_float(vol_oct)
        # However there is a need to flip up down along Z
        self.data_ = np.zeros(vol_oct.shape)
        for idx_z, im_oct in enumerate(vol_oct):
            self.data_[idx_z] = np.flipud(im_oct)    
        
        # Compute the information regarding the OCT images
        if dtype == 'uint8':
            self.nb_bins_ = 256

        # Compute the information regarding the OCT images
        self.update_histogram()

        return self
