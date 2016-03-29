""" DWI modality class.
"""

import numpy as np

from .multisequence_modality import MultisequenceModality


class DWIModality(MultisequenceModality):
    """ Class to handle DWI modality.

    Parameters
    ----------
    path_data : str, list of str, or None, optional (default=None)
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : str or list of str
        Location of the data.

    data_ : array-like, shape (B_seq, Y, X, Z)
        The different volume from different B sequuences. The data are
        saved in B_seq, Y, X, Z ordered.

    pdf_series_ : list, length (n_serie)
        List of the PDF for each serie.

    bin_series_ : list of ndarray, length (n_serie)
        List of the bins used to plot the pdfs.

    max_series_ : float
        Maximum intensity of all the DCE series.

    min_series_ : float
        Minimum intensity of all the DCE series.

    n_serie_ : integer
        Number of serie in this DCE sequence.

    max_series_list_ : list of float
        List of the maximum intensity for each DCE serie.

    min_series_list_ : list of float
        List of the minimum intensity for each DCE serie.
    """

    def __init__(self, path_data=None):
        super(DWIModality, self).__init__(path_data=path_data)
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
                             ' function read_data_from_path()')

        # Compute the min and max from all DCE series
        self.max_series_ = np.ndarray.max(self.data_)
        self.min_series_ = np.ndarray.min(self.data_)

        # For each serie compute the pdfs and store them
        pdf_series = []
        bin_series = []
        min_series_list = []
        max_series_list = []

        for data_serie in self.data_:
            bins = int(np.round(np.ndarray.max(data_serie) -
                                np.ndarray.min(data_serie)))

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

    def read_data_from_path(self, path_data=None):
        """Function to read DCE images which is of 3D volume over time.

        Parameters
        ----------
        path_data : str or None, optional (default=None)
            Path to the temporal data. It will overrides the path given
            in the constructor.

        Return
        ------
        self : object
           Returns self.
        """
        # Called the parent function to read the data
        super(DWIModality, self).read_data_from_path(path_data=path_data)

        # Compute the information regarding the pdf of the DCE series
        self._update_histogram()

        return self
