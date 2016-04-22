""" DCE modality class.
"""

import numpy as np

from .temporal_modality import TemporalModality


class DCEModality(TemporalModality):
    """ Class to handle DCE-MRI modality.

    Parameters
    ----------
    path_data : str or None, optional (default=None)
        Path where the data are located. Can also be specified
        when reading the data. It will be overidden with this function.

    Attributes
    ----------
    path_data_ : str
        Location of the data.

    data_ : ndarray, shape (T, Y, X, Z)
        The different volume of the DCE serie. The data are saved in
        T, Y, X, Z ordered.

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
        super(DCEModality, self).__init__(path_data=path_data)

    def update_histogram(self, nb_bins=None):
        """Function to compute histogram of each serie and store it
        The min and max of the series are also stored

        Parameters
        ----------
        nb_bins : list of int or None, optional (default=None)
            The numbers of bins to use to compute the histogram. Since that we
            deal with several series, a list needs to be provided. If None, the
            number of bins found at reading will be used.

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
        if ((nb_bins is not None) and
            (nb_bins != 'auto') and
            (len(nb_bins) != len(self.data_))):
            raise ValueError('Provide a list of number of bins with the same'
                             ' size as the number of serie in the data.')
        # Get the list of number of bins if not specify
        elif nb_bins is None:
            nb_bins = self.nb_bins_
        elif nb_bins is 'auto':
            nb_bins = []
            for data_serie in self.data_:
                nb_bins.append(int(np.round(np.ndarray.max(data_serie) -
                                            np.ndarray.min(data_serie))))


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

    def build_heatmap(self):
        """ Function which return a heatmap using the pdf of each serie

        Returns
        -------
        heatmap : ndarray, shape (n_serie, intensity_range)
             Return an heatmap of the different pdfs. This equivalent to
             pdf_series_ but properly shifted inside an array.
        """
        # Check that the data have been read
        if self.data_ is None:
            raise ValueError('You need to load the data first. Refer to the'
                             'function read_data_from_path().')

        # Allocate the heatmap array
        # Compute the intensity range
        int_range = int(np.ceil(self.max_series_) - np.floor(self.min_series_))
        heatmap = np.zeros((self.n_serie_, int_range))

        # Build the heatmap
        # Go through each serie and paste it inside the array
        for idx_serie, pdf_serie in enumerate(self.pdf_series_):
            # Define the range of the current histogram
            int_range = pdf_serie.size
            # Compute the offset between the minimum of all series and the
            # current one
            offset_minimum = int(np.floor(self.min_series_) -
                                 np.floor(self.min_series_list_[idx_serie]))
            # Copy the data at the right position
            heatmap[idx_serie, range(offset_minimum,
                                     offset_minimum + int_range)] = pdf_serie

        return heatmap

    def read_data_from_path(self, path_data=None):
        """Function to read DCE images which is of 3D volume over time.

        Parameters
        ----------
        path_data : str or None, optional (default=None)
            Path to the temporal data. It will overrides the path given
            in the constructor.

        Returns
        -------
        self : object
           Returns self.
        """
        # Called the parent function to read the data
        super(DCEModality, self).read_data_from_path(path_data=path_data)

        # Create the list of number of bins to compute the histogram
        self.nb_bins_ = []
        for data_serie in self.data_:
            self.nb_bins_.append(int(np.round(np.ndarray.max(data_serie) -
                                              np.ndarray.min(data_serie))))


        # Compute the information regarding the pdf of the DCE series
        self.update_histogram()

        return self
