"""DCE modality class."""

import numpy as np

from scipy.interpolate import interp1d

from .temporal_modality import TemporalModality

from ..utils import find_nearest

class DCEModality(TemporalModality):
    """Class to handle DCE-MRI modality.

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

    def get_pdf_list(self, roi_data=None, nb_bins='auto'):
        """ Extract the a list of pdf related with the data.

        Parameters
        ----------
        roi_data : tuple
            Indices of elements to consider while computing the histogram.
            The ROI is a 3D volume which will be used for each time serie.

        nb_bins : int, str or None, optional (default=None)
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If 'auto', the number of bins is found at fitting time.
            - Otherwise, an integer needs to be given.

        Returns
        -------
        pdf_list : list of ndarray, length (n_serie)
            List of the pdf with the associated series.

        bin_list : list of ndarray, length (n_series)
            List of the bins associated with the list of pdf.

        """
        # Check that the data have been read
        if self.data_ is None:
            raise ValueError('You need to load the data first. Refer to the'
                             ' function read_data_from_path().')

        # Check if we have to auto compute the range of the histogram
        # Check that we have a proper list of bins
        if ((nb_bins != 'auto') and
            (len(nb_bins) != len(self.data_))):
            raise ValueError('Provide a list of number of bins with the same'
                             ' size as the number of serie in the data.')
        elif nb_bins == 'auto':
            nb_bins = []
            for data_serie in self.data_:
                if roi_data is None:
                    nb_bins.append(int(np.round(np.ndarray.max(data_serie) -
                                                np.ndarray.min(data_serie))))
                else:
                    nb_bins.append(int(np.round(
                        np.ndarray.max(data_serie[roi_data]) -
                        np.ndarray.min(data_serie[roi_data]))))

        pdf_list = []
        bin_list = []
        # Go through each serie to compute the pdfs
        for data_serie, bins in zip(self.data_, nb_bins):
            if roi_data is None:
                pdf_s, bin_s = np.histogram(data_serie,
                                            bins=bins,
                                            density=True)
            else:
                pdf_s, bin_s = np.histogram(data_serie[roi_data],
                                            bins=bins,
                                            density=True)
            pdf_list.append(pdf_s)
            bin_list.append(bin_s)

        return pdf_list, bin_list

    def update_histogram(self, nb_bins=None):
        """Update the histogram of each serie and first-order statistics.

        Parameters
        ----------
        nb_bins : int, str, or None, optional (default=None)
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If None, the number of bins found at reading will be used.
            - If 'auto', the number of bins is found at fitting time.
            - Otherwise, an integer needs to be given.

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
        elif nb_bins == 'auto':
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

    def build_heatmap(self, roi_data=None, nb_bins='auto'):
        """Function which return a heatmap using the pdf of each serie.

        Parameters
        ----------
        roi_data : tuple
            Indices of elements to consider while computing the histogram.
            The ROI is a 3D volume which will be used for each time serie.

        nb_bins : int, str or None, optional (default=None)
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If 'None', the number of bins found at reading will be used.
            - If 'auto', the number of bins is found at fitting time.
            - Otherwise, an integer needs to be given.

        Returns
        -------
        heatmap : ndarray, shape (n_serie, intensity_range)
            Return an heatmap of the different pdfs. This equivalent to
            pdf_series_ but properly shifted inside an array.

        bins_heatmap : ndarray, shape (intensity_range, )
            Returns the bins associated with the heatmap.

        """
        # Check that the data have been read
        if self.data_ is None:
            raise ValueError('You need to load the data first. Refer to the'
                             ' function read_data_from_path().')

        if nb_bins is None:
            # We will used the value found during the data reading
            nb_bins = self.nb_bins_

        # Compute the list of pdfs
        pdf_list, bins_list = self.get_pdf_list(roi_data, nb_bins)
        # We will take the center for each bins
        center_bins_list = []
        for bins_serie in bins_list:
            center_bins_list.append((bins_serie[:-1] + bins_serie[1:]) / 2.)

        # Find the extrema
        max_series = []
        min_series = []
        ratio_bins = []
        for bins_serie in center_bins_list:
            # Find the max and min
            min_series.append(np.ndarray.min(bins_serie))
            max_series.append(np.ndarray.max(bins_serie))
            # Compute the width of each bin
            ratio_bins.append((np.ndarray.max(bins_serie) -
                               np.ndarray.min(bins_serie)) /
                              float(bins_serie.size))

        # Allocate the data for the heatmap
        bins_heatmap = np.linspace(np.min(min_series), np.max(max_series),
                                   int((np.max(max_series) -
                                        np.min(min_series)) /
                                       np.min(ratio_bins)))
        heatmap = np.zeros((self.n_serie_, bins_heatmap.size))

        # Build the heatmap
        # Go through each serie and paste it inside the array
        for idx_serie, (bin_serie, pdf_serie) in enumerate(zip(center_bins_list, pdf_list)):
            # We need to interpolate the histogram values
            min_value, min_idx = find_nearest(bins_heatmap,
                                              min_series[idx_serie])
            max_value, max_idx = find_nearest(bins_heatmap,
                                              max_series[idx_serie])

            # Interpolate the value using nearest neighbour
            f = interp1d(bin_serie, pdf_serie, kind='nearest',
                         bounds_error=False, fill_value=0.)
            
            nb_bin = int((max_value - min_value) / np.min(ratio_bins))
            bin_serie_interpolated = np.linspace(min_value, max_value, nb_bin)
            pdf_serie_interpolated = f(bin_serie_interpolated)

            # Copy the data at the right position
            heatmap[idx_serie, min_idx:min_idx + nb_bin] = pdf_serie_interpolated

        return heatmap, bins_heatmap

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
