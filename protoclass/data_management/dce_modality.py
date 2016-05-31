"""DCE modality class."""

import warnings

import numpy as np
import SimpleITK as sitk

from datetime import datetime

from scipy.interpolate import interp1d

from skimage.measure import label
from skimage.measure import regionprops

from sklearn.cluster import KMeans

from .temporal_modality import TemporalModality

from ..utils import find_nearest
from ..utils.validation import check_path_data

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

    metadata_ : dict
        A dictionary containing the metadata related to the DCE serie.

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

    time_info_ : ndarray, shape (n_serie, )
        Array containing the time information of the acquisition time
        in seconds

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

        nb_bins : list of int or str, optional (default='auto')
            The numbers of bins to use to compute the histogram.
            The possibilities are:
            - If 'auto', the number of bins is found at fitting time.
            - Otherwise, a list of integer needs to be given.

        Returns
        -------
        pdf_list : list of ndarray, length (n_serie)
            List of the pdf with the associated series.

        bin_list : list of ndarray, length (n_series + 1)
            List of the bins associated with the list of pdf.

        """
        # Check that the data have been read
        if self.data_ is None:
            raise ValueError('You need to load the data first. Refer to the'
                             ' function read_data_from_path().')

        # Check if we have to auto compute the range of the histogram
        # Check that we have a proper list of bins
        if isinstance(nb_bins, basestring):
            if nb_bins == 'auto':
                nb_bins = []
                for data_serie in self.data_:
                    if roi_data is None:
                        nb_bins.append(int(np.round(
                            np.ndarray.max(data_serie) -
                            np.ndarray.min(data_serie))))
                    else:
                        nb_bins.append(int(np.round(
                            np.ndarray.max(data_serie[roi_data]) -
                            np.ndarray.min(data_serie[roi_data]))))
            else:
                raise ValueError('Wrong string argument to specify the'
                                 ' number of bins.')
        elif isinstance(nb_bins, list):
            if (len(nb_bins) != len(self.data_) or
                    not all(isinstance(x, int) for x in nb_bins)):
                raise ValueError('Provide a list of number of bins with'
                                 ' the same size as the number of serie in'
                                 ' the data.')
        else:
            raise ValueError('Unknown arguments for `nb_bins`.')

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
                raise ValueError('Unknown string for `nb_bins`.')
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
        pdf_list, bins_list = self.get_pdf_list(roi_data=roi_data,
                                                nb_bins=nb_bins)
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
        for idx_serie, (bin_serie, pdf_serie) in enumerate(
                zip(center_bins_list,
                    pdf_list)):
            # We need to interpolate the histogram values
            min_value, min_idx = find_nearest(bins_heatmap,
                                              min_series[idx_serie])
            max_value, _ = find_nearest(bins_heatmap,
                                              max_series[idx_serie])

            # Interpolate the value using nearest neighbour
            f = interp1d(bin_serie, pdf_serie, kind='nearest',
                         bounds_error=False, fill_value=0.)

            nb_bin = int((max_value - min_value) / np.min(ratio_bins))
            bin_serie_interpolated = np.linspace(min_value, max_value, nb_bin)
            pdf_serie_interpolated = f(bin_serie_interpolated)

            # Copy the data at the right position
            heatmap[idx_serie, min_idx:min_idx+nb_bin] = pdf_serie_interpolated

        return heatmap, bins_heatmap

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
            # Store the data related
            self.metadata_['acq-time'] = [datetime.strptime(
                im.GetMetaData('0008|0032').replace(' ', ''),
                '%H%M%S.%f')]
            refit = True
        else:
            im = sitk.ReadImage(filename)
            self.metadata_['acq-time'].append(datetime.strptime(
                im.GetMetaData('0008|0032').replace(' ', ''),
                '%H%M%S.%f'))

        return refit

    def read_data_from_path(self, path_data=None):
        """Function to read temporal images which represent a 3D volume
        over time.

        Parameters
        ----------
        path_data : str, list of str, or None, optional (default=None)
            Path to the temporal data. It will overrides the path given
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
            series_time = np.array(reader.GetGDCMSeriesIDs(self.path_data_))

            # Check that you have more than one serie
            if len(series_time) < 2:
                raise ValueError('The time serie should at least contain'
                                 ' 2 series.')

            # The IDs need to be re-ordered in an incremental manner
            # Create a list by converting to integer the number after
            # the last full stop
            id_series_time_int = np.array([int(s[s.rfind('.')+1:])
                                          for s in series_time])
            # Sort and get the corresponding index
            idx_series_sorted = np.argsort(id_series_time_int)

            # Open the volume in the sorted order
            list_volume = []
            is_dict_built = False
            for id_time in series_time[idx_series_sorted]:
                # Get the filenames corresponding to the current ID
                dicom_names_serie = reader.GetGDCMSeriesFileNames(self.path_data_,
                                                                  id_time)
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

            # Compute the time information from the DICOM tag kept
            # Initialize the first value
            self.time_info_ = [0.]
            for idx_time in range(1, len(self.metadata_['acq-time'])):
                delta_sec = (self.metadata_['acq-time'][idx_time] -
                             self.metadata_['acq-time'][0])
                self.time_info_.append(delta_sec.total_seconds())
            self.time_info_ = np.array(self.time_info_)

            # We can create a numpy array
            # The first dimension corresponds to the time dimension
            # When processing the data, we need to slice the data
            # considering this dimension emphasizing the decision to let
            # it at the first position.
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

            # Compute the time information from the DICOM tag kept
            # Initialize the first value
            self.time_info_ = [0.]
            for idx_time in range(1, len(self.metadata_['acq-time'])):
                delta_sec = (self.metadata_['acq-time'][idx_time] -
                             self.metadata_['acq-time'][0])
                self.time_info_.append(delta_sec.total_seconds())
            self.time_info_ = np.array(self.time_info_)

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

    def compute_aif(self, n_clusters=6, eccentricity=.5, diameter=(10., 20.),
                    area=(100., 400.), estimator='median'):
        """Determine the AIF by segmenting the aorta in the kinetic sequence.

        Parameters
        ----------
        n_clusters : int, optional (default=6)
            The number of clusters to use to make the detection of the zone
            of interest to later segment the aorta or veins.

        eccentricity : float, optional (default=.5)
            The eccentricity is the ratio of the focal distance
            (distance between focal points) over the major axis length. The
            value is in the interval [0, 1). When it is 0, the ellipse becomes
            a circle. Greater is more permissive and find more regions of
            interest.

        diameter : tuple of float, optional (default=(10., 20.))
            Tuple of the minimum and maximum value of the diameters of the
            region. The region having a diameter included in this interval
            will be kept as potential region.

        area : tuple of float, optional (default=(100., 400.))
            Tuple of the minimum and maximum area in between which the region
            of interest will be kept.

        estimator : str, optional (default='median')
            The estimator used to estimate the AIF from the segmented region.
            Can be the following: 'median' and 'mean'

        Returns
        -------
        aif : ndarray, shape (n_series, )
            The estimated AIF signal.

        """
        # Check that the data have been read
        if self.data_ is None:
            raise RuntimeError('Read the data first.')

        # Get the size of the volume
        sz_vol = self.metadata_['size']

        # For each slice
        signal_aif = np.empty((0, self.n_serie_), dtype=float)
        for idx_sl in range(sz_vol[2]):

            # Crop the upper part of the image
            org_im = self.data_[:, 50:(sz_vol[1] / 2), :, idx_sl]

            # Reshape the data to make some clustring later on
            sz_croped_im = org_im.shape
            data = np.reshape(org_im, (sz_croped_im[0],
                                       sz_croped_im[1] *
                                       sz_croped_im[2])).T

            # Make a k-means filtering
            km = KMeans(n_clusters=n_clusters,
                        n_jobs=-1)
            # Fit and predict the data
            data_label = km.fit_predict(data)

            # Skip to the next iteration if we did not find any candidate
            if np.unique(data_label).size < 2:
                continue

            # Find the cluster with the highest enhancement - it will
            # correspond to blood
            cl_perc = []
            for cl in np.unique(data_label):

                # Compute the maximum enhancement of the current cluster
                # and find the 90 percentile
                perc = np.percentile(np.max(data[data_label == cl],
                                            axis=1), 90)
                cl_perc.append(perc)

            # Select only the cluster of interest
            cl_aorta = np.argmax(cl_perc)
            bin_im = np.reshape([data_label == cl_aorta], (sz_croped_im[1],
                                                           sz_croped_im[2]))
            # Transform the binary image into a labelled image
            label_im = label(bin_im.astype(int))

            # Compute the property for each region labelled
            regions = regionprops(label_im)

            # Remove the regions in the image which do not follow the
            # specificity imposed
            for idx_reg, reg in enumerate(regions):

                # Check the eccentricity
                if reg.eccentricity > eccentricity:
                    label_im[np.nonzero(label_im == idx_reg + 1)] = 0
                    continue

                # Check the diameter
                if (reg.equivalent_diameter < diameter[0] or
                        reg.equivalent_diameter > diameter[1]):
                    label_im[np.nonzero(label_im == idx_reg + 1)] = 0
                    continue

                # Check the area
                if reg.area < area[0] or reg.area > area[1]:
                    label_im[np.nonzero(label_im == idx_reg + 1)] = 0
                    continue

            # Store the signal that will be used to estimated the AIF
            if np.count_nonzero(label_im) > 0:
                signal_aif = np.vstack((signal_aif,
                                        org_im[:,
                                               np.nonzero(label_im)[0],
                                               np.nonzero(label_im)[1]].T))

        # Get the final estimate
        if estimator == 'median':
            aif = np.median(signal_aif, axis=0)
        elif estimator == 'mean':
            aif = np.mean(signal_aif, axis=0)
        else:
            raise ValueError('Unknown string for the parameter estimator.')

        return aif
