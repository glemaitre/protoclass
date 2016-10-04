"""MRSI spectra extraction from MRSI modality."""
from __future__ import division

import numpy as np

from scipy.linalg import norm as lnorm

from .mrsi_extraction import MRSIExtraction


KNOWN_NORMALIZATION = ('l2', 'l1')


class MRSISpectraExtraction(MRSIExtraction):
    """MRSI spectra extraction from MRSI modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    ppm_range : tuple of float, optional (default=(2., 4.))
        Define the range of ppm to extract from the spectrum.

    normalization : None or str, optional (default='l2')
        Apply a normalization or not. Choice are None, 'l2', or 'l1'.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, ppm_range=(2., 4.), normalization='l2'):
        super(MRSISpectraExtraction, self).__init__(base_modality)
        self.ppm_range = ppm_range
        self.normalization = normalization
        self.is_fitted = False
        self.data_ = None

    def fit(self, modality, ground_truth=None, cat=None):
        """Find the parameters needed to apply the extraction.

        Parameters
        ----------
        modality : object of type TemporalModality
            The modality object of interest.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Return
        ------
        self : object
             Return self.

        """
        super(MRSISpectraExtraction, self).fit(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)


        # Check if we need to know the normalization factor
        if self.normalization is not None:
            if self.normalization not in KNOWN_NORMALIZATION:
                raise ValueError('Unknown normalization.')

            self.fit_params_ = np.zeros((modality.data_.shape[1],
                                         modality.data_.shape[2],
                                         modality.data_.shape[3]))

            for y in range(modality.data_.shape[1]):
                for x in range(modality.data_.shape[2]):
                    for z in range(modality.data_.shape[3]):
                        if self.normalization == 'l1':
                            self.fit_params_[y, x, z] = lnorm(np.real(
                                modality.data_[:, y, x, z]), 1)
                        if self.normalization == 'l2':
                            self.fit_params_[y, x, z] = lnorm(np.real(
                                modality.data_[:, y, x, z]), 2)

        self.is_fitted = True

        return self

    def transform(self, modality, ground_truth=None, cat=None):
        """Extract the data from the given modality.

        Parameters
        ----------
        modality : object of type StandaloneModality
            The modality object of interest.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Returns
        ------
        data : ndarray, shape (n_sample, n_feature)
             A matrix containing the features extracted. The number of samples
             is equal to the number of positive label in the ground-truth.

        """
        super(MRSISpectraExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that the data have been fitted
        if not self.is_fitted:
            raise ValueError('Fit the data first.')

        # We need first to crop the data properly depending of the ppm range
        idx_ppm_crop = []
        for y in range(modality.bandwidth_ppm.shape[1]):
            for x in range(modality.bandwidth_ppm.shape[2]):
                for z in range(modality.bandwidth_ppm.shape[3]):
                    # Get the range for the current data
                    # Compute the delta
                    delta_ppm = np.abs((modality.bandwidth_ppm[1, y, x, z] -
                                        modality.bandwidth_ppm[0, y, x, z]))
                    # Compute the number of element to take
                    nb_element = int(np.ceil((self.ppm_range[1] -
                                              self.ppm_range[0]) / delta_ppm))
                    # Find the first index
                    first_idx = np.flatnonzero(
                        modality.bandwidth_ppm[:, y, x, z] >
                        self.ppm_range[0])[-1]
                    idx_mask = np.arange(first_idx, first_idx - nb_element, -1)
                    idx_ppm_crop.append(idx_mask)

        # Convert the list into an array
        idx_ppm_crop = np.array(idx_ppm_crop)

        # Reshape the array according to the data
        idx_ppm_crop = np.reshape(idx_ppm_crop.T,
                                  (idx_ppm_crop.shape[1],
                                   modality.bandwidth_ppm.shape[1],
                                   modality.bandwidth_ppm.shape[2],
                                   modality.bandwidth_ppm.shape[3]))

        # Extract the appropriate part of each signal
        data_crop = np.zeros(idx_ppm_crop.shape)
        for y in range(modality.data_.shape[1]):
            for x in range(modality.data_.shape[2]):
                for z in range(modality.data_.shape[3]):
                    data_crop[:, y, x, z] = np.real(modality.data_[
                        idx_ppm_crop[:, y, x, z], y, x, z])
                    # Apply the normalization if necessary
                    if self.normalization is not None:
                        data_crop[:, y, x, z] /= self.fit_params_[y, x, z]

        data_res = np.zeros((data_crop.shape[0],
                             ground_truth.data_.shape[1],
                             ground_truth.data_.shape[2],
                             ground_truth.data_.shape[3]))

        # Resample each ppm of the spectum
        for ii in range(data_crop.shape[0]):
            # Resample each ppm slice
            data_res[ii, :, :, :] = self._resampling_as_gt(
                data_crop[ii, :, :, :],
                modality,
                ground_truth)

        # Convert the roi to a numpy array
        roi_data = np.array(self.roi_data_)

        # Check the number of samples which will be extracted
        n_sample = roi_data.shape[1]
        # Check the number of dimension
        n_dimension = data_res.shape[0]

        # Allocate the array
        data = np.empty((n_sample, n_dimension))

        # Copy the data at the right place
        for idx_sample in range(n_sample):
            # Get the coordinate of the point to consider
            coord = roi_data[:, idx_sample]

            # Extract the data
            data[idx_sample, :] = data_res[:,
                                           coord[0],
                                           coord[1],
                                           coord[2]]

        return data
