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

        data = modality.data_[self.ppm_range[0]:self.ppm_range[1]]

        data_res = np.zeros((data.shape[0], modality.data_.shape[1],
                             modality.data_.shape[2], modality.data_.shape[3]))

        # Resample each ppm of the spectum
        for ii in range(data.shape[0]):
            slice_data = data[ii]
            # Resample this slice
            data_res[ii] = self._resampling_as_gt(slice_data, modality,
                                                  ground_truth)

        return data_res[self.roi_data_]
