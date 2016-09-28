"""Gaussian normalization to normalize standalone modality."""

import numpy as np

from joblib import Parallel, delayed

from scipy.linalg import norm

from .mrsi_normalization import MRSINormalization


KNOWN_KIND = ('l2', 'l1')


class LNormNormalization(MRSINormalization):
    """Normalization of MRSI data using l-norm.

    Normalization of the MRSI spectrum. Only the real part will be normalized.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from MRSIModality class.

    kind : str, optional (default='l2')
        The type of l-norm to use. Can be either 'l2' or 'l1'.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from MRSIModality class.

    fit_params_ : ndarray, shape (size_x, size_y, size_z)
        The constant to apply to each spectrum.

    is_fitted_ : bool
        Boolean to know if the `fit` function has been already called.

    """

    def __init__(self, base_modality, kind='l2'):
        super(LNormNormalization, self).__init__(base_modality)
        self.kind = kind
        self.is_fitted_ = False

    def fit(self, modality, ground_truth=None, cat=None):
        """Method to find the parameters needed to apply the normalization.

        Parameters
        ----------
        modality : object of type MRSIModality
            The modality object of interest.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Returns
        -------
        self : object
             Return self.

        """
        super(LNormNormalization, self).fit(modality=modality,
                                            ground_truth=ground_truth,
                                            cat=cat)

        # Check that the type of l-norm is known
        if self.kind not in KNOWN_KIND:
            raise ValueError('The type of l-norm is not known.')

        # Allocate the parameters array
        self.fit_params_ = np.zeros((modality.data_.shape[1],
                                     modality.data_.shape[2],
                                     modality.data_.shape[3]))

        for y in range(modality.data_.shape[1]):
            for x in range(modality.data_.shape[2]):
                for z in range(modality.data_.shape[3]):
                    if self.kind == 'l1':
                        self.fit_params_[y, x, z] = norm(np.real(
                            modality.data_[:, y, x, z]), 1)
                    if self.kind == 'l2':
                        self.fit_params_[y, x, z] = norm(np.real(
                            modality.data_[:, y, x, z]), 2)

        self.is_fitted_ = True

        return self

    def normalize(self, modality):
        """Method to normalize the given modality using the fitted parameters.

        Parameters
        ----------
        modality: object of type MRSIModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type MRSIModality
            The modality object in which the data will be normalized.

        """
        super(LNormNormalization, self).normalize(modality)

        # Check that the parameters have been fitted
        if not self.is_fitted_:
            raise ValueError('Fit the parameters previous to normalize'
                             ' the data.')

        for y in range(modality.data_.shape[1]):
            for x in range(modality.data_.shape[2]):
                for z in range(modality.data_.shape[3]):
                    modality.data_[:, y, x, z] = ((
                        np.real(modality.data_[:, y, x, z]) /
                        self.fit_params_[y, x, z]) + (
                            1j * np.imag(modality.data_[:, y, x, z])))

        return modality

    def denormalize(self, modality):
        """Denormalize the given modality using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.

        """
        super(LNormNormalization, self).denormalize(modality)

        # Check that the parameters have been fitted
        if not self.is_fitted_:
            raise ValueError('Fit the parameters previous to normalize'
                             ' the data.')

        for y in range(modality.data_.shape[1]):
            for x in range(modality.data_.shape[2]):
                for z in range(modality.data_.shape[3]):
                    modality.data_[:, y, x, z] = ((
                        np.real(modality.data_[:, y, x, z]) *
                        self.fit_params_[y, x, z]) + (
                            1j * np.imag(modality.data_[:, y, x, z])))

        return modality
