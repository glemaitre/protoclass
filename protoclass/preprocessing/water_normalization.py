"""Gaussian normalization to normalize standalone modality."""

import numpy as np

from joblib import Parallel, delayed

from scipy.special import wofz
from scipy.optimize import curve_fit
from scipy.integrate import simps

from six import string_types

from .mrsi_normalization import MRSINormalization


PPM_REFERENCE = {'water' : 4.65}
PPM_LIMITS = {'water': (4., 6.)}


def _voigt_profile(x, alpha, mu, sigma, gamma):
    """Private function to fit a Voigt profile.

    Parameters
    ----------
    x : ndarray, shape (len(x))
        The input data.

    alpha : float,
        The amplitude factor.

    mu : float,
        The shift of the central value.

    sigma : float,
        sigma of the Gaussian.

    gamma : float,
        gamma of the Lorentzian.

    Returns
    -------
    y : ndarray, shape (len(x), )
        The Voigt profile.

    """

    # Define z
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))

    # Compute the Faddeva function
    w = wofz(z)

    return alpha * (np.real(w)) / (sigma * np.sqrt(2. * np.pi))


def _fit_voigt_water_integral(ppm, spectra):
    """Private function to fit water peak in one spectra and return
    the integral of the signal.

    Parameters
    ----------
    ppm : ndarray, shape (n_samples, )
        The PPM array.

    spectra : ndarray, shape (n_samples, )
        The spectra on which the water has to be fitted.

    Returns
    -------
    int_water : float,
        The integral of the water peak.

    """

    # Get the value between of the spectra between 4 and 6
    # Find the indices in that range
    water_limits = PPM_LIMITS['water']
    idx_samples = np.flatnonzero(np.bitwise_and(ppm > water_limits[0],
                                                ppm < water_limits[1]))
    # Extrac the ppm and spectrum
    sub_ppm = ppm[idx_samples]
    sub_spectra = spectra[idx_samples]

    # Define the default parameters
    amp_dft = np.max(sub_spectra) / _voigt_profile(0., 1., 0., 1., 1.)
    mu_default = sub_ppm[np.argmax(sub_spectra)]
    popt_default = [amp_dft, mu_default, 1., 1.]
    # Define the bound
    param_bounds = ([0., water_limits[0], 0., 0.],
                    [np.inf, water_limits[1], np.inf, np.inf])

    try:
        popt, _ = curve_fit(_voigt_profile, sub_ppm, np.real(sub_spectra),
                            p0=popt_default, bounds=param_bounds)

        # Create x
        int_ppm = np.linspace(water_limits[0], water_limits[1], num=5000)
        # Make the integral
        int_water = simps(_voigt_profile(int_ppm, np.real(popt[0]),
                                         np.real(popt[1]),
                                         np.real(popt[2]),
                                         np.real(popt[3])), int_ppm)
    except RuntimeError:
        # The fitting failed return a small value
        int_water = np.finfo(float).eps

    return int_water


class WaterNormalization(MRSINormalization):
    """Normalization of MRSI data using water integral.

    Normalization of the MRSI spectrum. Only the real part will be normalized.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from MRSIModality class.

    params : str or str: float, optional (default='auto')
        The initial estimation of the parameters:

        - If 'auto', then the standard deviation and mean will be estimated
        from the data.
        - A float can be given to normalize the spectrum.

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

    def __init__(self, base_modality, params='auto'):
        super(WaterNormalization, self).__init__(base_modality)
        self.params = params
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
        super(WaterNormalization, self).fit(modality=modality,
                                            ground_truth=ground_truth,
                                            cat=cat)

        if isinstance(self.params, string_types):
            if self.params == 'auto':
                # During fitting we will find the constant parameter to use
                # to normalize each spectrum
                # 1. Reshape all data for parallel processing
                spectra = np.reshape(modality.data_, (
                    modality.data_.shape[0],
                    modality.data_.shape[1] *
                    modality.data_.shape[2] *
                    modality.data_.shape[3])).T

                ppm = np.reshape(modality.bandwidth_ppm, (
                    modality.bandwidth_ppm.shape[0],
                    modality.bandwidth_ppm.shape[1] *
                    modality.bandwidth_ppm.shape[2] *
                    modality.bandwidth_ppm.shape[3])).T

                # 2. Make the fitting and get the integral of the water peak
                self.fit_params_ = Parallel(n_jobs=-1)(
                    delayed(
                        _fit_voigt_water_integral)(p, s)
                    for p, s in zip(ppm, spectra))

                # 3. Reshape the output array
                self.fit_params_ = np.reshape(
                    self.fit_params_,
                    (modality.bandwidth_ppm.shape[1],
                     modality.bandwidth_ppm.shape[2],
                     modality.bandwidth_ppm.shape[3]))
            else:
                raise ValueError('The string in params is unknown.')
        elif isinstance(self.params, float):
            self.fit_params_ = np.ones((modality.data_.shape[1],
                                        modality.data_.shape[2],
                                        modality.data_.shape[3])) * self.params
        else:
            raise ValueError('The type of params is unknown.')

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
        super(WaterNormalization, self).normalize(modality)

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
        super(WaterNormalization, self).denormalize(modality)

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
