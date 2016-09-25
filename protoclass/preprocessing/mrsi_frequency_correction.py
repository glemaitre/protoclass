"""Frequency correction class for MRSI modality."""

import os

import cPickle as pickle
import numpy as np

from joblib import Parallel, delayed

from scipy.special import wofz
from scipy.optimize import curve_fit

from ..data_management import MRSIModality
from ..data_management import GTModality

from ..utils.validation import check_modality_inherit
from ..utils.validation import check_filename_pickle_load
from ..utils.validation import check_filename_pickle_save
from ..utils.validation import check_modality


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


def _fit_voigt_water(ppm, spectra):
    """Private function to fit water peak in one spectra.

    Parameters
    ----------
    ppm : ndarray, shape (n_samples, )
        The PPM array.

    spectra : ndarray, shape (n_samples, )
        The spectra on which the water has to be fitted.

    Returns
    -------
    popt : list of float,
        A list of the fitted parameters.

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
    param_bounds = ([0., 4., 0., 0.], [np.inf, 6., np.inf, np.inf])

    try:
        popt, _ = curve_fit(_voigt_profile, sub_ppm, np.real(sub_spectra),
                            p0=popt_default, bounds=param_bounds)
    except RuntimeError:
        popt = popt_default

    return np.real(popt)


class MRSIFrequencyCorrection(object):
    """Phase correction for MRSI modality.

    Parameters
    ----------
        base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    """

    def __init__(self, base_modality):
        self.base_modality_ = check_modality_inherit(base_modality,
                                                     MRSIModality)
        self.fit_params_ = None

    def fit(self, modality, ground_truth=None, cat=None):
        """Find the parameters needed to apply the phase correction.

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

        Return
        ------
        self : object
             Return self.

        """

        # Check that the modality is from the template class
        check_modality(modality, self.base_modality_)

        # Check that the data were read during the creation of the modality
        if not modality.is_read():
            raise ValueError('No data have been read during the construction'
                             ' of the modality object.')

        # In the fitting we will find the parameters needed to transform
        # the data
        # 1. Reshape all data for parallel processing
        spectra = np.reshape(modality.data_, (modality.data_.shape[0],
                                              modality.data_.shape[1] *
                                              modality.data_.shape[2] *
                                              modality.data_.shape[3])).T
        ppm = np.reshape(modality.bandwidth_ppm, (
            modality.bandwidth_ppm.shape[0],
            modality.bandwidth_ppm.shape[1] *
            modality.bandwidth_ppm.shape[2] *
            modality.bandwidth_ppm.shape[3])).T
        # 2. Make the fitting and get the parameters
        params_opt = Parallel(n_jobs=-1)(delayed(_fit_voigt_water)(p, s)
                                         for p, s in zip(ppm, spectra))
        # 3. Reshape the parameters array
        params_opt = np.array(params_opt)
        # Select only the shift parameters
        params_opt = params_opt[:, 1]
        self.fit_params_ = np.reshape(params_opt, (
            modality.bandwidth_ppm.shape[1],
            modality.bandwidth_ppm.shape[2],
            modality.bandwidth_ppm.shape[3]))

        # Once the fitting is performed, we can modify the erronous findings.
        # These findings are the voxel to far from the center which we are not
        # interested anyway. We will find them and affect the mean value to
        # these ones.
        # 1. Find the median value for the shift
        med_shift = np.median(self.fit_params_)
        # 2. We will loop through all the spectra and the one which do not have
        # a maximum in the water area will be shifted using the median value
        water_limits = PPM_LIMITS['water']

        for y in range(modality.data_.shape[1]):
            for x in range(modality.data_.shape[2]):
                for z in range(modality.data_.shape[3]):
                    idx_samples = np.flatnonzero(np.bitwise_and(
                        modality.bandwidth_ppm[:, y, x, z] > water_limits[0],
                        modality.bandwidth_ppm[:, y, x, z] < water_limits[1]))
                    if (np.max(modality.data_[:, y, x, z]) !=
                        np.max(modality.data_[idx_samples, y, x, z])):
                        self.fit_params_[y, x, z] = med_shift

        # 3. Apply a 2 sigma rules to remove the potential outliers
        med_shift = np.median(self.fit_params_)
        std_shift = np.std(self.fit_params_)
        for y in range(modality.data_.shape[1]):
            for x in range(modality.data_.shape[2]):
                for z in range(modality.data_.shape[3]):
                    idx_samples = np.flatnonzero(np.bitwise_and(
                        modality.bandwidth_ppm[:, y, x, z] > water_limits[0],
                        modality.bandwidth_ppm[:, y, x, z] < water_limits[1]))
                    if (self.fit_params_[y, x, z] >
                        med_shift + 2. * std_shift or
                        self.fit_params_[y, x, z] <
                        med_shift - 2. * std_shift):
                        self.fit_params_[y, x, z] = med_shift

        return self

    def transform(self, modality, ground_truth=None, cat=None):
        """Correct the phase of an MRSI modality.

        Parameters
        ----------
        modality : MRSIModality,
            The MRSI modality in which the phase need to be corrected.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Returns
        -------
        modality : MRSIModality,
            Return the MRSI modaity in which the phase has venn corrected.

        """

        # Check that the modality is from the template class
        check_modality(modality, self.base_modality_)

        # Check that the data were read during the creation of the modality
        if not modality.is_read():
            raise ValueError('No data have been read during the construction'
                             ' of the modality object.')

        if self.fit_params_ is None:
            raise RuntimeError('You should fit before to transform.')

        # Apply the shifting on the ppm grid
        for y in range(modality.bandwidth_ppm.shape[1]):
            for x in range(modality.bandwidth_ppm.shape[2]):
                for z in range(modality.bandwidth_ppm.shape[3]):
                    # Compute the shift to apply
                    shift_ppm = (self.fit_params_[y, x, z] -
                                 PPM_REFERENCE['water'])
                    # Apply this shift to all the bandwidth
                    modality.bandwidth_ppm[:, y, x, z] -= shift_ppm

        return modality

    @staticmethod
    def load_from_pickles(filename):
        """ Function to load a normalization object.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Returns
        -------
        bpp : object
            Returns the loaded object.

        """
        # Check the consistency of the filename
        filename = check_filename_pickle_load(filename)
        # Load the pickle
        bpp = pickle.load(open(filename, 'rb'))

        return bpp

    def save_to_pickles(self, filename):
        """ Function to save a normalizatio object using pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Returns
        -------
        None

        """
        # We need to check that the directory where the file will be exist
        dir_pickle = os.path.dirname(filename)
        if not os.path.exists(dir_pickle):
            os.makedirs(dir_pickle)
        # Check the consistency of the filename
        filename = check_filename_pickle_save(filename)
        # Create the pickle file
        pickle.dump(self, open(filename, 'wb'))

        return None
