"""Baseline correction class for MRSI modality."""

import os

import cPickle as pickle
import numpy as np

from joblib import Parallel, delayed

from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from statsmodels.nonparametric.smoothers_lowess import lowess

from ..data_management import MRSIModality
from ..data_management import GTModality

from ..utils.validation import check_modality_inherit
from ..utils.validation import check_filename_pickle_load
from ..utils.validation import check_filename_pickle_save
from ..utils.validation import check_modality


KNOWN_CORRECTION = ('fine', 'coarse')


def _find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def _noise_estimate_spectrum(spectrum, nb_split=20):
    """Private function to estimate the noise in a spectrum.

    Parameters
    ----------
    spectrum : ndarray, shape (n_samples)
        Spectrum from which the noise has to be estimated.

    nb_split : int, option (default=20)
        The number of regions splitting each spectrum

    Returns
    -------
    sigma : float,
        The estimate of the noise standard deviation.

    """

    # Check if we will be able to make a split
    nb_elt_out = spectrum.size % nb_split
    if nb_elt_out > 0:
        spectrum = spectrum[:-nb_elt_out]

    # Split the arrays into multiple sections
    sections = np.array(np.split(spectrum, nb_split))

    # Compute the mean and variance for each section
    mean_sec = []
    var_sec = []
    for sec in sections:
        mean_sec.append(np.mean(sec))
        var_sec.append(np.var(sec))

    out = lowess(np.array(var_sec), np.array(mean_sec),
                 frac=.9, it=0)
    mean_reg = out[:, 0]
    var_reg = out[:, 1]

    # Find the value for a zero mean intensity or the nearest to zero
    idx_null_mean = _find_nearest(mean_reg, 0.)

    return np.sqrt(var_reg[idx_null_mean])


def _find_baseline_bxr(spectrum, noise_std=None, A=None, B=None,
                       A_star=5.*10e-9, B_star=1.25, max_iter=30,
                       min_err=10e-6):
    """Private function to estimate the baseline of an MRSI spectrum.

    Parameters
    ----------
    spectrum : ndarray, shape (n_samples, )
        The spectrum from which the baseline needs to be estimated.

    noise_std : float, optional (default=None)
        An estimate of the noise standard deviation in the spectrum.

    A : float, optional (default=None)
        The smoothing factor.

    B : float, optional (default=None)
        The negative factor.

    A_star : float, optional (default=5.*10e-9)
        Constant for the smoothing factor

    B_star : float, optional (default=1.25)
        Constant for the negative penalty

    max_iter : int, optional (default=30)
        The maximum of iteration for early stopping.

    min_err : float, optional (default=10e-6)
        Norm of the difference of the baseline between each iteration.

    Returns
    -------
    baseline : ndarray, shape (n_samples, )
        The baseline which was minimizing the cost.

    """

    # Check if the spectrum is complex or real
    if np.any(np.iscomplex(spectrum)):
        spectrum = np.real(spectrum)
        print 'Keep only the real part of the spectrum'

    # Find the standard deviation of the noise in the spectrum if necessary
    if noise_std is None:
        noise_std = _noise_estimate_spectrum(spectrum)
        print 'The noise std was estimated at {}'.format(noise_std)

    # Affect A and B if needed:
    if A is None:
        A = -(spectrum.size ** 4. * A_star) / noise_std

    if B is None:
        B = -B_star / noise_std

    # Initialize the baseline using the median value of the spectrum
    baseline = np.array([np.median(spectrum)] * spectrum.size)
    prev_baseline = spectrum.copy()

    # Compute the initial error and the number of iteration
    err = np.linalg.norm(baseline - prev_baseline)
    it = 0

    # Create the vector
    M0 = np.array([-1 / A] * spectrum.size)
    # Create the Hessian matrix
    D0 = lil_matrix((spectrum.size, spectrum.size))
    D0.setdiag(np.array([2.] * spectrum.size), -2)
    D0.setdiag(np.array([-8.] * spectrum.size), -1)
    D0.setdiag(np.array([12.] * spectrum.size), 0)
    D0.setdiag(np.array([-8.] * spectrum.size), 1)
    D0.setdiag(np.array([2.] * spectrum.size), 2)
    # Change the borders
    D0[0, 0] = 2.
    D0[-1, -1] = 2.
    D0[1, 0] = -4.
    D0[0, 1] = -4.
    D0[-1, -2] = -4.
    D0[-2, -1] = -4.
    D0[1, 1] = 10.
    D0[-2, -2] = 10.

    while True:
        if it > max_iter or err < min_err:
            break

        # Update the different element
        M = M0.copy()
        D = D0.copy()
        prev_baseline = baseline.copy()

        # For each element in the spectrum compute the cost
        for ii in range(spectrum.size):
            if baseline[ii] > spectrum[ii]:
                M[ii] += 2. * B * spectrum[ii] / A
                D[ii, ii] += 2. * B / A

        D = D.tocsr()
        baseline = spsolve(D, M)
        err = np.linalg.norm(baseline - prev_baseline)

        # Increment the number of iteration
        it += 1

        print 'Iteration #{} - Error={}'.format(it, err)

    return baseline


class MRSIBaselineCorrection(object):
    """Phase correction for MRSI modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    correction : str, optional (default='fine')
        Apply 'fine' or 'coarse' correction. 'fine' is taking a specific range
        into account for the choline, spermine, creatine, and citrate.

    A_star_coarse : float, optional (default=5*10e-7)
        Smoothing constant while fitting a coarse baseline.

    B_star_coarse : float, optional (default=10)
        Negative penalty while fitting a coarse baseline.

    A_star_fine : float, optional (default=5*10e-6)
        Smoothing constant while fitting a fine baseline.

    B_star_fine : float, optional (default=10e2)
        Negative penalty while fitting a fine baseline.

    range_ppm : tuple of float, optional (default=(2.2, 3.5))
        The range of ppm to consider to fit the fine baseline.

    Attributes
    ----------
    fit_params : dict,
        The parameters fitted which is the baseline in this case.

    """

    def __init__(self, base_modality, correction='fine', A_star_coarse=5*10e-7,
                 B_star_coarse=10, A_star_fine=5*10e-6, B_star_fine=10e2,
                 range_ppm=(2.2, 3.5)):
        self.base_modality_ = check_modality_inherit(base_modality,
                                                     MRSIModality)
        self.correction = correction
        self.A_star_coarse = A_star_coarse
        self.B_star_coarse = B_star_coarse
        self.A_star_fine = A_star_fine
        self.B_star_fine = B_star_fine
        self.range_ppm = range_ppm
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

        # Check if the correction is known
        if self.correction not in KNOWN_CORRECTION:
            raise ValueError('Unknown type of correction.')

        # Apply a coarse follow by a fine baseline detection
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

        # 2. Estimate the noise for the different spectra
        noise_std = []
        for s in spectra:
            noise_std.append(_noise_estimate_spectrum(np.real(s)))
        # Convert to a numpy array
        noise_std = np.array(noise_std)
        # Remove the NaN if the estimate failed
        noise_std = noise_std[np.logical_not(np.isnan(noise_std))]
        # Ge the mean of noise standard deviation
        noise_std = np.mean(noise_std)
        print ' The estimate noise standard deviation is: {}'.format(noise_std)

        # 3. Get the coarse baseline for each spectrum
        coarse_baseline = Parallel(n_jobs=-1)(delayed(_find_baseline_bxr)(
            s, noise_std=noise_std,
            A_star=self.A_star_coarse, B_star=self.B_star_coarse)
                                              for s in spectra)
        # Convert to a numpy array
        coarse_baseline = np.array(coarse_baseline)

        # 4. Get the list of index to consider for the fine baseline detection
        if self.correction == 'fine':
            idx_int = [np.flatnonzero(np.bitwise_and(p > self.range_ppm[0],
                                                     p < self.range_ppm[1]))
                       for p in ppm]

            # 5. Extrac the sub spectra and ppm
            sub_spectra = np.array([s[ii] for s, ii in zip(spectra, idx_int)])
            sub_ppm = np.array([p[ii] for p, ii in zip(ppm, idx_int)])

            # 6. Get the fine baseline for each spectrum
            fine_baseline = Parallel(n_jobs=-1)(delayed(_find_baseline_bxr)(
                s, noise_std=noise_std,
                A_star=self.A_star_fine, B_star=self.B_star_fine)
                                                for s in sub_spectra)
            # Convert to a numpy array
            fine_baseline = np.array(fine_baseline)

        # 7. Merge both baseline
        baseline = coarse_baseline
        if self.correction == 'fine':
            for ii in range(fine_baseline.shape[0]):
                baseline[ii, idx_int[ii]] = fine_baseline[ii]

        self.fit_params_ = np.reshape(baseline.T,
                                      (modality.bandwidth_ppm.shape[0],
                                       modality.bandwidth_ppm.shape[1],
                                       modality.bandwidth_ppm.shape[2],
                                       modality.bandwidth_ppm.shape[3]))

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
                    # Subtract the baseline
                    modality.data_[:, y, x, z] = ((
                        np.real(modality.data_[:, y, x, z]) -
                        self.fit_params_[:, y, x, z]) +(
                            1j * np.imag(modality.data_[:, y, x, z])))

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
