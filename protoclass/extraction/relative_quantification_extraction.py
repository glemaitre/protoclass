"""Relative quantification extraction from MRSI modality."""
from __future__ import division

import numpy as np

from joblib import Parallel, delayed

from lmfit import minimize
from lmfit import Parameters

from scipy.special import wofz
from scipy.stats import norm
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.linalg import norm as lnorm

from .mrsi_extraction import MRSIExtraction


PPM_REFERENCE = {'water' : 4.65, 'citrate' : 2.58}
PPM_LIMITS = {'water': (4., 6.), 'citrate' : (2.30, 2.90)}
KNOWN_NORMALIZATION = ('l2', 'l1')
KNOWN_OUTPUT = ('ratio', 'integral')


def _gaussian_profile(x, alpha, mu, sigma):
    """Private function to create a Gaussian profile.

    Parameters
    ----------
    x : ndarray, shape (n_samples, )
        The abscisses to generate the Gaussian.

    alpha : float,
        The amplitude factor.

    mu : float,
        The mean.

    sigma :
        The standard deviation.

    Returns
    -------
    y : ndarray, shape (n_samples, )
        The Gaussian profile.

    """

    return alpha * norm.pdf(x, loc=mu, scale=sigma)


def _citrate_residual(params, ppm, data):
    """ Private function which will return the citrate profile

    Parameters
    ----------
    params : Parameters,
        A lmfit structure containing the different parameters.

    ppm : ndarray, shape (n_samples, )
        The abscisse to generate the profile.

    Returns
    -------
    residuals : ndarray, shape (n_samples)

    """

    # Define the list of parameters
    alpha1 = np.abs(params['citalpha1'])
    alpha2 = np.abs(params['citalpha2'])
    alpha3 = np.abs(params['citalpha3'])
    mu1 = np.abs(params['citmu1'])
    delta2 = np.abs(params['citdelta2'])
    delta3 = np.abs(params['citdelta3'])
    sigma1 = np.abs(params['citsigma1'])
    sigma2 = np.abs(params['citsigma2'])
    sigma3 = np.abs(params['citsigma3'])

    cit_1 = _gaussian_profile(ppm, alpha1, mu1, sigma1)
    cit_2 = _gaussian_profile(ppm, alpha2, mu1 + delta2, sigma2)
    cit_3 = _gaussian_profile(ppm, alpha3, mu1 - delta3, sigma3)
    # Compute the window
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > (mu1 - 2 * sigma1),
                                             ppm < (mu1 + 2 * sigma1)))
    mask[idx_mask] = 1.
    res_1 = ((cit_1 - data) * mask) / simps(cit_1, ppm)
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > ((mu1 + delta2) -
                                                    2 * sigma2),
                                             ppm < ((mu1 + delta2) +
                                                    2 * sigma2)))
    mask[idx_mask] = 1.
    res_2 = ((cit_2 - data) * mask) / simps(cit_2, ppm)
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > ((mu1 - delta3) -
                                                    2 * sigma1),
                                             ppm < ((mu1 - delta3) +
                                                    2 * sigma1)))
    mask[idx_mask] = 1.
    res_3 = ((cit_3 - data) * mask) / simps(cit_3, ppm)

    return res_1 + res_2 + res_3


def residual_choline(params, ppm, data):
    """ Private function which will return the choline profile

    Parameters
    ----------
    params : Parameters,
        A lmfit structure containing the different parameters.

    ppm : ndarray, shape (n_samples, )
        The abscisse to generate the profile.

    Returns
    -------
    residuals : ndarray, shape (n_samples)

    """

    # Define the list of parameters
    alpha1 = params['chalpha1']
    mu1 = params['chmu1']
    delta1 = params['chdelta1']
    sigma1 = params['chsigma1']

    choline = _gaussian_profile(ppm, alpha1, mu1 + delta1, sigma1)

    # Compute the window
    mask = np.zeros(ppm.shape)
    idx_mask = np.flatnonzero(np.bitwise_and(ppm > ((mu1 + delta1) -
                                                    2 * sigma1),
                                             ppm < ((mu1 + delta1) +
                                                    2 * sigma1)))
    mask[idx_mask] = 1.
    res_4 = ((choline - data) * mask)

    return res_4


def _metabolite_fitting(ppm, spectrum):
    """Private function to fit a citrate and choline metabolites.

    Parameters
    ----------
    ppm : ndarray, shape (n_samples, )
        The ppm associated to the spectrum.

    spectrum : ndarray, (n_samples, )
        The spectrum of the metabolites

    Returns
    -------
    parameters : tuple of Parameters,
        A tuple containing the parameters for the citrate and choline model.

    """

    # Citrate fitting
    # Define the limits of the citrate
    ppm_limits = PPM_LIMITS['citrate']
    # Crop the spectrum
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    # Reinterpolate with cubic spline
    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Define the default parameters
    # Define their bounds
    mu_bounds = (2.54, 2.68)
    delta_2_bounds = (.06, .16)
    delta_3_bounds = (.06, .16)

    # Define the default shifts
    ppm_cit = np.linspace(mu_bounds[0], mu_bounds[1], num=1000)
    mu_dft = ppm_cit[np.argmax(f(ppm_cit))]

    # Redefine the maximum to avoid to much motion
    mu_bounds = (mu_dft - 0.04, mu_dft + 0.04)

    # Redefine the limit of ppm to use for the fitting
    ppm_interp = np.linspace(mu_dft - .20, mu_dft + 0.20, num=5000)
    delta_2_dft = .1
    delta_3_dft = .1

    # Define the default amplitude
    alpha_1_dft = (f(mu_dft) /
                   _gaussian_profile(0., 1., 0., .01))
    alpha_2_dft = (f(mu_dft + delta_2_dft) /
                   _gaussian_profile(0., 1., 0., .01))
    alpha_3_dft = (f(mu_dft - delta_3_dft) /
                   _gaussian_profile(0., 1., 0., .01))

    # Define the list of parameters
    params = Parameters()
    params.add('citalpha1', value=alpha_1_dft, min=0, max=100)
    params.add('citalpha2', value=alpha_2_dft, min=0, max=100)
    params.add('citalpha3', value=alpha_3_dft, min=0, max=100)
    params.add('citmu1', value=mu_dft, min=mu_bounds[0], max=mu_bounds[1])
    params.add('citdelta2', value=delta_2_dft, min=delta_2_bounds[0],
               max=delta_2_bounds[1])
    params.add('citdelta3', value=delta_3_dft, min=delta_3_bounds[0],
               max=delta_3_bounds[1])
    params.add('citsigma1', value=.01, min=.01, max=0.1)
    params.add('citsigma2', value=.01, min=.01, max=0.03)
    params.add('citsigma3', value=.01, min=.01, max=0.03)

    # Make the optimization
    res_citrate = minimize(_citrate_residual, params, args=(ppm_interp, ),
                           kws={'data' : f(ppm_interp)},
                           method='least_squares')

    # Detection of the choline
    # Redefine precisely the central value of the citrate
    mu_dft = res_citrate.params['citmu1'].value
    delta_4_bounds = (.55, .59)
    delta_4_dft = .57

    # Crop and define the limits around the choline
    ppm_limits = (mu_dft + delta_4_bounds[0] - .02,
                  mu_dft + delta_4_bounds[1] + .02)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    # Reinterpolate the data
    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Refine the maximum value
    delta_4_dft = ppm_interp[np.argmax(f(ppm_interp))] - mu_dft
    # Recrop
    ppm_limits = (mu_dft + delta_4_dft - .04,
                  mu_dft + delta_4_dft + .04)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]
    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Define the default amplitude
    alpha_4_dft = (f(mu_dft + delta_4_dft) /
                   _gaussian_profile(0., 1., 0., .01))

    # Define the list of parameters
    params = Parameters()
    params.add('chalpha1', value=alpha_4_dft, min=0.01, max=100)
    params.add('chmu1', value=mu_dft, vary=False)
    params.add('chdelta1', value=delta_4_dft, vary=False)
    params.add('chsigma1', value=.01, min=.001, max=0.02)

    # Optimize the cost function
    res_choline = minimize(residual_choline, params, args=(ppm_interp, ),
                           kws={'data' : f(ppm_interp)},
                           method='least_squares')

    return res_citrate.params, res_choline.params


def _quantification_no_fitting(ppm, spectrum):
    """Private function to qunatified citrate and choline without any fitting.

    Parameters
    ----------

     ppm : ndarray, shape (n_samples, )
        The ppm associated to the spectrum.

    spectrum : ndarray, (n_samples, )
        The spectrum of the metabolites

    Returns
    -------
    parameters : tuple of Parameters,
        A tuple containing the parameters for the citrate and choline model.

    """

    # Citrate quantification
    # Define the limits of the citrate
    ppm_limits = PPM_LIMITS['citrate']
    # Crop the spectrum
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    # Reinterpolate with cubic spline
    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')

    # Define the default parameters
    # Define their bounds
    mu_bounds = (2.54, 2.68)

    # Define the default shifts
    ppm_cit = np.linspace(mu_bounds[0], mu_bounds[1], num=1000)
    mu_dft = ppm_cit[np.argmax(f(ppm_cit))]

    # Integrate the spectrum
    ppm_interval = np.linspace(mu_dft + .18, mu_dft - .18, num=1000)
    citrate_quant = simps(f(ppm_interval), ppm_interval)

    delta_4_bounds = (.55, .59)

    # Crop and define the limits around the choline
    ppm_limits = (mu_dft + delta_4_bounds[0] - .02,
                  mu_dft + delta_4_bounds[1] + .02)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]

    # Reinterpolate the data
    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=5000)

    # Refine the maximum value
    delta_4_dft = ppm_interp[np.argmax(f(ppm_interp))] - mu_dft
    ppm_limits = (mu_dft + delta_4_dft - .04,
                  mu_dft + delta_4_dft + .04)
    idx_ppm = np.flatnonzero(np.bitwise_and(ppm > ppm_limits[0],
                                            ppm < ppm_limits[1]))
    sub_ppm = ppm[idx_ppm]
    sub_spectrum = spectrum[idx_ppm]
    f = interp1d(sub_ppm, sub_spectrum, kind='cubic')
    ppm_interp = np.linspace(sub_ppm[0], sub_ppm[-1], num=1000)
    choline_quant = simps(f(ppm_interp), ppm_interp)

    return citrate_quant, choline_quant

class RelativeQuantificationExtraction(MRSIExtraction):
    """Relative quantification extraction from MRSI modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    normalization : None or str, optional (default='l2')
        Apply a normalization or not. Choice are None, 'l2', or 'l1'.

    output : str, optional (default='ratio')
        The type of output. Either return the ratio citrate over choline using
        'ratio' or the integral citrate and choline using 'integral'.

    fitting : bool, optional (default=True)
        Quantification by fitting or just on the original signal.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, normalization='l2', output='ratio',
                 fitting=True):
        super(RelativeQuantificationExtraction, self).__init__(base_modality)
        self.normalization = normalization
        self.output = output
        self.fitting = fitting
        self.is_fitted = False
        self.data_ = None

    def _citrate_profile(self, params, ppm):
        """ Private function which will return the citrate profile

        Parameters
        ----------
        params : Parameters,
            A lmfit structure containing the different parameters.

        ppm : ndarray, shape (n_samples, )
            The abscisse to generate the profile.

        Returns
        -------
        model : ndarray, shape (n_samples, )
            The model of the citrate generated

        """
        # Define the list of parameters which are common to Gaussian and Voigt
        alpha1 = params['citalpha1']
        alpha2 = params['citalpha2']
        alpha3 = params['citalpha3']
        mu1 = params['citmu1']
        delta2 = params['citdelta2']
        delta3 = params['citdelta3']
        sigma1 = params['citsigma1']
        sigma2 = params['citsigma2']
        sigma3 = params['citsigma3']

        model = _gaussian_profile(ppm, alpha1, mu1, sigma1)
        model += _gaussian_profile(ppm, alpha2, mu1 + delta2, sigma2)
        model += _gaussian_profile(ppm, alpha3, mu1 - delta3, sigma3)

        return model

    def _choline_profile(self, params, ppm):
        """ Private function which will return the choline profile

        Parameters
        ----------
        params : Parameters,
            A lmfit structure containing the different parameters.

        ppm : ndarray, shape (n_samples, )
            The abscisse to generate the profile.

        """
        # Define the list of parameters which are common to Gaussian and Voigt
        alpha1 = params['chalpha1']
        mu1 = params['chmu1']
        delta1 = params['chdelta1']
        sigma1 = params['chsigma1']

        model = _gaussian_profile(ppm, alpha1, mu1 + delta1, sigma1)

        return model

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
        super(RelativeQuantificationExtraction, self).fit(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        if self.output not in KNOWN_OUTPUT:
            raise ValueError('The output type is unknown.')

        # In the fitting we will find the parameters needed to transform
        # the data
        # 1. Reshape all data for parallel processing
        spectra = np.real(np.reshape(modality.data_,
                                     (modality.data_.shape[0],
                                      modality.data_.shape[1] *
                                      modality.data_.shape[2] *
                                      modality.data_.shape[3])).T)

        ppm = np.reshape(modality.bandwidth_ppm, (
            modality.bandwidth_ppm.shape[0],
            modality.bandwidth_ppm.shape[1] *
            modality.bandwidth_ppm.shape[2] *
            modality.bandwidth_ppm.shape[3])).T

        if self.normalization is not None:
            if self.normalization not in KNOWN_NORMALIZATION:
                raise ValueError('Unknown normalization.')
            # Allocate the parameters array
            self.fit_params_ = np.zeros((modality.data_.shape[1] *
                                         modality.data_.shape[2] *
                                         modality.data_.shape[3]))

            for idx_s, s in enumerate(spectra):
                if self.normalization == 'l1':
                    self.fit_params_[idx_s] = lnorm(s, 1)
                if self.normalization == 'l2':
                    self.fit_params_[idx_s] = lnorm(s, 2)

        # 2. Make the fitting and get the parameters
        if self.fitting:
            self.data_ = Parallel(n_jobs=-1)(delayed(
                _metabolite_fitting)(p, s)
                                             for p, s in zip(ppm, spectra))
        else:
            self.data_ = Parallel(n_jobs=-1)(delayed(
                            _quantification_no_fitting)(p, s)
                                             for p, s in zip(ppm, spectra))

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
        super(RelativeQuantificationExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that the data have been fitted
        if not self.is_fitted:
            raise ValueError('Fit the data first.')

        if self.fitting:
            # Define a range of ppm
            ppm = np.linspace(2., 4., num=5000)

            # Compute the citrate and choline signal integral
            citrate_array = []
            choline_array = []
            for idx_fit, fit_param in enumerate(self.data_):
                # Compute the citrate signal using the previous model
                citrate = self._citrate_profile(fit_param[0], ppm)
                if self.normalization is not None:
                    citrate /= self.fit_params_[idx_fit]

                citrate_array.append(simps(citrate, ppm))

                # Compute the choline signal using the previous model
                choline = self._choline_profile(fit_param[1], ppm)
                if self.normalization is not None:
                    choline /= self.fit_params_[idx_fit]

                choline_array.append(simps(choline, ppm))
        else:
            # Compute the citrate and choline signal integral
            citrate_array = []
            choline_array = []
            for idx_fit, fit_param in enumerate(self.data_):
                # Compute the citrate signal using the previous model
                citrate = fit_param[0]
                if self.normalization is not None:
                    citrate /= self.fit_params_[idx_fit]

                citrate_array.append(citrate)

                # Compute the choline signal using the previous model
                choline = fit_param[1]
                if self.normalization is not None:
                    choline /= self.fit_params_[idx_fit]

                choline_array.append(choline)

        if self.output == 'ratio':
            # Compute the ratio citrate over choline
            data = np.array(citrate_array) / np.array(choline_array)

            # Resize the data properly according to the modality
            data = np.reshape(data, (modality.data_.shape[1],
                                     modality.data_.shape[2],
                                     modality.data_.shape[3]))

            data_res = self._resampling_as_gt(data, modality, ground_truth)
            data_res = data_res[self.roi_data_]

        elif self.output == 'integral':
            # Reshape the choline and citrate signal
            citrate = np.reshape(np.array(citrate_array),
                                 (modality.data_.shape[1],
                                  modality.data_.shape[2],
                                  modality.data_.shape[3]))

            choline = np.reshape(np.array(choline_array),
                                 (modality.data_.shape[1],
                                  modality.data_.shape[2],
                                  modality.data_.shape[3]))

            # Resample both quantity
            citrate_res = self._resampling_as_gt(citrate, modality,
                                                 ground_truth)

            choline_res = self._resampling_as_gt(choline, modality,
                                                 ground_truth)

            data_res = np.zeros((self.roi_data_[0].size, 2))

            data_res[:, 0] = citrate_res[self.roi_data_]
            data_res[:, 1] = choline_res[self.roi_data_]



        return data_res
