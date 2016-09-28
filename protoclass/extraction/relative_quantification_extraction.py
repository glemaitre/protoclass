"""Relative quantification extraction from MRSI modality."""

import numpy as np

from joblib import Parallel, delayed

from lmfit import minimize
from lmfit import Parameters

from scipy.special import wofz
from scipy.stats import norm
from scipy.integrate import simps
from scipy.interpolate import interp1d

from .mrsi_extraction import MRSIExtraction


PPM_REFERENCE = {'water' : 4.65, 'citrate' : 2.58}
PPM_LIMITS = {'water': (4., 6.), 'citrate' : (2.30, 2.90)}


#KNOWN_PROFILE = ('gaussian', 'voigt')


# def _voigt_profile(x, alpha, mu, sigma, gamma):
#     """Private function to fit a Voigt profile.

#     Parameters
#     ----------
#     x : ndarray, shape (len(x))
#         The input data.

#     alpha : float,
#         The amplitude factor.

#     mu : float,
#         The shift of the central value.

#     sigma : float,
#         sigma of the Gaussian.

#     gamma : float,
#         gamma of the Lorentzian.

#     Returns
#     -------
#     y : ndarray, shape (len(x), )
#         The Voigt profile.

#     """

#     # Define z
#     z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))

#     # Compute the Faddeva function
#     w = wofz(z)

#     return alpha * (np.real(w)) / (sigma * np.sqrt(2. * np.pi))


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
    # if profile_type == 'voigt':
    #     gamma1 = params['citgamma1']
    #     gamma2 = params['citgamma2']
    #     gamma3 = params['citgamma3']

    # if profile_type == 'gaussian':
    cit_1 = _gaussian_profile(ppm, alpha1, mu1, sigma1)
    cit_2 = _gaussian_profile(ppm, alpha2, mu1 + delta2, sigma2)
    cit_3 = _gaussian_profile(ppm, alpha3, mu1 - delta3, sigma3)
    # elif profile_type == 'voigt':
    #     cit_1 = _voigt_profile(ppm, alpha1, mu1, sigma1, gamma1)
    #     cit_2 = _voigt_profile(ppm, alpha2, mu1 + delta2, sigma2, gamma2)
    #     cit_3 = _voigt_profile(ppm, alpha3, mu1 - delta3, sigma3, gamma3)

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
    params.add('citalpha1', value=alpha_1_dft, min=0.1, max=100)
    params.add('citalpha2', value=alpha_2_dft, min=0.1, max=100)
    params.add('citalpha3', value=alpha_3_dft, min=0.1, max=100)
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
    mu_dft = res_citrate.params['mu1'].value
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


class RelativeQuantificationExtraction(MRSIExtraction):
    """Relative quantification extraction from MRSI modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    params : None or Parameters()
        Default or define a list of Parameters() for the citrate and choline.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality):
        super(RelativeQuantificationExtraction, self).__init__(base_modality)
        self.data_ = None

    # def _validate_profile(self):
    #     """Private function to check the type of profile given by
    #     the user."""

    #     if self.profile_type not in KNOWN_PROFILE:
    #         raise ValueError('Unknown profile to work with.')

    #     return None

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
        # if self.profile_type == 'voigt':
        #     gamma1 = params['citgamma1']
        #     gamma2 = params['citgamma2']
        #     gamma3 = params['citgamma3']

        # if self.profile_type == 'gaussian':
        model = _gaussian_profile(ppm, alpha1, mu1, sigma1)
        model += _gaussian_profile(ppm, alpha2, mu1 + delta2, sigma2)
        model += _gaussian_profile(ppm, alpha3, mu1 - delta3, sigma3)
        # elif self.profile_type == 'voigt':
        #     model = _voigt_profile(ppm, alpha1, mu1, sigma1, gamma1)
        #     model += _voigt_profile(ppm, alpha2, mu1 + delta2, sigma2, gamma2)
        #     model += _voigt_profile(ppm, alpha3, mu1 - delta3, sigma3, gamma3)

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
        sigma1 = params['chsigma1']
        # if self.profile_type == 'voigt':
        #     gamma1 = params['chgamma1']

        # if self.profile_type == 'gaussian':
        model = _gaussian_profile(ppm, alpha1, mu1, sigma1)
        # elif self.profile_type == 'voigt':
        #     model = _voigt_profile(ppm, alpha1, mu1, sigma1, gamma1)

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
        params_opt = Parallel(n_jobs=-1)(delayed(_metabolite_fitting)(p, s)
                                         for p, s in zip(ppm, spectra))
        # 3. Reshape the parameters array
        params_opt = np.array(params_opt)
        self.data_ = np.reshape(params_opt, (
            modality.bandwidth_ppm.shape[1],
            modality.bandwidth_ppm.shape[2],
            modality.bandwidth_ppm.shape[3]))

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

        return modality.data_[self.roi_data_]
