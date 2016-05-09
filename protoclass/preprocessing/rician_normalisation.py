"""Rician normalization to normalize standalone modality."""

import warnings

import numpy as np

from scipy.stats import norm
from scipy.stats import rice
from scipy.optimize import curve_fit

from .standalone_normalization import StandaloneNormalization


class RicianNormalization(StandaloneNormalization):
    """Rician normalization to normalize standalone modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    params : str or dict of str: float, optional (default='auto')
        The initial estimation of the parameters:

        - If 'auto', then the standard deviation and mean will be estimated
        from the data.
        - If dict, a dictionary with the keys 'off' and 'sigma' should be
        specified. The corresponding value of these parameters should be
        float. They will be the initial value during fitting.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    fit_params_ : dict of str: float
        There is the following keys:

        - 'b' is the fitted b.
        - 'off' is the fitted offset.
        - 'sigma' is the standard deviation.

    is_fitted_ : bool
        Boolean to know if the `fit` function has been already called.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, params='auto'):
        super(RicianNormalization, self).__init__(base_modality)
        # Check the gaussian parameters argument
        if isinstance(params, basestring):
            if params == 'auto':
                # The parameters will have to be fitted at fit the fit time
                # when the data will be available
                self.params = params
                self.fit_params_ = None
            else:
                raise ValueError('The string for the object params is'
                                 ' unknown.')
        elif isinstance(params, dict):
            self.params = 'fixed'
            # Check that b, off and sigma are inside the dictionary
            valid_presets = ('b', 'off', 'sigma')
            for val_param in valid_presets:
                if val_param not in params.keys():
                    raise ValueError('At least the parameter {} is not specify'
                                     ' in the dictionary.'.format(val_param))
            # For each key, check if this is a known parameters
            for k_param in params.keys():
                if k_param in valid_presets:
                    # The key is valid, build our dictionary
                    if isinstance(params[k_param], float):
                        self.fit_params_[k_param] = params[k_param]
                    else:
                        raise ValueError('The parameter `b`, `off` and `sigma`'
                                         ' should be some float.')
                else:
                    raise ValueError('Unknown parameter inside the dictionary.'
                                     ' `b`, `off`, and `sigma` are '
                                     'the only two solutions and need to be'
                                     ' float.')
        else:
            raise ValueError('The type of the object params does not fulfill'
                             ' any requirement.')
        # Initialize the fitting boolean
        self.is_fitted_ = False

    def _model_fit(self, x, b, off, sigma, scale):
        """Function defining the Gaussian model.

        Parameters
        ----------
        x : ndarray, shape (n_samples, )
            Array for which we have to compute the PDF.

        b : float
            Distance between the reference point and the center of the
            bivariate distribution.

        off : float
            Offset of the Rician model.

        sigma : float
            Standard deviation of the model.

        scale : float
            Scaling factor of the model.

        Returns
        -------
        pdf : ndarray, shape (n_samples)
            The associated PDF to x parametrize through b, off, sigma,
            and scale.

        """
        # The pdf with the Rice distribution should be in the interval 0-1

        return rice.pdf(x, b, off, sigma) * scale

    def fit(self, modality, ground_truth=None, cat=None):
        """Method to find the parameters needed to apply the normalization.

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
        -------
        self : object
             Return self.

        """
        super(RicianNormalization, self).fit(modality=modality,
                                             ground_truth=ground_truth,
                                             cat=cat)

        # Check if we need to find the initial parameters or they are given
        # by the user
        if self.params == 'auto':
            # Find an original guess of the fitted parameters depending of
            # the data
            b = np.ndarray.mean(modality.data_[self.roi_data_])
            off = np.ndarray.min(modality.data_[self.roi_data_])
            sigma = np.ndarray.std(modality.data_[self.roi_data_])
        elif self.params == 'fixed':
            b = self.fit_params_['b']
            off = self.fit_params_['off']
            sigma = self.fit_params_['sigma']
            scale = (np.ndarray.max(modality.data_[self.roi_data_]) /
                     norm.pdf(b, b, sigma))
        else:
            raise ValueError('The value of self.params is unknown. Something'
                             ' went wrong.')

        # Compute parameters which will use during the optimization
        scale = (np.ndarray.max(modality.data_[self.roi_data_]) /
                 norm.pdf(b, b, sigma))

        # Compute the histogram that need to be fitted
        pdf, bins = modality.get_pdf(self.roi_data_, None)
        # Compute the bins centers
        bincenters = 0.5*(bins[1:]+bins[:-1])

        # Using the Rice distribution, there is a need to be in the range [0-1]
        # Normalize the data range and the parameters
        self.max_int = bincenters[-1]
        bincenters /= self.max_int

        # Normalize the paramters
        b /= self.max_int
        off /= self.max_int
        sigma /= self.max_int

        # Fit the histogram using the model defined previously
        try:
            popt, _ = curve_fit(self._model_fit, bincenters, pdf,
                                p0=(b, off, sigma, scale))
        except RuntimeError:
            warnings.warn('Optimal parameters not found!!! The original guess'
                          ' are kept but you should check that you data follow'
                          ' a Rice distribution.')
            popt = [b, off, sigma]

        # Assign the value after convergence
        self.fit_params_ = {}
        self.fit_params_['b'] = popt[0]
        self.fit_params_['off'] = popt[1]
        self.fit_params_['sigma'] = popt[2]

        self.is_fitted_ = True

        return self

    def normalize(self, modality):
        """Method to normalize the given modality using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.

        """
        super(RicianNormalization, self).normalize(modality)

        # Check that the parameters have been fitted
        if not self.is_fitted_:
            raise ValueError('Fit the parameters previous to normalize'
                             ' the data.')

        # Normalize the data of the modality
        modality.data_ -= rice.mean(self.fit_params_['b'],
                                    self.fit_params_['off'],
                                    self.fit_params_['sigma']) * self.max_int

        modality.data_ /= rice.std(self.fit_params_['b'],
                                   self.fit_params_['off'],
                                   self.fit_params_['sigma']) * self.max_int

        # Update the histogram associated to the data
        modality.update_histogram()

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
        super(RicianNormalization, self).denormalize(modality)

        # Check that the parameters have been fitted
        if not self.is_fitted_:
            raise ValueError('Fit the parameters previous to normalize'
                             ' the data.')

        # Normalize the data of the modality
        modality.data_ *= rice.std(self.fit_params_['b'],
                                   self.fit_params_['off'],
                                   self.fit_params_['sigma']) * self.max_int

        modality.data_ += rice.mean(self.fit_params_['b'],
                                    self.fit_params_['off'],
                                    self.fit_params_['sigma']) * self.max_int

        # Update the histogram associated to the data
        modality.update_histogram()

        return modality
