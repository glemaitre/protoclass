"""Gaussian normalization to normalize standalone modality."""

import numpy as np

from scipy.stats import norm
from scipy.optimize import curve_fit

from .standalone_normalization import StandaloneNormalization


class GaussianNormalization(StandaloneNormalization):
    """Gaussian normalization to normalize standalone modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    params : str or dict of str: float, optional (default='auto')
        The initial estimation of the parameters:

        - If 'auto', then the standard deviation and mean will be estimated
        from the data.
        - If dict, a dictionary with the keys 'mu' and 'sigma' should be
        specified. The corresponding value of these parameters should be
        float. They will be the initial value during fitting.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    fit_params_ : dict of str: float
        There is the following keys:

        - 'mu' is the fitted mean.
        - 'sigma' is the standard deviation.

        The precision of the parameters is the unit to avoid any precision
        problem during normalization and denormalization.

    is_fitted_ : bool
        Boolean to know if the `fit` function has been already called.

    roi_data_ : ndarray, shape ()
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, params='auto'):
        super(GaussianNormalization, self).__init__(base_modality)
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
            # Check that mu and sigma are inside the dictionary
            valid_presets = ('mu', 'sigma')
            for val_param in valid_presets:
                if val_param not in params.keys():
                    raise ValueError('At least the parameter {} is not specify'
                                      ' in the dictionary.'.format(val_param))
            # For each key, check if this is a known parameters
            self.fit_params_ = {}
            for k_param in params.keys():
                if k_param in valid_presets:
                    # The key is valid, build our dictionary
                    if isinstance(params[k_param], float):
                        self.fit_params_[k_param] = params[k_param]
                    else:
                        raise ValueError('The parameter mu and sigma should be'
                                         ' some float.')
                else:
                    raise ValueError('Unknown parameter inside the dictionary.'
                                     ' `mu` and `sigma` are the only two'
                                     ' solutions and need to be float.')
        else:
            raise ValueError('The type of the object params does not fulfill'
                             ' any requirement.')
        # Initialize the fitting boolean
        self.is_fitted_ = False

    def _model_fit(self, x, mu, sigma):
        """Function defining the Gaussian model.

        Parameters
        ----------
        x : ndarray, shape (n_samples, )
            Array for which we have to compute the PDF.

        mu : float
            Mean of the Gaussian model.

        std : float
            Standard deviation of the model.

        Returns
        -------
        pdf : ndarray, shape (n_samples)
            The associated PDF to x parametrize through mu and sigma.

        """

        return norm.pdf(x, mu, sigma)

    def _compute_histogram(self, X):
        """Function allowing to compute the histogram from the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, )
            The data of interest from which we want to estimate the PDF.

        Returns
        -------
        pdf : ndarray, shape (n_samples, )
            The PDF associated with the data of interest define by the ROI.

        bins : ndarray, shape (n_samples + 1, )
            The bins associated with the PDF.

        """
        # Compute the histogram from the data of insterest
        pdf, bins = np.histogram(X, bins=np.max(X) - np.min(X), density=True)

        return (pdf, bins)

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
        super(GaussianNormalization, self).fit(modality=modality,
                                               ground_truth=ground_truth,
                                               cat=cat)

        # Check if we need to find the initial parameters or they are given
        # by the user
        if self.params == 'auto':
            # Find an original guess of the fitted parameters depending of
            # the data
            mu = np.ndarray.mean(modality.data_[self.roi_data_])
            sigma = np.ndarray.std(modality.data_[self.roi_data_])
        elif self.params == 'fixed':
            mu = self.fit_params_['mu']
            sigma = self.fit_params_['sigma']
        else:
            raise ValueError('The value of self.params is unknown. Something'
                             ' went wrong.')

        # Compute the histogram that need to be fitted
        pdf, bins = self._compute_histogram(modality.data_[self.roi_data_])
        # Compute the bins centers
        bincenters = 0.5*(bins[1:]+bins[:-1])

        # Fit the histogram using the model defined previously
        popt, _ = curve_fit(self._model_fit, bincenters, pdf,
                            p0=(mu, sigma))

        # Assign the value after convergence
        self.mu_ = np.round(popt[0])
        self.sigma_ = np.round(popt[1])
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
        super(GaussianNormalization, self).normalize(modality)

        # Check that the parameters have been fitted
        if not self.is_fitted_:
            raise ValueError('Fir the parameters previous to normalize'
                             ' the data.')

        # Normalize the data of the modality
        modality.data_ -= self.mu_
        modality.data_ /= self.sigma_

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
        super(GaussianNormalization, self).denormalize(modality)

        # Check that the parameters have been fitted
        if not self.is_fitted_:
            raise ValueError('Fir the parameters previous to normalize'
                             ' the data.')

        # Normalize the data of the modality
        modality.data_ *= self.sigma_
        modality.data_ += self.mu_

        # Update the histogram associated to the data
        modality.update_histogram()

        return modality
