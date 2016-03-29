""" Gaussian normalization to normalize standalone modality.
"""

from .standalone_normalization import StandaloneNormalization


class GaussianNormalization(StandaloneNormalization):
    """ Gaussian normalization to normalize standalone modality.

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
            if params not in valid_presets:
                raise ValueError('You need to specify to arguments mu and'
                                 ' sigma for the mean and std, respectively.')
            else:
                self.fit_params_ = {}
                for k in valid_presets:
                    # Incrementaly add the key and values
                    self.fit_params_[k] = params[k]
        else:
            raise ValueError('The type of the object params does not fulfill'
                             ' any requirement.')
        # Initialize the fitting boolean
        self.is_fitted_ = False

    def fit(self, modality, ground_truth=None, cat=None):
        """ Method to find the parameters needed to apply the
        normalization.

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
        super(GaussianNormalization, self).fit(modality=modality,
                                               ground_truth=ground_truth,
                                               cat=cat)

        # Check if we need to find the initial parameters or they are given
        # by the user
        if self.params == 'auto':
            # Find an original guess of the fitted parameters depending of
            # the data
            mu = np.ndarray.mean(modality.data_)
            sigma = 1
        elif self.params == 'fixed':
            mu = self.fit_params_['mu']
            sigma = self.fit_params_['sigma']
        else:
            raise ValueError('The value of self.params is unknown. Something'
                             ' went wrong.')

        return self
