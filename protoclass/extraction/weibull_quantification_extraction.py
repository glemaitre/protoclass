"""Weibull quantification extraction from temporal modality."""
from __future__ import division

import numpy as np

from joblib import Parallel, delayed

from scipy.optimize import curve_fit

from .temporal_extraction import TemporalExtraction

from .tofts_quantification_extraction import ToftsQuantificationExtraction
from ..utils.validation import check_modality


def _weibull_model(t, A=1., B=1.):
    """Weibull model to compute the concentration.

    Paramters
    ---------
    t : ndarray, shape (n_serie, )
        The time array.

    A, B : float, optional (default=1.1)
        Variables for the modeling.

    Returns
    -------
    s_t : ndarray, shape (n_serie, )

    """

    # Transform the time into minutes
    t = t / 60.

    # Define the signal
    s_t = A * t * np.exp(np.power(-t, B))

    return s_t


def _fit_weibull_model(t_mod, s_t, S0, init_params):
    """Private function to fit concentration to extended Weibull model.

    Parameters
    ----------
    t_mod : ndarray, shape (n_serie, )
        The time associated to the concentration kinetic.

    s_t : ndarray, shape (n_serie, )
        Signal of the pixel to be fitted.

    S0 : float,
        The value of the baseline to normalize `s_t`.

    init_param : list of float,
        The initial parameters for A, kep, and kel.

    Returns
    -------
    param : dict of str: float
        The A, kep, and kel parameters.

    """

    def fit_func(t, A, B): return _weibull_model(t, A=A, B=B)

    if S0 < 1.:
        S0 = 1.

    # Define the default parameters in case the fitting fail.
    popt_default = [-1, -1]

    # Perform the curve fitting
    try:
        popt, _ = curve_fit(fit_func,
                            t_mod,
                            s_t / S0,
                            p0=init_params)
    except RuntimeError:
        popt = popt_default

    # Return the parameters Ktrans, ve, vp
    return {'A': popt[0], 'B': popt[1]}


class WeibullQuantificationExtraction(TemporalExtraction):
    """Enhancement signal extraction from temporal modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    Attributes
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, random_state=None):
        super(WeibullQuantificationExtraction, self).__init__(base_modality)
        self.random_state = random_state

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
        super(WeibullQuantificationExtraction, self).fit(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # We need to detect the AIF in order to find the value of the starting
        # of the enhancement and subsquently the baseline intensity
        aif_signal = ToftsQuantificationExtraction.compute_aif(
            modality,
            random_state=self.random_state)

        # Find the index to consider for the pre-contrast
        # - Find the index corresponfing to the maximum of the first derivative
        # of the AIF signal.
        # - Find the index related to the maximum of the second derivate
        # considering the AIF signal from the start to the previous
        # found index.
        shift_idx = 2
        idx_st_dev = np.diff(aif_signal)[2:].argmax() + shift_idx
        self.start_enh_ = np.diff(np.diff(aif_signal))[
            shift_idx:idx_st_dev].argmax() + shift_idx

        return self

    def transform(self, modality, ground_truth=None, cat=None):
        """Extract the data from the given modality.

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

        Returns
        ------
        data : ndarray, shape (n_sample, n_feature)
            A matrix containing the features extracted. The number of samples
            is equal to the number of positive label in the ground-truth.
            The feature will be the following (A, B)

        """
        super(WeibullQuantificationExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Organise that data such that we will compute the Toft's parameter
        # for each entry
        # Convert the roi to a numpy array
        roi_data = np.array(self.roi_data_)

        # Check the number of samples which will be extracted
        n_sample = roi_data.shape[1]
        # Check the number of dimension
        n_dimension = modality.n_serie_

        # Allocate the array
        signal_dce = np.empty((n_sample, n_dimension))

        # Copy the data at the right place
        for idx_sample in range(n_sample):
            # Get the coordinate of the point to consider
            coord = roi_data[:, idx_sample]

            # Extract the data
            signal_dce[idx_sample, :] = modality.data_[:,
                                                       coord[0],
                                                       coord[1],
                                                       coord[2]]

        print 'DCE signal of interest extracted: {}'.format(signal_dce.shape)

        # Define default parameter
        coef0 = [5., .6]

        # Perform the fitting in parallel
        pp = Parallel(n_jobs=-1)(delayed(_fit_weibull_model)(
            modality.time_info_,
            curve,
            np.mean(curve[:self.start_enh_]),
            coef0)
                                 for curve in signal_dce)

        # Convert the output to an numpy array
        param_kwd = ('A', 'B')

        # Allocate the data matrix
        data = np.zeros((len(pp), len(param_kwd)))

        for idx_key, key in enumerate(param_kwd):
            data[:, idx_key] = np.ravel([pp[i][key]
                                         for i in range(len(pp))])

        return data
