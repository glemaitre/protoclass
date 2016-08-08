"""Semi quantification extraction from temporal modality."""
from __future__ import division

import numpy as np

from joblib import Parallel, delayed

from scipy.optimize import curve_fit
from scipy.optimize import simps

from .temporal_extraction import TemporalExtraction

from .tofts_quantification_extraction import ToftsQuantificationExtraction
from ..utils.validation import check_modality


K_MAD = 1.4826


# def _huisman_model(t, a0=2., r=1., beta=1.):
#     """Huisman model to compute the concentration.

#     Paramters
#     ---------
#     t : ndarray, shape (n_serie, )
#         The time array.

#     A, B : float, optional (default=1.1)
#         Variables for the modeling.

#     Returns
#     -------
#     s_t : ndarray, shape (n_serie, )

#     """

#     # Transform the time into minutes
#     t = t / 60.

#     # Define the signal
#     s_t = np.exp(r * t + (1. / beta) * (a0 - r)(np.exp(beta * t) - 1))

#     return s_t


# def res_std_dev(model, estimate):
#     """Compute the residual standard deviation.

#     Parameters
#     ----------
#     model : array-like, shape (n_sample, )
#          Value used to made the fitting.

#     estimate : array-like, shape (n_sample, )
#          Value obtained by fitting.

#     Returns
#     -------
#     residual : float
#         Residual standard deviation.

#     """

#     if model.shape != estimate.shape:
#         raise ValueError('The model and estimate arrays should have'
#                          ' the same size.')

#     return np.sqrt(np.sum((model - estimate) ** 2) /
#                    (float(model.size) - 2.))


# def _fit_huisman_model(t_mod, s_t, start_enh, M=8, ngh_sz=3):
#     """Private function to fit concentration to extended PUN model.

#     Parameters
#     ----------
#     t_mod : ndarray, shape (n_serie, )
#         The time associated to the concentration kinetic.

#     s_t : ndarray, shape (n_serie, )
#         Signal of the pixel to be fitted.

#     start_enh : int
#         The index when the enhancement start.

#     M : int, optional (default=8)
#         The number of level to use to compute the most probable steepest slope.
#         Should be a multiple of 2.

#     ngh_sz : int, optional (default=3)
#         Number of neighbours to check around to find a new maximum.

#     Returns
#     -------
#     param : dict of str: float
#         The A, kep, and kel parameters.

#     """

#     if not M % 2 == 0:
#         raise ValueError('M should be a multiple of 2.')

#     # Compute the derivative at the different scale
#     scale_list = [M // i for i in range(1, M // 2) if M // i > 1]
#     s_prime_scale = [
#         SemiQuantificationExtraction._derivative_sliding_window(s_t, t_mod, m)
#         for m in scale_list]
#     # Compute the std of the original signal noise
#     std_noise_s = SemiQuantificationExtraction._estimate_noise_std(s_t,
#                                                                    start_enh)
#     # Compute the estimate of the std noise of the slope
#     sigma_prime_scale = [
#         SemiQuantificationExtraction._estimate_slope_std(std_noise_s, m,
#                                                          t_mod[1] - t_mod[0])
#         for m in scale_list]

#     # Make an iterative search to find t0 and m
#     slope_max_idx = []
#     # Find the intial maximum at coarser scale
#     sigma_thresh = 3 * sigma_prime_scale[0]
#     # Find the index of the interesting samples
#     idx_candidates = np.flatnonzero(s_prime_scale[0] > sigma_thresh)
#     # Find the maximum
#     slope_max_idx.append(
#         idx_candidates[sigma_prime_scale[0][idx_candidates].argmax()])
#     find_max = True
#     itr_slope = 1
#     while find_max or itr_slope > s_prime_scale.size:
#         # Find a maximum next to the previous one
#         # Compute the lower bound
#         low_bound = slope_max_idx[-1] - ngh_sz
#         if low_bound < 0:
#             low_bound = 0
#         high_bound = slope_max_idx[-1] + ngh_sz
#         if (s_prime_scale[itr_slope].max() >
#             sigma_prime_scale[itr_slope - 1][slope_max_idx] +
#             (2. * sigma_prime_scale[itr_slope])):
#             # Add the index of the new max in the list
#             slope_max_idx.append(s_prime_scale[itr_slope].argmax())
#             # Increment the number of iteration
#             itr_slope += 1
#         else:
#             # If we don't find any maximum, let's stop
#             find_max = False

#     # Let's compute S0
#     S0 = np.mean(s_t[:slope_max_idx[-1]])
#     # Let's fit the plateau
#     idx_start_plateau = (slope_max_idx[-1] +
#                          2 * scale_list[len(slope_max_idx) - 1])
#     # Make a line fitting on the plateau
#     poly_coeff = np.polyfit(t_mod[idx_start_plateau:],
#                             s_t[idx_start_plateau:], 1)
#     p = np.poly1d(poly_coeff)
#     # If the slop is significant
#     thresh_slope = .5
#     if p[0] > thresh_slope:
#         # Compute the std of the residual
#         std_err = res_std_dev(s_t[idx_start_plateau:],
#                               p(t_mod[idx_start_plateau:]))
#         # Find the first point which enter in the confidence level
#         idx_cand_tm = (np.flatnonzero(
#             np.abs(s_t[slope_max_idx[-1]] -
#                    p(t_mod[slope_max_idx[-1]])) < 2. * std_err)[0] +
#                        slope_max_idx[-1])
#         # Compute the wash_out and Sm
#         Sm = s_t[idx_cand_tm]
#         wash_out = p[0]
#     else:
#         wash_out = 0
#         Sm = np.mean(s_t[idx_start_plateau:])

#     # Return the parameters Ktrans, ve, vp
#     return {'a0': popt[0], 'r': popt[1], 'beta': popt[2]}

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def _piecewise_exp_linear_model(t, t0, S0, tau, Sm, wash_out):
    """Piecewise exponential linear model for enhancement signal.

    Parameters
    ----------
    t : ndarray, shape (n_serie, )
        The time array.

    t0 : int,
        Index of the time of start of enhancement.

    S0 : float,
        Baseline intensity.

    tau : float,
        Charing time constant.

    Sm : float,
        Maximum enhanced intensity.

    wash_out : float,
        Wash-out on the curve.

    Returns
    -------
    s_t : ndarray, shape (n_series, )
        The enhancement signal.

    """

    s_t = np.zeros(t.shape)

    # Compute the signal from 0 to t0
    s_t[:t0] = S0
    # Compute the signal between t0 and 2 * tau
    idx_tau = find_nearest(t, tau * 2.)
    s_t[t0:idx_tau] = Sm - (Sm - S0) * np.exp(- (t[t0:idx_tau] - t[t0]) / tau)
    # Compute the signal between 2 * tau to the end
    s_t[idx_tau:] = (Sm -
                     (Sm - S0) * np.exp(- (t[idx_tau] - t[t0]) / tau) +
                     wash_out * (t[idx_tau:] - t[t0] - t[idx_tau]))

    return s_t


def _fit_piecewise_model(t_mod, s_t, t0, S0, init_params):
    """Private function to fit piecewise exponential linear model
    for enhancement signal.

    Parameters
    ----------
    t_mod : ndarray, shape (n_serie, )
        The time array.

    s_t : ndarray, shape(n_serie, )
        Original signal to fit.

    t0 : int,
        Index of the time of start of enhancement.

    S0 : float,
        Baseline intensity.

    Returns
    -------
    s_t : ndarray, shape (n_series, )
        The enhancement signal.

    """

    # Define the default parameters in case the fitting fail.
    popt_default = [-1, -1, -1]

    def fit_func(t, tau, Sm, wash_out):
        return _piecewise_exp_linear_model(t, t0, S0, tau, Sm, wash_out)

    # Perform the curve fitting
    param_bounds = ([t_mod[t0], 0, -np.inf], [t_mod[-1], s_t.max(), np.inf])
    try:
        popt, _ = curve_fit(fit_func, t_mod,
                            s_t, p0=init_params,
                            bounds=param_bounds)
    except RuntimeError:
        popt = popt_default

    # Return the parameters Ktrans, ve, vp
    return {'tau': popt[0], 'Sm': popt[1], 'wash-out': popt[2]}


class SemiQuantificationExtraction(TemporalExtraction):
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
        super(SemiQuantificationExtraction, self).__init__(base_modality)
        self.random_state = random_state

    # @staticmethod
    # def _derivative_sliding_window(s_t, t, win_sz):
    #     """Estimate of the derivate of a signal using a sliding window.

    #     Parameters
    #     ----------
    #     s_t : ndarray, shape (nsamples, )
    #         The signal from which the derivative should be estimated.

    #     t : ndarray, shape (n_samples, )
    #         The time associated with the signal `s_t`.

    #     win_sz: int,
    #         The size of the sliding window used to estimate the slope.

    #     Returns
    #     -------
    #     s_prime : ndarray, (n_samples, )
    #         The signal derivated with zero padding

    #     """

    #     # Allocation of the output
    #     s_prime = np.zeros(s_t.shape)

    #     # Compute the derivative by iteratively fitting a line to the samples
    #     for t_i in range(s_prime.size - win_sz):
    #         poly_coeff = np.polyfit(t[t_i:t_i + win_sz],
    #                                 s_t[t_i:t_i + win_sz],
    #                                 1)
    #         # Get the higest order coefficient
    #         s_prime[t_i] = poly_coeff[0]

    #     return s_prime

    # @staticmethod
    # def _estimate_noise_std(s_t, start_enh):
    #     """Estimate the noise standard deviation in a signal using
    #     median absolute deviation.

    #     Parameters
    #     ----------
    #     s_t : ndarray, shape (n_samples, )
    #         The signal from which the noise standard deviation needs to
    #         be estimated.

    #     start_enh : int,
    #         The indication about when the enhancement start. It will be used to
    #         extract the baseline and estimate the noise standard deivation on
    #         this portion.

    #     Returns
    #     -------
    #     sigma : float,
    #         The standard deviation estimate of the noise.

    #     """

    #     # Compute the median absolute deviation
    #     mad = np.median(np.abs(s_t[:start_enh] - np.median(s_t[:start_enh])))

    #     return K_MAD * mad

    # def _estimate_slope_std(std_s, win_sz, time_int):
    #     """Compute the standard deviation of the slope estimate at a
    #     given scale.

    #     Parameters
    #     ----------
    #     std_s : float,
    #         Standard deviation of the signal noise.

    #     win_sz : int,
    #         Number of samples used to compute the derivative.

    #     time_int : float,
    #         The time interval between two samples.

    #     Returns
    #     -------
    #     sigma_prime : float,
    #         Standard deviation of the slope estimate at a given scale.

    #     """

    #     return np.sqrt((12. * std_s) /
    #                    (np.power(win_sz, 3) - win_sz) * time_int)

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
        super(SemiQuantificationExtraction, self).fit(
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
            The feature will be the following (wash-in, wash-out, AUC, tau,
            relative enhancement)

        """
        super(SemiQuantificationExtraction, self).transform(
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
        # Compute the initial guess
        # Find the maximum sig

        # Perform the fitting in parallel
        idx_split = 15
        init_params = []
        for curve in signal_dce:
            # Initialize tau to t0
            param1 = modality.time_info_[self.start_enh_]
            # Initialize Sm to the maximum enhancement
            param2 = curve[:idx_split].max()
            # Initialize the wash-out using the slop of the end signal
            param3 = np.polyfit(modality.time_info_[idx_split:],
                        curve[idx_split:], 1)[0]
            init_params.append([param1, param2, param3])

        pp = Parallel(n_jobs=-1)(delayed(_fit_piecewise_model)(
            modality.time_info_,
            curve,
            self.start_enh_,
            np.mean(curve[:self.start_enh_]),
            coef0)
                                 for curve, coef0 in zip(signal_dce,
                                                         init_params))

        data = np.zeros((n_sample, 5))
        for i, curve in zip(signal_dce, range(len(pp))):
            # Extract the four semi-quantitative parameters
            # 1. Wash-in
            data[i, 0] = ((pp[i]['Sm'] - np.mean(curve[:self.start_enh_])) /
                          (pp[i]['tau'] -
                           modality.time_info_[self.start_enh_]))

            # 2. Wash-out
            data[i, 1] = pp[i]['wash-out']

            # 3. IAUC
            data[i, 2] = simps(curve[self.start_enh_:],
                               modality.time_info_[self.start_enh_])

            # 4. tau as the time enhancement
            data[i, 3] = pp[i]['tau']

            # 5. Relative enhancment
            data[i, 4] = pp[i]['Sm'] - np.mean(curve[:self.start_enh_])

        return data
