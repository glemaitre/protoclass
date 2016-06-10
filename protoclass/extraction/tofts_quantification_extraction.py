"""Toft quantification extraction from temporal modality."""

import numpy as np

from joblib import Parallel, delayed

from scipy.integrate import simps
from scipy.optimize import curve_fit

from skimage.measure import label
from skimage.measure import regionprops

from sklearn.cluster import KMeans

from .temporal_extraction import TemporalExtraction

from ..data_management import DCEModality
from ..utils.validation import check_modality


def _regular_tofts(t, cp_t, Ktrans=0.1, ve=0.2):
    """Regular Tofts model to compute the concentration.

    Paramters
    ---------
    t : ndarray, shape (n_serie, )
        The time array.

    cp_t : ndarray, shape (n_serie, )
        The plasma concentration.

    Ktrans : float, optional (default=0.1)
        Transfer constantB.

    ve : float, optional (default=0.2)
        Fractional volume of the extravascular extracellular space.

    Returns
    -------
    ct_t : ndarray, shape (n_serie, )

    Notes
    -----
    The model is defined in [1]_.

    References
    ----------
    .. [1] Tofts, P.S., Brix, G., Buckley, D.L., Evelhoch, J.L., Henderson, E.,
       Knopp, M.V., Larsson, H.B., Lee, T.Y., Mayr, N.A., Parker, G.J. and
       Port, R.E. (1999). Estimating kinetic parameters from dynamic
       contrast-enhanced T 1-weighted MRI of a diffusable tracer: standardized
       quantities and symbols. Journal of Magnetic Resonance Imaging, 10(3),
       223-232.

    """

    # Pre-allocation
    ct_t = np.zeros(cp_t.shape)

    # Transform the time into minutes
    t = t / 60.

    for k in range(t.size):
        ct_t[k] = simps(np.exp(-Ktrans * (t[k] - t[:k+1]) / ve) *
                        cp_t[:k+1], t[:k+1]) * Ktrans

    return ct_t


def _fit_regular_tofts(t_mod, ct_t, cp_t, init_params):
    """Private function to fit concentration to regular Tofts model.

    Parameters
    ----------
    t_mod : ndarray, shape (n_serie, )
        The time associated to the concentration kinetic.

    ct_t : ndarray, shape (n_serie, )
        Concentration of the pixel to be fitted.

    cp_t : ndarray, shape (n_serie, )
        Concentration of the AIF.

    init_param : list of float,
        The initial parameters for Ktrans, ve, and vp.

    Returns
    -------
    param : dict of str: float
        The Ktrans, ve, and vp parameters.

    """

    # Define the default parameters in case the fitting fail.
    popt_default = [-1, -1]

    # Define the function to use
    def fit_func(t, Ktrans, ve): return _regular_tofts(t, cp_t, Ktrans=Ktrans,
                                                       ve=ve)

    # Perform the curve fitting
    try:
        popt, _ = curve_fit(fit_func, t_mod,
                            ct_t, p0=init_params)
    except RuntimeError:
        popt = popt_default

    # Return the parameters Ktrans, ve, vp
    return {'Ktrans': popt[0], 've': popt[1]}


def _extended_tofts(t, cp_t, Ktrans=0.1, ve=0.2, vp=0.1):
    """Extended Tofts model to compute the concentration.

    Paramters
    ---------
    t : ndarray, shape (n_serie, )
        The time array.

    cp_t : ndarray, shape (n_serie, )
        The plasma concentration.

    Ktrans : float, optional (default=0.1)
        Transfer constantB.

    ve : float, optional (default=0.2)
        Fractional volume of the extravascular extracellular space.

    vp : float, optional (default=0.1)
        Fractional volume of the intravascular space.

    Returns
    -------
    ct_t : ndarray, shape (n_serie, )

    Notes
    -----
    The model is defined in [1]_.

    References
    ----------
    .. [1] Tofts, P. S. (1997). Modeling tracer kinetics in dynamic Gd-DTPA
       MR imaging. Journal of Magnetic Resonance Imaging, 7(1), 91-101.

    """

    # Pre-allocation
    ct_t = np.zeros(cp_t.shape)

    # Transform the time into minutes
    t = t / 60.

    for k in range(t.size):
        ct_t[k] = simps(np.exp(-Ktrans * (t[k] - t[:k+1]) / ve) *
                        cp_t[:k+1], t[:k+1]) * Ktrans + vp * cp_t[k]
    return ct_t


def _fit_extended_tofts(t_mod, ct_t, cp_t, init_params):
    """Private function to fit concentration to extended Tofts model.

    Parameters
    ----------
    t_mod : ndarray, shape (n_serie, )
        The time associated to the concentration kinetic.

    ct_t : ndarray, shape (n_serie, )
        Concentration of the pixel to be fitted.

    cp_t : ndarray, shape (n_serie, )
        Concentration of the AIF.

    init_param : list of float,
        The initial parameters for Ktrans, ve, and vp.

    Returns
    -------
    param : dict of str: float
        The Ktrans, ve, and vp parameters.

    """

    # Define the default parameters in case the fitting fail.
    popt_default = [-1, -1, -1]

    # Define the function to use
    def fit_func(t, Ktrans, ve, vp): return _extended_tofts(t, cp_t,
                                                            Ktrans=Ktrans,
                                                            ve=ve, vp=vp)

    # Perform the curve fitting
    try:
        popt, _ = curve_fit(fit_func, t_mod,
                            ct_t, p0=init_params)
    except RuntimeError:
        popt = popt_default

    # Return the parameters Ktrans, ve, vp
    return {'Ktrans': popt[0], 've': popt[1], 'vp': popt[2]}


class ToftsQuantificationExtraction(TemporalExtraction):
    """Enhancement signal extraction from temporal modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    T10 : float
        Relaxation time without contrast agent. Perfectly it should be given
        using a T1 map. We do not have such data for the moment. The unit is s.

    r1 : float
        Relaxivity of the contrast agent. The unit is mmol.L.s^{-1}.

    hematocrit : float, optional (default=0.42)
        Hematrocrit level in percentage.

    Attributes
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    T10_ : float
        Relaxation time without contrast agent. Perfectly it should be given
        using a T1 map. We do not have such data for the moment. The unit is s.

    r1_ : float
        Relaxivity of the contrast agent. The unit is mmol.L.s^{-1}.

    hematocrit_ : float
        Hematrocrit level in percentage.

    TR_ : float
        Repetition time in s units.

    flip_angle : float
        Flip angle in deg units.

    cb_t_ : ndarray, shape (n_series, )
        The blood concentration associated with the AIF.

    cp_t_ : ndarray, shape (n_series, )
        The plasma concentration associated with the AIF.

    start_enh_ : int
        The time index from which the contrast appear in the blood.

    """

    def __init__(self, base_modality, T10, r1, hematocrit=0.42,
                 random_state=None):
        super(ToftsQuantificationExtraction, self).__init__(base_modality)

        self.T10_ = T10
        self.r1_ = r1
        self.hematocrit_ = hematocrit
        self.TR_ = None
        self.flip_angle_ = None
        self.cb_t_ = None
        self.cp_t_ = None
        self.ct_t_ = None
        self.random_state = random_state

    @staticmethod
    def compute_aif(dce_modality, n_clusters=6, eccentricity=.5,
                    diameter=(10., 20.), area=(100., 400.),
                    thres_sel=0.9, estimator='median',
                    random_state=None):
        """Determine the AIF by segmenting the aorta in the kinetic sequence.

        Parameters
        ----------
        dce_modality : object DCEModality
            The modality to use to compute the AIF, signal.

        n_clusters : int, optional (default=6)
            The number of clusters to use to make the detection of the zone
            of interest to later segment the aorta or veins.

        eccentricity : float, optional (default=.5)
            The eccentricity is the ratio of the focal distance
            (distance between focal points) over the major axis length. The
            value is in the interval [0, 1). When it is 0, the ellipse becomes
            a circle. Greater is more permissive and find more regions of
            interest.

        diameter : tuple of float, optional (default=(10., 20.))
            Tuple of the minimum and maximum value of the diameters of the
            region. The region having a diameter included in this interval
            will be kept as potential region.

        area : tuple of float, optional (default=(100., 400.))
            Tuple of the minimum and maximum area in between which the region
            of interest will be kept.

        thres_sel : float, optional (default=0.9)
            For each region detected only the voxels with an enhancement
            greater than this threshold will be kept to compute the final AIF.
            The value should be in the range [0., 1.].

        estimator : str, optional (default='median')
            The estimator used to estimate the AIF from the segmented region.
            Can be the following: 'median', 'max', and 'mean'

        random_state : integer or numpy.RandomState, optional (default=None)
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        aif : ndarray, shape (n_series, )
            The estimated AIF signal converted into mMol.

        Notes
        -----
        The selection of the most enhanced voxels is done to address the
        problem of partial volume effects. The proposed thresholding is taken
        from [1]_.

        References
        ----------
        .. [1] Schabel, Matthias C., and Dennis L. Parker. "Uncertainty and
           bias in contrast concentration measurements using spoiled gradient
           echo pulse sequences." Physics in medicine and biology 53.9
           (2008): 2345.

        """
        # Check that the modality provided is from the good class
        check_modality(dce_modality, DCEModality())

        # Check that the parameters have acceptable values
        if eccentricity > 1 or eccentricity < 0:
            raise ValueError('Check the value of the parameters'
                             ' `eccentricity`.')
        if thres_sel > 1 or thres_sel < 0:
            raise ValueError('Check the value of the parameters `thres_sel`.')

        # Check that the data have been read
        if dce_modality.data_ is None:
            raise RuntimeError('Read the data first.')

        # Check the type of estimator
        choice_est = ('median', 'mean', 'max')
        if estimator not in choice_est:
            raise ValueError('Wrong type of estimator choosen')

        # Get the size of the volume
        sz_vol = dce_modality.metadata_['size']

        # For each slice
        signal_aif = np.empty((0, dce_modality.n_serie_), dtype=float)
        for idx_sl in range(sz_vol[2]):

            # Crop the upper part of the image
            org_im = dce_modality.data_[:, 50:(sz_vol[1] / 2), :, idx_sl]

            # Reshape the data to make some clustring later on
            sz_croped_im = org_im.shape
            data = np.reshape(org_im, (sz_croped_im[0],
                                       sz_croped_im[1] *
                                       sz_croped_im[2])).T

            # Make a k-means filtering
            km = KMeans(n_clusters=n_clusters,
                        n_jobs=-1,
                        random_state=random_state)
            # Fit and predict the data
            data_label = km.fit_predict(data)

            # Skip to the next iteration if we did not find any candidate
            if np.unique(data_label).size < 2:
                continue

            # Find the cluster with the highest enhancement - it will
            # correspond to blood
            cl_perc = []
            for cl in np.unique(data_label):

                # Compute the maximum enhancement of the current cluster
                # and find the 90 percentile
                perc = np.percentile(np.max(data[data_label == cl],
                                            axis=1), 90)
                cl_perc.append(perc)

            # Select only the cluster of interest
            cl_aorta = np.argmax(cl_perc)
            bin_im = np.reshape([data_label == cl_aorta], (sz_croped_im[1],
                                                           sz_croped_im[2]))
            # Transform the binary image into a labelled image
            label_im = label(bin_im.astype(int))

            # Compute the property for each region labelled
            regions = regionprops(label_im)

            # Remove the regions in the image which do not follow the
            # specificity imposed
            for idx_reg, reg in enumerate(regions):

                # Check the eccentricity
                if reg.eccentricity > eccentricity:
                    label_im[np.nonzero(label_im == idx_reg + 1)] = 0
                    continue

                # Check the diameter
                if (reg.equivalent_diameter < diameter[0] or
                        reg.equivalent_diameter > diameter[1]):
                    label_im[np.nonzero(label_im == idx_reg + 1)] = 0
                    continue

                # Check the area
                if reg.area < area[0] or reg.area > area[1]:
                    label_im[np.nonzero(label_im == idx_reg + 1)] = 0
                    continue

                # Remove the voxels which are contaminated by partial volume
                # effect
                # Find the index of the data of interest
                idx_region_data = np.nonzero(np.resize(label_im == idx_reg + 1,
                                                       (sz_croped_im[1] *
                                                        sz_croped_im[2])))
                # Extract the data of interest
                roi_data = data[idx_region_data, :]
                # Compute the maximum enhancement for these voxels
                max_enhancement = np.ravel(np.abs(np.max(roi_data, axis=1) -
                                                  roi_data[:, 0]))
                # Find the voxels index to keep which are enhanced enough
                # Put to zero the one which are not
                idx_not_aif_voxel = np.nonzero(max_enhancement <
                                               (np.max(max_enhancement) *
                                                thres_sel))
                label_im[np.unravel_index(
                    idx_region_data[0][idx_not_aif_voxel],
                    (sz_croped_im[1],
                     sz_croped_im[2]))] = 0

            # Store the signal that will be used to estimated the AIF
            if np.count_nonzero(label_im) > 0:
                signal_aif = np.vstack((signal_aif,
                                        org_im[:,
                                               np.nonzero(label_im)[0],
                                               np.nonzero(label_im)[1]].T))

        # Get the final estimate
        if estimator == 'median':
            aif = np.median(signal_aif, axis=0)
        elif estimator == 'mean':
            aif = np.mean(signal_aif, axis=0)
        elif estimator == 'max':
            aif = np.max(signal_aif, axis=0)

        return aif

    def fit(self, modality, ground_truth=None, cat=None, fit_aif=True):
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

        fit_aif : bool, optional (default=True)
            Either to estimate the AIF from the data or from a population-based
            studdy.

        Return
        ------
        self : object
             Return self.

        """
        super(ToftsQuantificationExtraction, self).fit(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Extract TR and alpha from the metadata
        # The dicom store the results in milliseconds. To be consistent save
        # it in seconds.
        self.TR_ = float(modality.metadata_['TR']) / 1000.
        self.flip_angle_ = float(modality.metadata_['flip-angle'])

        # We need to find when the enhancement of the curve will start
        # For that matter, let's start by computing the AIF t0 have all the
        # parmeters
        aif_signal = self.compute_aif(modality, random_state=self.random_state)

        # Find the index to consider for the pre-contrast
        # - Find the index corresponfing to the maximum of the first derivative
        # of the AIF signal.
        # - Find the index related to the maximum of the second derivate
        # considering the AIF signal from the start to the previous
        # found index.
        idx_st_dev = np.diff(aif_signal).argmax()
        self.start_enh_ = np.diff(np.diff(aif_signal))[:idx_st_dev].argmax()

        # Check if we compute or generate the AIF
        if fit_aif:
            self.cb_t_ = self.signal_to_conc(
                aif_signal,
                np.min(aif_signal[:self.start_enh_]))
        else:
            self.cb_t_ = self.population_based_aif(modality,
                                                   delay=self.start_enh_)

        # Define the concentration in plasma
        self.cp_t_ = self.cb_t_ / (1. - self.hematocrit_)

        return self

    def conc_to_signal(self, conc, s_pre_contrast):
        """Given the concentration compute the MRI signal using FLASH sequence.

        Parameters
        ----------
        conc : ndarray, shape (n_series, )
            Concentration in mMol.

        s_pre_contrast : float
            Signal before injection of contrast agent.

        Returns
        -------
        signal : ndarray, shape (n_series, )
            Concentration related to the signal.

        Notes
        -----
        The nonlinear approach is based on [1]_.

        References
        ----------
        .. [1] Dale, B. M., Jesberger, J. A., Lewin, J. S., Hillenbrand, C. M.,
           & Duerk, J. L. (2003). Determining and optimizing the precision of
           quantitative measurements of perfusion from dynamic contrast
           enhanced MRI. Journal of Magnetic Resonance Imaging, 18(5), 575-584.

        """
        # Check that the fitting was performed
        if self.TR_ is None:
            raise RuntimeError('You should fit the data before to try any'
                               ' conversion.')

        # Compute the relaxation
        R10 = 1. / self.T10_
        # Define the relaxation rate in the presence of contrast agent
        E1 = np.exp(-self.TR_ * (R10 + self.r1_ * conc))
        # Define the relaxation rate contrast agent free
        E10 = np.exp(-self.TR_ * R10)

        # Define the cosine of the flip angle
        flip_angle_rad = np.radians(self.flip_angle_)
        cos_alpha = np.cos(flip_angle_rad)

        # Compute the relative enhancement
        E = (((1 - E1) * (1 - E10 * cos_alpha)) /
             ((1 - E1 * cos_alpha) * (1 - E10)))

        return E * s_pre_contrast

    def signal_to_conc(self, signal, s_pre_contrast):
        """Given the MRI signal compute the concentration using FLASH sequence.

        Parameters
        ----------
        signal : ndarray, shape (n_series, )
            Signal obtained from the DCE modality.

        s_pre_contrast : float or ndarray, shape (n_series, )
            Signal before injection of contrast agent.

        Returns
        -------
        C_t : ndarray, shape (n_series, )
            Concentration related to the signal in mMol.

        Notes
        -----
        The nonlinear approach is based on [1]_.

        References
        ----------
        .. [1] Dale, B. M., Jesberger, J. A., Lewin, J. S., Hillenbrand, C. M.,
           & Duerk, J. L. (2003). Determining and optimizing the precision of
           quantitative measurements of perfusion from dynamic contrast
           enhanced MRI. Journal of Magnetic Resonance Imaging, 18(5), 575-584.

        """
        # Check that the fitting was performed
        if self.TR_ is None:
            raise RuntimeError('You should fit the data before to try any'
                               ' conversion.')


        # Compute the relative enhancement post-contrast / pre-contrast
        s_rel = signal / s_pre_contrast

        # Convert the flip angle into radians
        flip_angle_rad = np.radians(self.flip_angle_)

        # Compute the numerator
        A = (np.exp(-2. * self.TR_ / self.T10_) *
             np.cos(flip_angle_rad) * (1. - s_rel) +
             np.exp(-self.TR_ / self.T10_) *
             (s_rel * np.cos(flip_angle_rad) - 1.))
        # Compute the denominator
        B = (np.exp(-self.TR_ / self.T10_) *
             (np.cos(flip_angle_rad) - s_rel) + s_rel - 1.)

        return np.abs((1. / (self.TR_ * self.r1_)) * np.log(A / B))

    @staticmethod
    def population_based_aif(modality, A=(48.54, 19.8), T=(10.2276, 21.9),
                             sigma=(3.378, 7.92), alpha=1.050, beta=0.0028083,
                             s=0.63463, tau=28.98, delay=0):
        """Generate an AIF from a population-based AIF.

        The model is based on a mixture of 2 Guassians plus an exponential
        modulated with a sigmoid function.

        Parameters
        ----------
        A : tuple of 2 floats, optional (default=(48.54, 19.8))
            Scaling constants of the Gaussians.

        T : tuple of 2 floats, optional (default=(10.2276, 21.9))
            Center of the Gaussians.

        sigma : tuple of 2 floats, optional (default=(3.378, 7.92))
            Width of the Gaussians.

        alpha : float, optional (default=1.050)
            Amplitude of the exponential.

        beta : float, optional (default=0.0028083)
            Decay of the exponential.

        s : float, optional (default=0.63463)
            Width of the sigmoid.

        tau : float, optional (default=28.98)
            Center of the sigmoid.

        delay : int, optional (default=0)
            From which time (index) the AIF will start.

        Returns
        -------
        cb_t : ndarray, shape (n_serie, )
            Concentration in mMol of the an population-based AIF.

        Notes
        -----
        The method is based on [1]_. The default parameters have been found
        infered from 67 AIFs.

        References
        ----------
        .. [1] Parker, G.J., Roberts, C., Macdonald, A., Buonaccorsi, G.A.,
           Cheung, S., Buckley, D.L., Jackson, A., Watson, Y., Davies, K. and
           Jayson, G.C. (2006). Experimentally-derived functional form for a
           population-averaged high-temporal-resolution arterial input function
           for dynamic contrast-enhanced MRI. Magnetic resonance in medicine,
           56(5), 993-1000.

        """

        # Allocate the output
        cb_t = np.zeros((modality.n_serie_, ))
        for i in range(2):
            for idx_t in range(cb_t.size):
                cb_t[idx_t] += ((A[i] / (sigma[i] * np.sqrt(2. * np.pi))) *
                                (np.exp(-((modality.time_info_[idx_t] -
                                           T[i]) ** 2) /
                                        (2. * sigma[i] ** 2))) +
                                (alpha * np.exp(-beta *
                                                modality.time_info_[idx_t]) /
                                 (1 + np.exp(-s * (modality.time_info_[idx_t] -
                                                   tau)))))

        # Take into account the starting time
        cb_t = np.roll(cb_t, delay)
        # Put to the zero the initial value
        cb_t[:delay] = 0.

        return cb_t

    def transform(self, modality, ground_truth=None, cat=None,
                  kind='extended'):
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

        kind : string, optional (default='extended')
            Type of Tofts model to use: 'regular' or 'extended'.

        Returns
        ------
        data : ndarray, shape (n_sample, n_feature)
            A matrix containing the features extracted. The number of samples
            is equal to the number of positive label in the ground-truth.
            The feature will be the following:

            - If 'regular': 1st feature will be `Ktrans` and the 2nd will be
            `ve`.
            - If 'extended': 1st feature will be `Ktrans`, the 2nd will be
            `ve`, and the third will be `vp`.

        """
        super(ToftsQuantificationExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check the parameter kind
        param_kind = ('regular', 'extended')
        if kind not in param_kind:
            raise ValueError('Unknown parameter for kind.')

        # Check that the data have been fitted
        if self.cp_t_ is None:
            raise RuntimeError('Fit the data previous to transform them.')

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

        # Compute the concentration for each DCE signal
        # Let's do it in parallel
        # Build a matrix contraining the baseline
        pre_contrast_dce = np.min(signal_dce[:, :self.start_enh_], axis=1)
        self.ct_t_ = self.signal_to_conc(signal_dce,
                                         np.tile(pre_contrast_dce,
                                                 (signal_dce.shape[1], 1)).T)
        # Convert the possible NaN of inf
        self.ct_t_ = np.nan_to_num(self.ct_t_)

        print 'Concentration computed from the signal'

        # Fit the Tofts parameters
        if kind == 'extended':
            # Define default parameter
            coef0 = [0.01, 0.01, 0.01]

            # Perform the fitting in parallel
            pp = Parallel(n_jobs=-1)(delayed(_fit_extended_tofts)(
                modality.time_info_,
                curve,
                self.cp_t_,
                coef0)
                                     for curve in self.ct_t_)

            # Convert the output to an numpy array
            param_kwd = ('Ktrans', 've', 'vp')

            # Allocate the data matrix
            data = np.zeros((len(pp), len(param_kwd)))

            for idx_key, key in enumerate(param_kwd):
                data[:, idx_key] = np.ravel([pp[i][key]
                                             for i in range(len(pp))])

        elif kind == 'regular':
            # Define default parameter
            coef0 = [0.01, 0.01]

            # Perform the fitting in parallel
            pp = Parallel(n_jobs=-1)(delayed(_fit_regular_tofts)(
                modality.time_info_,
                curve,
                self.cp_t_,
                coef0)
                                     for curve in self.ct_t_)

            # Convert the output to an numpy array
            param_kwd = ('Ktrans', 've')

            # Allocate the data matrix
            data = np.zeros((len(pp), len(param_kwd)))

            for idx_key, key in enumerate(param_kwd):
                data[:, idx_key] = np.ravel([pp[i][key]
                                             for i in range(len(pp))])

        return data
