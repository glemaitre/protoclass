"""Phase congruency extraction from standalone modality."""

import numpy as np

from phasepack import phasecong
from phasepack import phasecongmono

from .standalone_extraction import StandaloneExtraction


KNOWN_FILTER = ('regular', 'monogenic')


class PhaseCongruencyExtraction(StandaloneExtraction):
    """Edge signal extraction from standalone modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    type_filter : string, optional (default='regular')
        Can be either 'regular' or 'monogenic'. Monogenic filters provide a
        speed-up for the computation

    dict_params : dict, optional (default=parameter in `phasepack`)
        A dictionary contaning the different parameters for the phase
        congruency methods. For the list of parameters, refer to phasepack.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, type_filter='regular', dict_params={}):
        super(PhaseCongruencyExtraction, self).__init__(base_modality)
        self.type_filter = type_filter
        self.dict_params = dict_params
        self.data_ = None

    def fit(self, modality, ground_truth=None, cat=None):
        """Compute the images images.

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
        super(PhaseCongruencyExtraction, self).fit(modality=modality,
                                              ground_truth=ground_truth,
                                              cat=cat)

        # Check that the filter provided is known
        if self.type_filter not in KNOWN_FILTER:
            raise ValueError('{} filter is unknown'.format(self.type_filter))

        # Regular filters
        if self.type_filter == 'regular':

            # Extract the parameters for the function
            nscale = self.dict_params.pop('n_scale', 5)
            norient = self.dict_params.pop('n_orient', 6)
            minWaveLength = self.dict_params.pop('minWaveLength', 3)
            mult = self.dict_params.pop('mult', 2.1)
            sigmaOnf = self.dict_params.pop('sigmaOnf', .55)
            k = self.dict_params.pop('k', 2.)
            cutOff = self.dict_params.pop('cutOff', .5)
            g = self.dict_params.pop('g', 10)
            noiseMethod = self.dict_params.pop('noiseMethod', -1)

            # Create a list to catch the different outputs
            M_vol = []
            ori_vol = []
            ft_vol = []

            # Compute the convolution for each slice
            for sl in range(modality.data_.shape[2]):
                M, _, ori, ft, _, _, _ = phasecong(modality.data_[:, :, sl],
                                                   nscale=nscale,
                                                   norient=norient,
                                                   minWaveLength=minWaveLength,
                                                   mult=mult,
                                                   sigmaOnf=sigmaOnf,
                                                   k=k,
                                                   cutOff=cutOff,
                                                   g=g,
                                                   noiseMethod=noiseMethod)
                # Append the data
                M_vol.append(M)
                ori_vol.append(ori)
                ft_vol.append(ft)

        # Monogenic filters
        elif self.type_filter == 'monogenic':

            # Extract the parameters for the function
            nscale = self.dict_params.pop('n_scale', 5)
            minWaveLength = self.dict_params.pop('minWaveLength', 3)
            mult = self.dict_params.pop('mult', 2.1)
            sigmaOnf = self.dict_params.pop('sigmaOnf', .55)
            k = self.dict_params.pop('k', 2.)
            cutOff = self.dict_params.pop('cutOff', .5)
            g = self.dict_params.pop('g', 10)
            noiseMethod = self.dict_params.pop('noiseMethod', -1)
            deviationGain = self.dict_params.pop('deviationGain', 1.5)

            # Create a list to catch the different outputs
            M_vol = []
            ori_vol = []
            ft_vol = []

            # Compute the convolution for each slice
            for sl in range(modality.data_.shape[2]):
                M, ori, ft, _ = phasecongmono(modality.data_[:, :, sl],
                                              nscale=nscale,
                                              minWaveLength=minWaveLength,
                                              mult=mult,
                                              sigmaOnf=sigmaOnf,
                                              k=k,
                                              cutOff=cutOff,
                                              g=g,
                                              noiseMethod=noiseMethod,
                                              deviationGain=deviationGain)
                # Append the data
                M_vol.append(M)
                ori_vol.append(ori)
                ft_vol.append(ft)

        # Convert all the list to numpy array
        M_vol = np.array(M_vol)
        ori_vol = np.array(ori_vol)
        ft_vol = np.array(ft_vol)

        # Roll the axis
        M_vol = np.rollaxis(M_vol, 0, 3)
        ori_vol = np.rollaxis(ori_vol, 0, 3)
        ft_vol = np.rollaxis(ft_vol, 0, 3)

        self.data_ = np.array([M_vol, ori_vol, ft_vol])

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
        super(PhaseCongruencyExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that we fitted the data
        if self.data_ is None:
            raise RuntimeError('Fit the data before to extract anything.')

        # Allocate the data
        data = np.zeros((self.roi_data_[0].size, 3))
        # Extract the data for each feature
        for feat_dim in range(3):
            feat_data = self.data_[feat_dim]
            data[:, feat_dim] = feat_data[self.roi_data_]

        return data
