"""LBP extraction from standalone modality."""

import numpy as np

from skimage.feature import local_binary_pattern

from .standalone_extraction import StandaloneExtraction


KNOWN_KIND = ('default', 'ror', 'uniform', 'nri_uniform',
              'var')


class LBPExtraction(StandaloneExtraction):
    """LBP extraction from standalone modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    p : int, optional (default=8)
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).

    r : float, optional (default=1.)
        Radius of circle (spatial resolution of the operator).

    kind : str, optional (default='uniform')
        Type of LBP to choose: 'default', 'ror', 'uniform', 'nri_uniform', and
        'var'

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, p=8, r=1., kind='uniform'):
        super(LBPExtraction, self).__init__(base_modality)
        self.p = p
        self.r = r
        self.kind = kind
        self.data_ = None

    def fit(self, modality, ground_truth=None, cat=None):
        """Compute the LBP images in the three-ortogonal planes.

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
        super(LBPExtraction, self).fit(modality=modality,
                                       ground_truth=ground_truth,
                                       cat=cat)

        # Fix the z axis
        lbp_z = np.zeros(modality.data_.shape)
        for z in range(modality.data_.shape[2]):
            lbp_z[:, :, z] = local_binary_pattern(modality.data_[:, :, z],
                                                  self.p, self.r,
                                                  method=self.kind)
        # Fix the x axis
        lbp_x = np.zeros(modality.data_.shape)
        for x in range(modality.data_.shape[1]):
            lbp_x[:, x, :] = local_binary_pattern(modality.data_[:, x, :],
                                                  self.p, self.r,
                                                  method=self.kind)

        # Fix the y axis
        lbp_y = np.zeros(modality.data_.shape)
        for y in range(modality.data_.shape[0]):
            lbp_y[y, :, :] = local_binary_pattern(modality.data_[y, :, :],
                                                  self.p, self.r,
                                                  method=self.kind)

        self.data_ = np.array((lbp_z, lbp_x, lbp_y))

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
        super(LBPExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that we fitted the data
        if self.data_ is None:
            raise RuntimeError('Fit the data before to extract anything.')

        data = np.zeros((self.roi_data_[0].size, self.data_.shape[0]))
        for feat_dim in range(self.data_.shape[0]):
            feat_data = self.data_[feat_dim, :, :, :]
            data[:, feat_dim] = feat_data[self.roi_data_]

        return data
