"""Edge signal extraction from standalone modality."""

import numpy as np

from scipy.ndimage.filters import generic_gradient_magnitude
from scipy.ndimage.filters import generic_laplace
from scipy.ndimage.filters import prewitt
from scipy.ndimage.filters import sobel

from .standalone_extraction import StandaloneExtraction

FILTER = ('sobel', 'prewitt')
DERIVATIVE = ('1st', '2nd')

class EdgeSignalExtraction(StandaloneExtraction):
    """Edge signal extraction from standalone modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    edge_detector : str, optional (default='sobel')
        Name of the filter to apply. Can be 'sobel', 'prewitt'.

    n_derivative : str, optional (default='1st')
        Which level of derivative to compute. Can be '1st', '2nd'.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, edge_detector='sobel',
                 n_derivative='1st'):
        super(EdgeSignalExtraction, self).__init__(base_modality)
        self.edge_detector = edge_detector
        self.n_derivative = n_derivative
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
        super(EdgeSignalExtraction, self).fit(modality=modality,
                                              ground_truth=ground_truth,
                                              cat=cat)

        # Check that the filter provided is known
        if self.edge_detector not in FILTER:
            raise ValueError('{} filter is unknown'.format(self.edge_detector))

        # Check the value of the derivative to compute
        if self.n_derivative not in DERIVATIVE:
            raise ValueError('It is impossible to compute the {}. Only `1st`'
                             ' and `2nd` can be computed'.format(
                                 self.n_derivative))

        if self.edge_detector == 'sobel':
            if self.n_derivative == '1st':
                self.data_ = generic_gradient_magnitude(modality.data_, sobel)
            else:
                self.data_ = generic_laplace(modality.data_, sobel)
        if self.edge_detector == 'prewitt':
            if self.n_derivative == '1st':
                self.data_ = generic_gradient_magnitude(modality.data_,
                                                        prewitt)
            else:
                self.data_ = generic_laplace(modality.data_, prewitt)

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
        super(EdgeSignalExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that we fitted the data
        if self.data_ is None:
            raise RuntimeError('Fit the data before to extract anything.')

        return self.data_[self.roi_data_]
