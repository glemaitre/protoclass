"""Edge signal extraction from standalone modality."""

import numpy as np

from scipy.ndimage.filters import generic_gradient_magnitude
from scipy.ndimage.filters import generic_laplace
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import prewitt
from scipy.ndimage.filters import sobel

from .standalone_extraction import StandaloneExtraction

FILTER = ('sobel', 'prewitt', 'kirsch')
DERIVATIVE = ('1st', '2nd')
KIRSCH_FILTERS = (np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                  np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
                  np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                  np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                  np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
                  np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
                  np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                  np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))


class EdgeSignalExtraction(StandaloneExtraction):
    """Edge signal extraction from standalone modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    edge_detector : str, optional (default='sobel')
        Name of the filter to apply. Can be 'sobel', 'prewitt', 'kirsch'.

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
        elif self.edge_detector == 'prewitt':
            if self.n_derivative == '1st':
                self.data_ = generic_gradient_magnitude(modality.data_,
                                                        prewitt)
            else:
                self.data_ = generic_laplace(modality.data_, prewitt)
        elif self.edge_detector == 'kirsch':
            if self.n_derivative == '1st':
                conv_data = np.zeros((modality.data_.shape[0],
                                      modality.data_.shape[1],
                                      modality.data_.shape[2],
                                      len(KIRSCH_FILTERS)))
                # Compute the convolution for each slice
                for sl in range(modality.data_.shape[2]):
                    for idx_kirsch, kirsh_f in enumerate(KIRSCH_FILTERS):
                        conv_data[:, :, sl, idx_kirsch] = convolve(
                            modality.data_[:, :, sl], kirsh_f, mode='reflect')

                # Extract the maximum gradients
                self.data_ = np.ndarray.max(conv_data, axis=3)
            else:
                conv_data = np.zeros((modality.data_.shape[0],
                                      modality.data_.shape[1],
                                      modality.data_.shape[2],
                                      len(KIRSCH_FILTERS)))
                # Compute the convolution for each slice
                for sl in range(modality.data_.shape[2]):
                    for idx_kirsch, kirsh_f in enumerate(KIRSCH_FILTERS):
                        conv_data[:, :, sl, idx_kirsch] = convolve(
                            modality.data_[:, :, sl], kirsh_f, mode='reflect')

                dev_1 = np.ndarray.max(conv_data, axis=3)

                # Compute the second derivative
                for sl in range(dev_1.shape[2]):
                    for idx_kirsch, kirsh_f in enumerate(KIRSCH_FILTERS):
                        conv_data[:, :, sl, idx_kirsch] = convolve(
                            dev_1[:, :, sl], kirsh_f, mode='reflect')

                # Extract the maximum gradients
                self.data_ = np.ndarray.max(conv_data, axis=3)

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
