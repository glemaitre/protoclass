"""Edge signal extraction from standalone modality."""

import numpy as np

from scipy.ndimage.filters import generic_gradient_magnitude
from scipy.ndimage.filters import laplace
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import prewitt
from scipy.ndimage.filters import sobel
from scipy.ndimage.filters import _ni_support
from scipy.ndimage.filters import correlate1d

from .standalone_extraction import StandaloneExtraction

FILTER = ('sobel', 'prewitt', 'kirsch', 'scharr', 'laplacian')
KIRSCH_FILTERS = (np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                  np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
                  np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                  np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                  np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
                  np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
                  np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                  np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
KIRSCH_DIRECTIONS = np.array([np.pi / 2., 3. * np.pi / 4.,
                              np.pi, -3. * np.pi / 4.,
                              -np.pi / 2., -np.pi / 4.,
                              0., np.pi / 4.])

def scharr(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Scharr filter.

    Parameters
    ----------
    %(input)s
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s

    """
    input = np.asarray(input)
    axis = _ni_support._check_axis(axis, input.ndim)
    output, return_value = _ni_support._get_output(output, input)
    correlate1d(input, [-1, 0, 1], axis, output, mode, cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, [3, 10, 3], ii, output, mode, cval, 0)
    return return_value


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

    data_ : ndarray (1, 2, or 3, volume_size)
        The computed gradient:

        - 'laplacian' will compute a single array containing the laplacian
        value.
        - 'sobel', 'prewitt', and 'scharr' compute: (i) the gradient magnitude,
        (ii) the gradient azimuth, and (iii) the gradient elevation.
        - 'kirsch' is computed in 2D and therefore the gradient magnitude and
        the gradient orientation are available

    """

    def __init__(self, base_modality, edge_detector='sobel'):
        super(EdgeSignalExtraction, self).__init__(base_modality)
        self.edge_detector = edge_detector
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

        self.data_ = []

        # SOBEL filter
        if self.edge_detector == 'sobel':
            # Compute the gradient in the three direction Y, X, Z
            grad_y = sobel(modality.data_, axis=0)
            grad_x = sobel(modality.data_, axis=1)
            grad_z = sobel(modality.data_, axis=2)
            # Compute the magnitude gradient
            self.data_.append(generic_gradient_magnitude(modality.data_,
                                                         sobel))
            # Compute the gradient azimuth
            self.data_.append(np.arctan2(grad_y, grad_x))
            # Compute the gradient elevation
            self.data_.append(np.arccos(grad_z / self.data_[0]))

        # PREWITT filter
        elif self.edge_detector == 'prewitt':
            # Compute the gradient in the three direction Y, X, Z
            grad_y = prewitt(modality.data_, axis=0)
            grad_x = prewitt(modality.data_, axis=1)
            grad_z = prewitt(modality.data_, axis=2)
            # Compute the magnitude gradient
            self.data_.append(generic_gradient_magnitude(modality.data_,
                                                         prewitt))
            # Compute the gradient azimuth
            self.data_.append(np.arctan2(grad_y, grad_x))
            # Compute the gradient elevation
            self.data_.append(np.arccos(grad_z / self.data_[0]))

        # SCHARR filter
        elif self.edge_detector == 'scharr':
            # Compute the gradient in the three direction Y, X, Z
            grad_y = scharr(modality.data_, axis=0)
            grad_x = scharr(modality.data_, axis=1)
            grad_z = scharr(modality.data_, axis=2)
            # Compute the magnitude gradient
            self.data_.append(generic_gradient_magnitude(modality.data_,
                                                         scharr))
            # Compute the gradient azimuth
            self.data_.append(np.arctan2(grad_y, grad_x))
            # Compute the gradient elevation
            self.data_.append(np.arccos(grad_z / self.data_[0]))

        # KIRSCH filter
        elif self.edge_detector == 'kirsch':
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
            self.data_.append(np.ndarray.max(conv_data, axis=3))
            # Extract the orientattion of the gradients
            self.data_.append(KIRSCH_DIRECTIONS[np.ndarray.argmax(conv_data,
                                                                  axis=3)])

        # LAPLACIAN filter
        elif self.edge_detector == 'laplacian':
            self.data_.append(laplace(modality.data_))

        # Convert the data into a numpy array
        self.data_ = np.array(self.data_)
        # Replace the few NaN value to zero
        self.data_ = np.nan_to_num(self.data_)

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

        # LAPLACIAN
        if self.edge_detector == 'laplacian':
            # Allocate the data
            data = self.data_[0]
            data = data[self.roi_data_]

        # KIRSCH
        elif self.edge_detector == 'kirsch':
            # Allocate the data
            data = np.zeros((self.roi_data_.size, 2))
            # Extract the data for each feature
            for feat_dim in range(2):
                feat_data = self.data_[feat_dim]
                data[feat_dim, :] = feat_data[self.roi_data_]

        # ALL THE OTHER CASES
        else:
            # Allocate the data
            data = np.zeros((self.roi_data_.size, 3))
            # Extract the data for each feature
            for feat_dim in range(3):
                feat_data = self.data_[feat_dim]
                data[feat_dim, :] = feat_data[self.roi_data_]

        return data
