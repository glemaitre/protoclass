"""DCT extraction from standalone modality."""
from __future__ import division

import numpy as np

from joblib import Parallel, delayed

from scipy.fftpack import dct

from sklearn.feature_extraction.image import extract_patches

from .standalone_extraction import StandaloneExtraction


def _compute_dct_features(patch):
    """Compute the haralick feature.

    This function is used for parallel processing.

    Parameters
    ----------
    patch: ndarray, (patch_size)
        The patch to consider to compute the DCT.

    Returns
    -------
    dct_features: ndarray, shape (patch.size, )
        The DCT features.
    """

    return dct(dct(dct(patch).transpose(
        0, 2, 1)).transpose(
            1, 2, 0)).transpose(
                1, 2, 0).transpose(
                    0, 2, 1).reshape(-1)


class DCTExtraction(StandaloneExtraction):
    """DCT extraction from standalone modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    patch_size : int or tuple, optional (default=(9, 9, 3))
        The size of the sliding used to extract patches later used to compute
        the cooccurence matrix.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, patch_size=(9, 9, 3)):
        super(DCTExtraction, self).__init__(base_modality)
        self.patch_size = patch_size
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
        super(DCTExtraction, self).fit(modality=modality,
                                       ground_truth=ground_truth,
                                       cat=cat)

        # Extract the set of patches from the modality data
        patches = extract_patches(modality.data_, patch_shape=self.patch_size)

        # Allocate the haralick maps, one for each feature that
        # will be computed
        nb_features = np.prod(self.patch_size)
        self.data_ = np.zeros((modality.data_.shape[0],
                               modality.data_.shape[1],
                               modality.data_.shape[2],
                               nb_features))

        # # Extract Haralick feature for each patch
        # # Define the shift to apply
        if isinstance(self.patch_size, tuple):
            y_shift = int(np.ceil((self.patch_size[0] - 1) / 2.))
            x_shift = int(np.ceil((self.patch_size[1] - 1) / 2.))
            z_shift = int(np.ceil((self.patch_size[2] - 1) / 2.))
        elif isinstance(self.patch_size, int):
            y_shift = int(np.ceil((self.patch_size - 1) / 2.))
            x_shift = int(np.ceil((self.patch_size - 1) / 2.))
            z_shift = int(np.ceil((self.patch_size - 1) / 2.))

        # Create the list of indices to process
        yy, xx, zz = np.meshgrid(range(patches.shape[0]),
                                 range(patches.shape[1]),
                                 range(patches.shape[2]))
        # Linearize for fast processing
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)
        zz = zz.reshape(-1)

        # Go for the parallel loop
        dct_features = Parallel(n_jobs=-1)(delayed(
            _compute_dct_features)(patches[y, x, z, :])
                                                for y, x, z in zip(yy, xx, zz))

        # Convert to numpy array
        dct_features = np.array(dct_features)
        # Reshape the feature matrix
        dct_features = dct_features.reshape((patches.shape[0],
                                                       patches.shape[1],
                                                       patches.shape[2],
                                                       nb_features))
        # Copy the feature into the object
        self.data_[y_shift : -y_shift,
                   x_shift : -x_shift,
                   z_shift : -z_shift] = dct_features

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
        super(DCTExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that we fitted the data
        if self.data_ is None:
            raise RuntimeError('Fit the data before to extract anything.')

        # Convert the roi to a numpy array
        roi_data = np.array(self.roi_data_)

        # Allocate the output
        n_sample = roi_data.shape[1]
        n_dimension = np.prod(self.patch_size)
        data = np.empty((n_sample, n_dimension))

        # Copy the data at the right place
        for idx_sample in range(n_sample):
            # Get the coordinate of the point to consider
            coord = roi_data[:, idx_sample]

            # Extract the data
            data[idx_sample, :] = self.data_[coord[0],
                                             coord[1],
                                             coord[2],
                                             :].reshape(-1)

        return data
