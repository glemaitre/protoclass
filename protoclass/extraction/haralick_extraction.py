"""Haralick extraction from standalone modality."""
from __future__ import division

import numpy as np

from joblib import Parallel, delayed

from mahotas.features.texture import haralick

from sklearn.feature_extraction.image import extract_patches

from .standalone_extraction import StandaloneExtraction


def _compute_haralick_features(patch, distance):
    """Compute the haralick feature.

    This function is used for parallel processing.

    Parameters
    ----------
    patch: ndarray, (patch_size)
        The patch to consider to compute the cooccurence matrix.

    distance: int,
        The distance to use to compute the cooccurence matrix.

    Returns
    -------
    haralick_features: ndarray, shape (4, 13)
        The haralick features.
    """

    return haralick(patch, distance=distance)


class HaralickExtraction(StandaloneExtraction):
    """Haralick extraction from standalone modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    distance : int, optional (default=1)
        The distance used to compute the cooccurence matrix

    patch_size : int or tuple, optional (default=(9, 9, 3))
        The size of the sliding used to extract patches later used to compute
        the cooccurence matrix.

    levels : int, optional (default=256)
        The input image should contain integers in [0, levels-1], where levels
        indicate the number of grey-levels counted (typically 256 for an 8-bit
        image). This argument is required for 16-bit images or higher and is
        typically the maximum of the image. As the output matrix is at least
        levels x levels, it might be preferable to use binning of the input
        image rather than large values for levels.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, distance=1, patch_size=(9, 9, 3),
                 levels=256):
        super(HaralickExtraction, self).__init__(base_modality)
        self.distance = distance
        self.patch_size = patch_size
        self.levels = levels
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
        super(HaralickExtraction, self).fit(modality=modality,
                                              ground_truth=ground_truth,
                                              cat=cat)

        # Get the data and rescale as integers within the given levels
        vol_haralick = ((modality.data_ - np.ndarray.min(modality.data_)) *
                        ((self.levels -1) /
                         (np.ndarray.max(modality.data_) -
                          np.ndarray.min(modality.data_)))).astype(int)

        # Extract the set of patches from the modality data
        patches = extract_patches(vol_haralick, patch_shape=self.patch_size)

        # Allocate the haralick maps, one for each feature that
        # will be computed
        nb_directions = 13
        nb_features = 13
        self.data_ = np.zeros((modality.data_.shape[0],
                               modality.data_.shape[1],
                               modality.data_.shape[2],
                               nb_directions,
                               nb_features))

        # WE NEED TO PARALLELIZE THIS CODE

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

        # for y in range(patches.shape[0]):
        #     for x in range(patches.shape[1]):
        #         for z in range(patches.shape[2]):
        #             print 'Compute for the pixel at position {}{}{}'.format(
        #                 y, x, z)
        #             # Compute the haralick features
        #             self.data_[y + y_shift,
        #                        x + x_shift,
        #                        z + z_shift, :] = haralick(
        #                            patches[y, x, z, :],
        #                            distance=self.distance)

        # Create the list of indices to process
        yy, xx, zz = np.meshgrid(range(patches.shape[0]),
                                 range(patches.shape[1]),
                                 range(patches.shape[2]))
        # Linearize for fast processing
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)
        zz = zz.reshape(-1)

        # Go for the parallel loop
        haralick_features = Parallel(n_jobs=-1)(delayed(
            _compute_haralick_features)(patches[y, x, z, :], self.distance)
                                                for y, x, z in zip(yy, xx, zz))

        # Convert to numpy array
        haralick_features = np.array(haralick_features)
        # Reshape the feature matrix
        haralick_features = haralick_features.reshape((patches.shape[0],
                                                       patches.shape[1],
                                                       patches.shape[2],
                                                       nb_directions,
                                                       nb_features))
        # Copy the feature into the object
        self.data_[y_shift : -y_shift,
                   x_shift : -x_shift,
                   z_shift : -z_shift] = haralick_features

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
        super(HaralickExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that we fitted the data
        if self.data_ is None:
            raise RuntimeError('Fit the data before to extract anything.')

        # Convert the roi to a numpy array
        roi_data = np.array(self.roi_data_)

        # Allocate the output
        nb_directions = 13
        nb_features = 13
        n_sample = roi_data.shape[1]
        n_dimension = nb_directions * nb_features
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
