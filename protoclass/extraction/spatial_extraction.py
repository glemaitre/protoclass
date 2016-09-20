"""Spatial feature extraction from standalone modality."""
from __future__ import division

import numpy as np

from scipy.spatial import cKDTree
from scipy.ndimage.measurements import center_of_mass

from skimage.measure import find_contours

from sklearn.preprocessing import MinMaxScaler

from .standalone_extraction import StandaloneExtraction

KIND_KNOWN = ('position', 'distance')
COORD_SYSTEM_KNOWN = ('euclidean', 'cylindrical')
REFERENCE = ('centre', 'nn-contour-point')



class SpatialExtraction(StandaloneExtraction):
    """Spatial extraction from standalone modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    kind : str, optional (default='position')
        The type of spatial feature to extract. Can be either 'position' or
        'distance'.

    coord_system : str, optional (default='euclidean')
        The type of coordinate system to use. Can be either 'euclidean' or
        'cylindrical'.

    reference : str, optional (default='centre')
        If type is 'distance', the reference can be either 'centre' or
        'nn-contour-point'

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    data_ : ndarray,
        Containing the data of all the slices according to the desired spatial
        information

    """

    def __init__(self, base_modality, kind='position',
                 coord_system='euclidean', reference='centre'):
        super(SpatialExtraction, self).__init__(base_modality)
        self.kind = kind
        self.coord_system = coord_system
        self.reference = reference
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
        super(SpatialExtraction, self).fit(modality=modality,
                                           ground_truth=ground_truth,
                                           cat=cat)

        if ground_truth is None:
            raise ValueError('It is not possible to compute relative position'
                             ' without a segmentation of the prostate organ.')

        if self.kind == 'distance' and self.coord_system == 'cylindrical':
            raise ValueError('The distance cannot be expressed in cylindrical'
                             ' coordinate.')

        # Compute the affine matrix to compute the real world coordinate
        direc_mat = np.reshape(modality.metadata_['direction'], (3, 3))
        orig_mat = np.array(modality.metadata_['origin'])
        spac_mat = np.array(modality.metadata_['spacing'])

        affine_mat = np.matrix([[direc_mat[0, 0] * spac_mat[0],
                                 direc_mat[0, 1] * spac_mat[1],
                                 direc_mat[0, 2] * spac_mat[2],
                                 orig_mat[0]],
                                [direc_mat[1, 0] * spac_mat[0],
                                 direc_mat[1, 1] * spac_mat[1],
                                 direc_mat[1, 2] * spac_mat[2],
                                 orig_mat[1]],
                                [direc_mat[2, 0] * spac_mat[0],
                                 direc_mat[2, 1] * spac_mat[1],
                                 direc_mat[2, 2] * spac_mat[2],
                                 orig_mat[2]],
                                [0, 0, 0, 1]])

        # Build all the possible coordinate of the image in the real world
        sz_mat = modality.metadata_['size']
        X, Y, Z = np.meshgrid(range(sz_mat[0]),
                              range(sz_mat[1]),
                              range(sz_mat[2]))
        # Linearize the indexes
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        Z = Z.reshape(-1)
        # Build a huge matrix with all the coordinate
        coord_voxel = np.matrix([X, Y, Z, np.array([1] * X.size)])
        # Compute the projection in real world
        coord_real = affine_mat * coord_voxel
        # Remove the last row which is not useful to compute the distances
        coord_real = coord_real[:-1, :]

        # We need to make the computation regarding the reference to consider
        if self.kind == 'distance' and self.reference == 'nn-contour-point':
            # Create a ground-truth volume
            gt = np.zeros(modality.data_.shape)
            gt[self.roi_data_] = 255.
            contour_idx = np.empty([0, 3])
            # We need to compute the contour of the ground-truth
            for sl in range(gt.shape[2]):
                # Check that this is a slice with some ground-truth
                if np.unique(gt[:, :, sl]).size == 1:
                    continue
                slice_idx = find_contours(gt[:, :, sl], 254.)
                print len(slice_idx)
                if len(slice_idx) != 1:
                    raise ValueError('There is a problem. More than one'
                                     ' contour have been found.')
                # We need to concatenate the slice position
                slice_idx = np.concatenate((slice_idx[0],
                                            np.atleast_2d(
                                                [sl] *
                                                slice_idx[0].shape[0]).T),
                                           axis=1)
                # Good to be concatenated
                print contour_idx.shape
                print slice_idx.shape
                contour_idx = np.concatenate((contour_idx, slice_idx), axis=0)
            # We need to swap X and Y to be sure that everything will go fine
            contour_idx[:, 0], contour_idx[:, 1] = (contour_idx[:, 1],
                                                    contour_idx[:, 0].copy())
            # Transform to homogeneous coordinates
            contour_idx = np.concatenate((contour_idx,
                                          np.atleast_2d(
                                              [1] * contour_idx.shape[0]).T),
                                         axis=1)

            # Transform the data in the real world coordinate
            contour_idx = (affine_mat * contour_idx.T).T

            # Remove the last row which is not useful to compute the distance
            # later
            contour_idx = contour_idx[:, :-1]

            # Fit a tree to find the nearest neighbours contour point later on
            self.tree = cKDTree(contour_idx)

            # Compute the distances for all the data
            dist, _ = self.tree.query(coord_real.T, k=1, n_jobs=-1)

            # We need to reshape the distance such that it corresponds to the
            # original size and we need to swap Y and X back
            self.data_ = np.swapaxes(np.reshape(dist, sz_mat), 0, 1)

            print self.data_.shape

        elif self.reference == 'center':
            # Create a ground-truth volume
            gt = np.zeros(modality.data_.shape)
            gt[self.roi_data_] = 1.
            # Compute the center of mass of the ground-truth
            barycenter = center_of_mass(gt)

            # We need to change X and Y and express in homogeneous coordinate
            barycenter = np.atleast_2d((barycenter[1],
                                        barycenter[0],
                                        barycenter[2],
                                        1)).T
            # Compute the coordinate of the barycenter in the real world
            # coordinate system
            barycenter = affine_mat * barycenter
            # Remove the last row
            self.barycenter = barycenter[:-1, :]

            # Compute the relative position in the euclidean space which will
            # be used afterwards
            coord_real -= self.barycenter

            # Reshape the coordinate properly
            X = np.squeeze(np.array(coord_real[0, :]))
            Y = np.squeeze(np.array(coord_real[1, :]))
            Z = np.squeeze(np.array(coord_real[2, :]))

            self.data_ = np.array([Y.reshape(modality.data_.shape),
                                   X.reshape(modality.data_.shape),
                                   Z.reshape(modality.data_.shape)])

            # We need to extract either the position or the distance
            if self.kind == 'position' and self.coord_system == 'cylindrical':
                # Convert the euclidean coordinate to cylindrical coordinate
                # Copy the data temporary
                eucl_data = self.data_.copy()
                self.data_[0] = np.sqrt(eucl_data[0] ** 2 + eucl_data[1] ** 2)
                self.data_[1] = np.arctan2(eucl_data[0], eucl_data[1])

            elif self.kind == 'distance':
                # Compute the euclidean distance
                self.data_ = np.sum(self.data_ ** 2, axis=0)

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
            If distance, a single feature is returned. If position,
            3 features are returned.

        """
        super(SpatialExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that we fitted the data
        if self.data_ is None:
            raise RuntimeError('Fit the data before to extract anything.')

        if self.kind == 'position':
            # Allocate the data
            data = np.zeros((self.roi_data_[0].size, 3))
            # Extract the data for each feature
            for feat_dim in range(3):
                feat_data = self.data_[feat_dim]
                data[:, feat_dim] = feat_data[self.roi_data_]

            # We need to normalize the data
            if self.coord_system == 'euclidean':
                # Define an object for the normalization
                norm_obj = MinMaxScaler()
                data = norm_obj.fit_transform(data)

            elif self.coord_system == 'cyclindrical':
                # We need to normalize r and z between -1 and 1
                norm_obj = MinMaxScaler(feature_range=(-1, 1))
                # r component
                data[:, 0] = MinMaxScaler(data[:, 0])
                # Z component
                data[:, 2] = MinMaxScaler(data[:, 2])

        elif self.kind == 'distance':
            data = self.data_[self.roi_data_]
            # Normalize the distance with the max
            data = data / np.max(data)

        return data
