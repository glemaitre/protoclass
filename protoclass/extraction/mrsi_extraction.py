"""Basic class to extract feature from MRSI modality."""

import warnings

import numpy as np
import SimpleITK as sitk

from abc import ABCMeta, abstractmethod

from ..data_management import MRSIModality
from ..data_management import GTModality

from .base_extraction import BaseExtraction

from ..utils.validation import check_modality
from ..utils.validation import check_modality_inherit
from ..utils.validation import check_modality_gt_mrsi


class MRSIExtraction(BaseExtraction):
    """Basic class to extract feature from MRSI modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, base_modality):
        """Constructor"""
        super(MRSIExtraction, self).__init__()
        self.base_modality_ = check_modality_inherit(base_modality,
                                                     MRSIModality)

    def _validate_gt_cat(self, modality, ground_truth, cat):
        """Set-up the roi to use for this object.

        Parameters
        ----------
        modality : object of type MRSIModality
            The modality object of interest.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Returns
        -------
        None

        """

        if ground_truth is None and cat is not None:
            warnings.warn('You specified a category for the ground-truth'
                          ' without giving any ground-truth. The whole volume'
                          ' will be considered for the fitting.')
            self.roi_data_ = np.nonzero(np.ones((np.size(modality.data_, 0),
                                                 np.size(modality.data_, 1),
                                                 np.size(modality.data_, 2))))
        elif ground_truth is None and cat is None:
            self.roi_data_ = np.nonzero(np.ones((np.size(modality.data_, 0),
                                                 np.size(modality.data_, 1),
                                                 np.size(modality.data_, 2))))
        elif ground_truth is not None and cat is None:
            raise ValueError('The category label of the ground-truth from'
                             ' which you want to extract the information needs'
                             ' to be specified.')
        else:
            self.roi_data_ = check_modality_gt_mrsi(modality,
                                                    ground_truth,
                                                    cat)

        return None

    def _resampling_as_gt(self, data, modality, ground_truth):
        """Private function to resample the data of the modality to the
        size of the ground-truth.

        Parameters
        ----------
        data : ndarray, shape (y, x, z)
            The data to be resampled.

        modality : object of type MRSIModality
            The modality object of interest.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        Return
        ------
        data_res : ndarray, shape (as in GTModality)

        """

        # Check that the modality is from the template class
        check_modality(modality, self.base_modality_)

        # Check that the data were read during the creation of the modality
        if not modality.is_read():
            raise ValueError('No data have been read during the construction'
                             ' of the modality object.')

        # Check that the ground-truth has been opened
        if not ground_truth.is_read():
            raise ValueError('Read the ground-truth.')

        # Check that the data has the same dimension than the modality given
        # We will use the X, Y, Z standard since that the spatial information
        # come from SimpleITK
        if not ((data.shape[1], data.shape[0], data.shape[2]) ==
                modality.metadata_['size']):
            raise ValueError('The dimension of the data and the modality given'
                             ' are not corresponding.')

        # Convert MRSI data to a SimpleITK structure
        # We need to convert from numpy array to ITK
        # Our convention was Y, X, Z
        # We need to convert it in Z, Y, X which will be converted
        # in X, Y, Z by ITK
        mrsi_img = sitk.GetImageFromArray(np.swapaxes(
            np.swapaxes(data, 0, 1), 0, 2))
        # Affect the spatial information to the object
        mrsi_img.SetDirection(modality.metadata_['direction'])
        mrsi_img.SetOrigin(modality.metadata_['origin'])
        mrsi_img.SetSpacing(modality.metadata_['spacing'])

        # Create a resampler object
        transform = sitk.Transform()
        transform.SetIdentity()

        # Setup the resampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mrsi_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        resampler.SetOutputDirection(ground_truth.metadata_['direction'])
        resampler.SetOutputOrigin(ground_truth.metadata_['origin'])
        resampler.SetOutputSpacing(ground_truth.metadata_['spacing'])
        resampler.SetSize(ground_truth.metadata_['size'])

        # Get the image resampled and convert in Y, X, Z order
        res_img = resampler.Execute(mrsi_img)
        data_res = np.swapaxes(np.swapaxes(sitk.GetArrayFromImage(res_img),
                                           0, 2), 0, 1)

        return data_res

    @abstractmethod
    def fit(self, modality, ground_truth=None, cat=None):
        """Find the parameters needed to apply the extraction.

        Parameters
        ----------
        modality : object of type MRSIModality
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
        # Check that the modality is from the template class
        check_modality(modality, self.base_modality_)

        # Check that the data were read during the creation of the modality
        if not modality.is_read():
            raise ValueError('No data have been read during the construction'
                             ' of the modality object.')

        # Check the consistency of the data
        self._validate_gt_cat(modality, ground_truth, cat)

        return self

    @abstractmethod
    def transform(self, modality, ground_truth=None, cat=None):
        """Extract the data from the given modality.

        Parameters
        ----------
        modality : object of type MRSIModality
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
             A matrix containing the features extracted.

        """
        # Check that the modality is from the template class
        check_modality(modality, self.base_modality_)

        # Check that the data were read during the creation of the modality
        if not modality.is_read():
            raise ValueError('No data have been read during the construction'
                             ' of the modality object.')

        # Check the consistency of the data
        self._validate_gt_cat(modality, ground_truth, cat)

        return self
