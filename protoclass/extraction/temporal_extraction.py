"""Basic class to extract feature from temporal modality."""

import warnings

import numpy as np

from abc import ABCMeta, abstractmethod

from ..data_management import TemporalModality
from ..data_management import GTModality

from .base_extraction import BaseExtraction

from ..utils.validation import check_modality
from ..utils.validation import check_modality_inherit
from ..utils.validation import check_modality_gt


class TemporalExtraction(BaseExtraction):
    """Basic class to extract feature from temporal modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, base_modality):
        """Constructor"""
        super(TemporalExtraction, self).__init__()
        self.base_modality_ = check_modality_inherit(base_modality,
                                                     TemporalModality)

    def _validate_gt_cat(self, modality, ground_truth, cat):
        """Set-up the roi to use for this object.

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

        Returns
        -------
        None

        """

        if ground_truth is None and cat is not None:
            warnings.warn('You specified a category for the ground-truth'
                          ' without giving any ground-truth. The whole volume'
                          ' will be considered for the fitting.')
            self.roi_data_ = np.nonzero(np.ones((np.size(modality.data_, 1),
                                                 np.size(modality.data_, 2),
                                                 np.size(modality.data_, 3))))
        elif ground_truth is None and cat is None:
            self.roi_data_ = np.nonzero(np.ones((np.size(modality.data_, 1),
                                                 np.size(modality.data_, 2),
                                                 np.size(modality.data_, 3))))
        elif ground_truth is not None and cat is None:
            raise ValueError('The category label of the ground-truth from'
                             ' which you want to extract the information needs'
                             ' to be specified.')
        else:
            self.roi_data_ = check_modality_gt(modality,
                                               ground_truth,
                                               cat)

        return None

    @abstractmethod
    def fit(self, modality, ground_truth=None, cat=None):
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
        modality : object of type TemporalModality
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
