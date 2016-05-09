"""Basic class to extract feature from temporal modality."""

import warnings

import numpy as np

from abc import ABCMeta, abstractmethod

from ..data_management import TemporalModality
from ..data_management import GTModality

from .base_extraction import BaseExtraction

from ..utils.validation import check_modality


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
        self.base_modality = base_modality
        self._validate_modality()

    def _validate_modality(self):
        """Check if the provided modality is of interest with the type of
        normalization."""

        # Check that the base modality is a subclass of TemporalModality
        if not issubclass(type(self.base_modality), TemporalModality):
            raise ValueError('The base modality provided in the constructor is'
                             ' not a TemporalModality.')
        else:
            self.base_modality_ = self.base_modality

    def _validate_modality_gt(self, modality, ground_truth, cat):
        """Check the consistency of the modality with the ground-truth.

        Parameters
        ----------
        modality : object
            The modality object of interest.

        ground-truth : object of type GTModality
            The ground-truth of GTModality.

        cat : str
            String corresponding at the ground-truth of interest.

        Returns
        -------
        roi_data : ndarray, shape (non_zero_samples, 3)
            Corresponds to the indexes of the data of insterest
            extracted from the ground-truth.

        """

        # Check that the ground-truth is from GTModality
        if not isinstance(ground_truth, GTModality):
            raise ValueError('The ground-truth should be an object of'
                             ' class GTModality.')

        # Check that the ground truth has been read
        if not ground_truth.is_read():
            raise ValueError('No data have been read during the'
                             'construction of the GT modality object.')

        # Check that the size of the ground-truth and the modality
        # are consistant
        # In this case check only the last three dimension
        if ((np.size(modality.data_, 1),
             np.size(modality.data_, 2),
             np.size(modality.data_, 3)) !=
                ground_truth.extract_gt_data(cat, 'data').shape):
            raise ValueError('The ground-truth does not correspond to the'
                             ' given modality volume.')

        # Find the element which are not zero
        return ground_truth.extract_gt_data(cat, 'index')

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
            self.roi_data_ = self._validate_modality_gt(modality,
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
