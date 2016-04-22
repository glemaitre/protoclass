""" Basic class to normalize standalone modality.
"""

import numpy as np
import warnings

from abc import ABCMeta, abstractmethod

from .base_normalization import BaseNormalization

from ..data_management import StandaloneModality
from ..data_management import GTModality

from ..utils.validation import check_modality


class StandaloneNormalization(BaseNormalization):
    """ Basic class to normalize standalone modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    def __init__(self, base_modality):
        super(StandaloneNormalization, self).__init__()
        self.base_modality = base_modality
        self._validate_modality()

    def _validate_modality(self):
        """ Check if the provided modality is of interest with the type of
        normalization. """

        # Check that the base modality is a subclass of TemporalModality
        if not issubclass(type(self.base_modality), StandaloneModality):
            raise ValueError('The base modality provided in the constructor is'
                             ' not a StandaloneModality.')
        else:
            self.base_modality_ = self.base_modality

    def _validate_modality_gt(self, modality, ground_truth, cat):
        """ Method to check the consistency of the modality with the
        ground-truth.

        Parameters
        ----------
        modality : object
            The modality object of interest.

        ground-truth : object of type GTModality
            The ground-truth of GTModality.

        cat : str
            String corresponding at the ground-truth of interest.

        Return
        ------
        roi_data : ndarray, shape (non_zero_samples, 3)
            Corresponds to the indexes of the data of insterest
            extracted from the ground-truth.
        """

        # Check that the ground-truth is from GTModality
        if not isinstance(ground_truth, GTModality):
            raise ValueError('The ground-truth should be an object of'
                             ' class GTModality.')

        # Check that the category given is part of the ground-truth
        if cat not in ground_truth.cat_gt_:
            raise ValueError('The ground-truth category required has not been'
                             ' loaded during the instance of the'
                             ' GTModality object.')
        else:
            # Extract the indexes of interest to be returned
            idx_gt = ground_truth.cat_gt_.index(cat)

        # Check that the size of the ground-truth and the modality
        # are consistant
        if modality.data_.shape != ground_truth.data_[idx_gt, :, :, :].shape:
            raise ValueError('The ground-truth does not correspond to the'
                             ' given modality volume.')

        # Find the element which are not zero
        return np.nonzero(ground_truth.data_[idx_gt, :, :, :])

    @abstractmethod
    def fit(self, modality, ground_truth=None, cat=None):
        """ Method to find the parameters needed to apply the
        normalization.

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
        if ground_truth is None and cat is not None:
            warnings.warn('You specified a category for the ground-truth'
                          ' without giving any ground-truth. The whole volume'
                          ' will be considered for the fitting.')
            self.roi_data_ = np.nonzero(np.ones(modality.data_.shape))
        elif ground_truth is None and cat is None:
            self.roi_data_ = np.nonzero(np.ones(modality.data_.shape))
        elif ground_truth is not None and cat is None:
            raise ValueError('The category label of the ground-truth from'
                             ' which you want to extract the information needs'
                             ' to be specified.')
        else:
            # Check that the data were read during the creation of the
            # ground-truth modality
            if not ground_truth.is_read():
                raise ValueError('No data have been read during the'
                                 'construction of the GT modality object.')

            self.roi_data_ = self._validate_modality_gt(modality,
                                                        ground_truth,
                                                        cat)

        return self

    @abstractmethod
    def normalize(self, modality):
        """ Method to normalize the given modality using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.
        """
        # Check that the class of modality is the same than the template
        # modality
        check_modality(modality, self.base_modality_)

        return self

    @abstractmethod
    def denormalize(self, modality):
        """ Method to denormalize the given modality using the
        fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.
        """
        # Check that the class of modality is the same than the template
        # modality
        check_modality(modality, self.base_modality_)

        return self
