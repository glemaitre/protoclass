"""Basic class to normalize standalone modality."""

import numpy as np
import warnings

from abc import ABCMeta, abstractmethod

from .base_normalization import BaseNormalization

from ..data_management import StandaloneModality
from ..data_management import GTModality

from ..utils.validation import check_modality
from ..utils.validation import check_modality_inherit
from ..utils.validation import check_modality_gt


class StandaloneNormalization(BaseNormalization):
    """Basic class to normalize standalone modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    def __init__(self, base_modality):
        super(StandaloneNormalization, self).__init__()
        self.base_modality_ = check_modality_inherit(base_modality,
                                                     StandaloneModality)

    @abstractmethod
    def fit(self, modality, ground_truth=None, cat=None):
        """Find the parameters needed to apply the normalization.

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
            self.roi_data_ = check_modality_gt(modality,
                                               ground_truth,
                                               cat)

        return self

    @abstractmethod
    def normalize(self, modality):
        """Method to normalize the given modality using the fitted parameters.

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
        """Denormalize the given modality using the fitted parameters.

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
