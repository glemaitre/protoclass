""" Basic class to normalize multisequence modality.
"""

from abc import ABCMeta, abstractmethod

from .base_normalization import BaseNormalization
from ..data_management import MultisequenceModality


class MultisequenceNormalization(BaseNormalization):
    """ Basic class to normalize multisequence modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """ Constructor. """
        raise NotImplementedError

    @abstractmethod
    def _validate_modality(self):
        """ Check if the provided modality is of interest with the type of
        normalization. """
        raise NotImplementedError

    def _validate_modality_gt(self, modality, ground_truth, cat):
        """ Method to check the consistency of the modality with the
        ground-truth. """
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """ Method to find the parameters needed to apply the
        normalization. """
        raise NotImplementedError
