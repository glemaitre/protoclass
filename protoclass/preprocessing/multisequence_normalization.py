"""Basic class to normalize multisequence modality."""

from abc import ABCMeta, abstractmethod

from .base_normalization import BaseNormalization
# from ..data_management import MultisequenceModality


class MultisequenceNormalization(BaseNormalization):
    """Basic class to normalize multisequence modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Constructor."""
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """Find the parameters needed to apply the normalization. """
        raise NotImplementedError
