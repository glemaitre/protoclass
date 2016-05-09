"""Basic class for extraction."""


from abc import ABCMeta, abstractmethod


class BaseExtraction(object):
    """Basic class for extraction.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Constructor"""
        pass

    @abstractmethod
    def _validate_modality(self):
        """Check if the provided modality is of interest with the type of
        normalization."""
        raise NotImplementedError

    @abstractmethod
    def _validate_modality_gt(self, modality, ground_truth, cat):
        """Method to check the consistency of the modality with the
        ground-truth."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, modality, ground_truth=None, cat=None):
        """Find parameters for later transformation."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, modality, ground_truth=None, cat=None):
        """Extract the data from the given modality."""
        raise NotImplementedError
