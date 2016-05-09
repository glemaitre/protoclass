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
    def fit(self, modality, ground_truth=None, cat=None):
        """Find parameters for later transformation."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, modality, ground_truth=None, cat=None):
        """Extract the data from the given modality."""
        raise NotImplementedError
