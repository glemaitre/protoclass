"""Basic class for multisequence modality (DW, GT)."""

from abc import ABCMeta, abstractmethod

from .base_modality import BaseModality


class MultisequenceModality(BaseModality):
    """Basic class for multisequence medical modality (DW).

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data=None):
        """Constructor."""
        super(MultisequenceModality, self).__init__(path_data=path_data)

    @abstractmethod
    def update_histogram(self):
        """Method to compute histogram and statistics."""
        raise NotImplementedError

    @abstractmethod
    def read_data_from_path(self, path_data=None):
        """Read the data from a specific path."""
        raise NotImplementedError
