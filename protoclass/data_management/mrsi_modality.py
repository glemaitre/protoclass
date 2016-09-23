"""Basic class for MRSI modality"""

from abc import ABCMeta, abstractmethod

from .base_modality import BaseModality


class MRSIModality(BaseModality):
    """Basic class for MRSI.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data=None):
        """Constructor."""
        super(MRSIModality, self).__init__(path_data=path_data)

    @abstractmethod
    def update_histogram(self):
        """Method to compute histogram and statistics."""
        raise NotImplementedError

    @abstractmethod
    def read_data_from_path(self, path_data=None):
        """Read the data from a path."""
        raise NotImplementedError
