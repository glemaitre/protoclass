"""Basic class for standalone modality (T1, T2, etc.)."""

from abc import ABCMeta, abstractmethod

from .base_modality import BaseModality


class StandaloneModality(BaseModality):
    """Basic class for medical modality (T1, T2, etc.).

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data=None):
        """Constructor."""
        super(StandaloneModality, self).__init__(path_data=path_data)

    @abstractmethod
    def update_histogram(self):
        """Method to compute histogram and statistics."""
        raise NotImplementedError

    @abstractmethod
    def read_data_from_path(self, path_data=None):
        """Read the data from a path."""
        raise NotImplementedError
