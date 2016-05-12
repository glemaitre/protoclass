""" Base class for temporal modality (DCE).
"""

from abc import ABCMeta, abstractmethod

from .base_modality import BaseModality


class TemporalModality(BaseModality):
    """ Basic class for temporal medical modality (DCE).

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data=None):
        """ Constructor. """
        super(TemporalModality, self).__init__(path_data=path_data)

    @abstractmethod
    def update_histogram(self):
        """ Method to compute histogram and statistics. """
        raise NotImplementedError

    @abstractmethod
    def read_data_from_path(self, path_data=None):
        """Read the data from a path."""
        raise NotImplementedError
