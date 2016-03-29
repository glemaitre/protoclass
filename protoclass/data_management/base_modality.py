""" Basic class for modality.
"""

from abc import ABCMeta, abstractmethod

from ..utils.validation import check_path_data


class BaseModality(object):
    """ Basic class for medical modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data=None):
        """ Constructor. """
        if path_data is not None:
            self.path_data_ = check_path_data(path_data)
        else:
            self.path_data_ = None
        self.data_ = None

    @abstractmethod
    def _update_histogram(self):
        """ Method to compute histogram and statistics. """
        raise NotImplementedError

    @abstractmethod
    def read_data_from_path(self, path_data):
        """ Method allowing to read the data. """
        raise NotImplementedError

    def is_read(self):
        """ Function to know if the data have been read.

        Return
        ------
        is_read : bool
            If True, the data have been read at least once.
        """
        if self.data_ is None:
            return False
        else:
            return True
