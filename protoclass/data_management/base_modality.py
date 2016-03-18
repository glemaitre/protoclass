""" Basic class for modality.
"""

from abc import ABCMeta, abstractmethod


class BaseModality(object):
    """ Basic class for medical modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, path_data):
        self.path_data_ = path_data

    @abstractmethod
    def read_data_from_path(self):
        """ Method allowing to read the data. """
        raise NotImplementedError
