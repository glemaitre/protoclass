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
    def __init__(self, base_modality):
        super(MultisequenceNormalization, self).__init__()
        self.base_modality = base_modality
        self._validate_modality()

    def _validate_modality(self):
        """ Check if the provided modality is of interest with the type of
        normalization. """

        # Check that the base modality is a subclass of MultisequenceModality
        if not issubclass(self.base_modality, MultisequenceModality):
            raise ValueError('The base modality provided in the constructor is'
                             'not a Multisequencemodality.')
