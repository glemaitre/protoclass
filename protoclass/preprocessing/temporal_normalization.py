""" Basic class to normalize temporal modality.
"""

from abc import ABCMeta, abstractmethod

from .base_normalization import BaseNormalization
from ..data_management import TemporalModality
from ..utils.validation import check_modality

class TemporalNormalization(BaseNormalization):
    """ Basic class to normalize temporal modality.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, base_modality):
        super(TemporalNormalization, self).__init__()
        self.base_modality = base_modality
        self._validate_modality()

    def _validate_modality(self):
        """ Check if the provided modality is of interest with the type of
        normalization. """

        # Check that the base modality is a subclass of TemporalModality
        if not issubclass(type(self.base_modality), TemporalModality):
            raise ValueError('The base modality provided in the constructor is'
                             ' not a TemporalModality.')
        else:
            self.base_modality_ = self.base_modality

    @abstractmethod
    def fit(self, modality):
        """ Method to find the parameters needed to apply the
        normalization.

        Parameters
        ----------
        modality : object
            Object inherated from TemporalModality.

        Return
        ------
        self : object
             Return self.
        """
        # Check that the class of modality is the same than the template
        # modality
        check_modality(modality, self.base_modality_)

        return self
