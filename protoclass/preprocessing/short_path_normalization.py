""" Short path normalization to normalize temporal modality.
"""

from .temporal_normalization import TemporalNormalization


class ShortPathNormalization(TemporalNormalization):
    """ Short path normalization to normalize temporal modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.
    """

    def __init__(self, base_modality):
        super(ShortPathNormalization, self).__init__(base_modality)

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
        super(ShortPathNormalization, self).fit(modality=modality)

        return self
