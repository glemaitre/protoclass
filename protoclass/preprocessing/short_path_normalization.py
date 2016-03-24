""" Short path normalization to normalize temporal modality.
"""

from .temporal_normalization import TemporalNormalization


class ShortPathNormalization(TemporalNormalization):
    """ Short path normalization to normalize temporal modality.

    Parameters
    ----------

    base_modality: object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    Attributes
    ----------

    base_modality_: object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.
    """

    def __init__(self, base_modality):
        super(ShortPathNormalization, self).__init__(base_modality)

