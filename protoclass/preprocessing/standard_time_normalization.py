""" Standard time normalization to normalize temporal modality.
"""

from .temporal_normalization import TemporalNormalization


class StandardTimeNormalization(TemporalNormalization):
    """ Standard normalization to normalize temporal modality.

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
        super(StandardTimeNormalization, self).__init__(base_modality)

    def fit(self, modality, ground_truth=None, cat=None):
        """ Method to find the parameters needed to apply the
        normalization.

        Parameters
        ----------
        modality : object
            Object inherated from TemporalModality.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Returns
        -------
        self : object
             Return self.
        """
        super(StandardTimeNormalization, self).fit(modality=modality)

        return self

    def normalize(self, modality):
        """ Method to normalize the given modality using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.
        """
        super(StandardTimeNormalization, self).normalize(modality=modality)

        return self

    def denormalize(self, modality):
        """ Method to denormalize the given modality using the
        fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.
        """
        super(StandardTimeNormalization, self).denormalize(modality=modality)

        return self

