"""
The :mod:`protoclass.preprocessing` modules allows to apply some
pre-processing on medical imaging data.
"""

from .base_normalization import BaseNormalization

from .standalone_normalization import StandaloneNormalization
from .temporal_normalization import TemporalNormalization
from .multisequence_normalization import MultisequenceNormalization

from .short_path_normalization import ShortPathNormalization
from .gaussian_normalisation import GaussianNormalization

__all__ = ['BaseNormalization',
           'StandaloneNormalization',
           'TemporalNormalization',
           'MultisequenceNormalization',
           'ShortPathNormalization',
           'GaussianNormalization']
