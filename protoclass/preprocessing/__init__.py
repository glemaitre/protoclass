"""
The :mod:`protoclass.preprocessing` modules allows to apply some
pre-processing on medical imaging data.
"""

from .base_normalization import BaseNormalization

from .standalone_normalization import StandaloneNormalization
from .temporal_normalization import TemporalNormalization
from .multisequence_normalization import MultisequenceNormalization

from .standard_time_normalization import StandardTimeNormalization
from .gaussian_normalisation import GaussianNormalization
from .rician_normalisation import RicianNormalization
from .piecewise_linear_normalisation import PiecewiseLinearNormalization

from .mrsi_phase_correction import MRSIPhaseCorrection
from .mrsi_frequency_correction import MRSIFrequencyCorrection
from .mrsi_baseline_correction import MRSIBaselineCorrection
from .water_normalization import WaterNormalization
from .l_norm_normalization import LNormNormalization

__all__ = ['BaseNormalization',
           'StandaloneNormalization',
           'TemporalNormalization',
           'MultisequenceNormalization',
           'StandardTimeNormalization',
           'GaussianNormalization',
           'RicianNormalization',
           'PiecewiseLinearNormalization',
           'MRSIPhaseCorrection',
           'MRSIFrequencyCorrection',
           'MRSIBaselineCorrection',
           'WaterNormalization',
           'LNormNormalization']
