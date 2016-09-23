"""
The :mod:`protoclass.data_management` modules allows to handle medical
data format.
"""

from .base_modality import BaseModality
from .standalone_modality import StandaloneModality
from .temporal_modality import TemporalModality
from .multisequence_modality import MultisequenceModality
from .mrsi_modality import MRSIModality

from .dce_modality import DCEModality
from .t2w_modality import T2WModality
from .adc_modality import ADCModality
from .dwi_modality import DWIModality
from .rda_modality import RDAModality
from .gt_modality import GTModality

from .oct_modality import OCTModality

__all__ = ['BaseModality',
           'StandaloneModality',
           'TemporalModality',
           'MultisequenceModality',
           'MRSIModality',
           'DCEModality',
           'T2WModality',
           'ADCModality',
           'DWIModality',
           'RDAModality',
           'GTModality',
           'OCTModality']
