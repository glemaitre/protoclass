"""
The :mod:`protoclass.data_management` modules allows to handle medical
data format.
"""

from .base_modality import BaseModality
from .standalone_modality import StandaloneModality
from .temporal_modality import TemporalModality
from .multisequence_modality import MultisequenceModality

from .dce_modality import DCEModality
from .t2w_modality import T2WModality
from .dwi_modality import DWIModality
from .gt_modality import GTModality

__all__ = ['BaseModality',
           'StandaloneModality',
           'TemporalModality',
           'MultisequenceModality',
           'DCEModality',
           'T2WModality',
           'DWIModality',
           'GTModality']
