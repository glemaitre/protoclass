""" Test the standalone normalization. """

import os

from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality


def test_validate_modality_gt_wrong_gt():
    """ Test either if an error is raised when the object given as
    ground-truth is not a GTModality object. """

    # Create a T2W object
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_data_t2w = os.path.join(currdir, 'data', 't2w')
    t2w_mod = T2WModality(path_data_t2w)
    
