################
`protoclass` API
################

This is the full API documentation of the `protoclass` toolbox.

.. _data_management_ref:

Data management module
======================

.. automodule:: protoclass.data_management
     :no-members:
     :no-inherited-members:

Classes
-------
.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     data_management.ADCModality
     data_management.T2WModality
     data_management.DWIModality
     data_management.DCEModality
     data_management.GTModality
     data_management.OCTModality

.. _preprocessing_ref:

Pre-processing module
=====================

.. automodule:: protoclass.preprocessing
     :no-members:
     :no-inherited-members:

Standalone related methods
--------------------------
.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     preprocessing.GaussianNormalization
     preprocessing.RicianNormalization
     preprocessing.PiecewiseLinearNormalization

Temporal related methods
------------------------
.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     preprocessing.StandardTimeNormalization

MRSI related methods
--------------------
.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     preprocessing.WaterNormalization
     preprocessing.LNormNormalization


MRSI related methods
------------------------
.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     preprocessing.MRSIPhaseCorrection
     preprocessing.MRSIFrequencyCorrection
     preprocessing.MRSIBaselineCorrection

.. _extraction_ref:

Extraction module
=================

.. automodule:: protoclass.extraction
     :no-members:
     :no-inherited-members:

Standalone related methods
--------------------------
.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     extraction.IntensitySignalExtraction
     extraction.EdgeSignalExtraction
     extraction.HaralickExtraction
     extraction.PhaseCongruencyExtraction
     extraction.GaborBankExtraction
     extraction.DCTExtraction
     extraction.SpatialExtraction
     extraction.gabor_filter_3d
     
Temporal related methods
------------------------

.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     extraction.EnhancementSignalExtraction
     extraction.BrixQuantificationExtraction
     extraction.PUNQuantificationExtraction
     extraction.SemiQuantificationExtraction
     extraction.ToftsQuantificationExtraction
     extraction.WeibullQuantificationExtraction

MRSI related methods
------------------------

.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     extraction.RelativeQuantificationExtraction

Utils module
============

.. automodule:: protoclass.utils
     :no-members:
     :no-inherited-members:

Functions
---------
.. currentmodule:: protoclass

.. autosummary::
     :toctree: generated/

     utils.find_nearest
     utils.check_path_data
     utils.check_modality
     utils.check_img_filename
     utils.check_npy_filename
     utils.check_filename_pickle_load
     utils.check_filename_pickle_save
     utils.check_modality_inherit
     utils.check_modality_gt
