"""Phase correction class for MRSI modality."""

import os
import warnings

import cPickle as pickle
import numpy as np

from joblib import Parallel, delayed

from nmrglue.process.proc_autophase import autops

from ..data_management import MRSIModality
from ..data_management import GTModality

from ..utils.validation import check_modality_inherit
from ..utils.validation import check_filename_pickle_load
from ..utils.validation import check_filename_pickle_save
from ..utils.validation import check_modality


KNOWN_METHOD = ('acme')


def _correct_phase(data, method='acme'):
    """Private function to perform the phase correction in parallel.

    Parameters
    ----------
    data : ndarray, shape (n_samples, )
        Original MRSI signal.

    method : string, optional (default='acme')
        The method to use to perform the correction.

    Returns
    -------
    data : ndarray, shape (n_samples, )
        Phase corrected signal.

    """

    return autops(data, method)


class MRSIPhaseCorrection(object):
    """Phase correction for MRSI modality.

    """

    def __init__(self, base_modality, method='acme'):
        self.base_modality_ = check_modality_inherit(base_modality,
                                                     MRSIModality)
        self.method = method

    def fit(self, modality, ground_truth=None, cat=None):
        """Find the parameters needed to apply the phase correction.

        Parameters
        ----------
        modality : object of type StandaloneModality
            The modality object of interest.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Return
        ------
        self : object
             Return self.

        """

        # Check that the modality is from the template class
        check_modality(modality, self.base_modality_)

        # Check that the data were read during the creation of the modality
        if not modality.is_read():
            raise ValueError('No data have been read during the construction'
                             ' of the modality object.')

        return self

    def transform(self, modality, ground_truth=None, cat=None):
        """Correct the phase of an MRSI modality.

        Parameters
        ----------
        modality : MRSIModality,
            The MRSI modality in which the phase need to be corrected.

        ground-truth : object of type GTModality or None
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        Returns
        -------
        modality : MRSIModality,
            Return the MRSI modaity in which the phase has venn corrected.

        """

        # Check that the modality is from the template class
        check_modality(modality, self.base_modality_)

        # Check that the data were read during the creation of the modality
        if not modality.is_read():
            raise ValueError('No data have been read during the construction'
                             ' of the modality object.')

        # Check that the method requested is known
        if self.method not in KNOWN_METHOD:
            raise ValueError('Unknown method requested for the phase'
                             ' correction.')

        # Get the data from the modality
        data = modality.data_.reshape((modality.data_.shape[0],
                                       modality.data_.shape[1] *
                                       modality.data_.shape[2] *
                                       modality.data_.shape[3])).T

        # Apply the phase correction
        data_ph_corr = Parallel(n_jobs=-1)(delayed(_correct_phase)(signal)
                                           for signal in data)

        modality.data_ = np.reshape(np.array(data_ph_corr).T, (
            modality.data_.shape[0],
            modality.data_.shape[1],
            modality.data_.shape[2],
            modality.data_.shape[3]))

        return modality

    @staticmethod
    def load_from_pickles(filename):
        """ Function to load a normalization object.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Returns
        -------
        bpp : object
            Returns the loaded object.

        """
        # Check the consistency of the filename
        filename = check_filename_pickle_load(filename)
        # Load the pickle
        bpp = pickle.load(open(filename, 'rb'))

        return bpp

    def save_to_pickles(self, filename):
        """ Function to save a normalizatio object using pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Returns
        -------
        None

        """
        # We need to check that the directory where the file will be exist
        dir_pickle = os.path.dirname(filename)
        if not os.path.exists(dir_pickle):
            os.makedirs(dir_pickle)
        # Check the consistency of the filename
        filename = check_filename_pickle_save(filename)
        # Create the pickle file
        pickle.dump(self, open(filename, 'wb'))

        return None
