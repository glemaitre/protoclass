"""Basic class for normalization."""

import os

import cPickle as pickle

from abc import ABCMeta, abstractmethod

from ..utils.validation import check_filename_pickle_load
from ..utils.validation import check_filename_pickle_save


class BaseNormalization(object):
    """Basic class for normalization.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        """Constructor """
        self.fit_params_ = {}

    @abstractmethod
    def fit(self, modality, ground_truth=None, cat=None):
        """Method to find the parameters needed to apply the
        normalization."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, modality):
        """Normalize the data after fitting the data."""
        raise NotImplementedError

    @abstractmethod
    def denormalize(self, modality):
        """Reverse the normalization."""
        raise NotImplementedError

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
