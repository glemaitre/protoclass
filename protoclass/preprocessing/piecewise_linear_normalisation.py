"""Piecewise-linear normalization to normalize standalone modality"""

import os

import numpy as np

from .standalone_normalization import StandaloneNormalization

from ..utils import check_npy_filename


class PiecewiseLinearNormalization(StandaloneNormalization):
    """Piecewise-linear normalization to normalize standalone modality.

    Parameters
    ----------

    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    nb_landmarks : int, optional (default=5)
        The number of landmarks to consider to divide the signal.

    Attributes
    ----------

    landmarks_model : ndarray, shape (nb_landmarks, )
        The landmarks which will be match during the normalization.

    fit_params_ : ndarray, shape (nb_landmarks, )
        The array of the landmarks for the current fitted modality

    is_fitted_ : bool
        Boolean to know if the `fit` function has been already called.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, nb_landmarks=5):
        super(PiecewiseLinearNormalization, self).__init__(base_modality)

        self.nb_landmarks = nb_landmarks
        self.fit_params_ = None
        self.is_model_fitted_ = False
        self.is_fitted_ = False
        self.counter_partial_fit = 0

    def partial_fit_model(self, modality, ground_truth=None, cat=None,
                          refit=False, verbose=True):
        """Online fitting to update the landmarks used for the normalization.

        Parameters
        ----------
        modality : object
            Object inherated from StandalineModality.

        ground-truth : object of type GTModality or None, optional
            (default=None)
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None, optional (default=None)
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        refit : bool, optional (default=False)
            Either to refit the model or not.

        verbose : bool, optional (default=True)
            Whether to show the fitting process information.

        Returns
        -------
        self : object
             Return self.

        """

        # Call the parent fit such that we get information about the roi
        super(PiecewiseLinearNormalization, self).fit(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Compute which percentile we should compute for the current modality
        landmarks_perc = np.linspace(0., 100., num=self.nb_landmarks,
                                     endpoint=True)

        # Find the value of the percentile
        landmarks = np.array([np.percentile(modality.data_[self.roi_data_],
                                            land)
                              for land in landmarks_perc])

        # Compute the mean model if there is already computed data
        if self.counter_partial_fit > 0:
            # Check the number of landmarks make sense
            if landmarks.size != self.landmarks_model.size:
                raise ValueError('The number of landmarks already computed are'
                                 ' different from the one you are computing'
                                 ' now.')
            # Compute the running mean
            self.counter_partial_fit += 1
            for idx_land, (avg_land, curr_land) in enumerate(zip(
                    self.landmarks_model,
                    landmarks)):
                self.landmarks_model[idx_land] = (avg_land +
                                             ((curr_land - avg_land) /
                                              float(self.counter_partial_fit)))
        elif (self.counter_partial_fit == 0) or refit:
            # Affect directly the landamarks which were found
            self.landmarks_model = landmarks
            self.counter_partial_fit = 1

        self.is_model_fitted_ = True

        return self

    def load_model(self, filename):
        """Load a model used at the time to align the RMSE.

        Parameters
        ----------
        filename : str
            The path to npy file with the data of the model inside.

        Returns
        -------
        self : object
            Returns self.

        """
        # Check that the filename is ok
        filename = check_npy_filename(filename)

        # Load the model
        self.landmarks_model = np.load(filename)

        # Store that we loaded the model
        self.is_model_fitted_ = True

        return self

    def save_model(self, filename):
        """Store the model into an npy file.

        Parameters
        ----------
        filename : str
            The path where to store the model.

        Returns
        -------
        None

        """
        # Check that the model has been fitted
        if not self.is_model_fitted_:
            raise ValueError('Fit a model before to save it.')

        # Check that the file is an npy file
        if not filename.endswith('.npy'):
            raise ValueError('The file provided needs to be of `npy`'
                             ' extension.')

        dir_storage = os.path.dirname(filename)
        if not os.path.exists(dir_storage):
            os.makedirs(dir_storage)

        np.save(filename, self.landmarks_model)

        return None

    def fit(self, modality, ground_truth=None, cat=None):
        """Method to find the parameters needed to apply the normalization.

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

        Returns
        -------
        self : object
             Return self.

        """

        # Check that a model has been fitted
        if not self.is_model_fitted_:
            raise ValueError('A model needs to be either loaded or fitted.')

        # Call the parent fit such that we get information about the roi
        super(PiecewiseLinearNormalization, self).fit(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Compute which percentile we should compute for the current modality
        landmarks_perc = np.linspace(0., 100., num=self.nb_landmarks,
                                     endpoint=True)

        # Find the value of the percentile
        self.fit_params_ = np.array([np.percentile(
            modality.data_[self.roi_data_],
            land) for land in landmarks_perc])

        self.is_fitted_ = True

        return self

    @staticmethod
    def _rescale_parts(org_data, res_data, land_inf, land_sup,
                       land_mod_inf, land_mod_sup):
        """Private function to rescale the data using landmarks."""

        # Find the index of the data of interest
        idx_data = np.nonzero(np.bitwise_and(org_data >= land_inf,
                                             org_data < land_sup))

        # Rescale the data linearaly
        factor = (land_mod_sup - land_mod_inf) / (land_sup - land_inf)
        res_data[idx_data] = land_mod_inf + ((org_data[idx_data] - land_inf) *
                                             factor)
        return res_data

    def normalize(self, modality):
        """Normalize the data using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.


        """

        # Make a copy of the data related to the modality
        data_norm = modality.data_.copy()

        # Make the rescaling
        for interval in range(self.landmarks_model.size - 1):
            data_norm = self._rescale_parts(modality.data_, data_norm,
                                            self.fit_params_[interval],
                                            self.fit_params_[interval + 1],
                                            self.landmarks_model[interval],
                                            self.landmarks_model[interval + 1])

        modality.data_ = data_norm
        # Update the histogram
        modality.update_histogram()

        return modality

    def denormalize(self, modality):
        """Denormalize the data using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be denormalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be denormalized.


        """

        # Make a copy of the data related to the modality
        data_norm = modality.data_.copy()

        # Make the rescaling
        for interval in range(self.landmarks_model.size - 1):
            data_norm = self._rescale_parts(modality.data_, data_norm,
                                            self.landmarks_model[interval],
                                            self.landmarks_model[interval + 1],
                                            self.fit_params_[interval],
                                            self.fit_params_[interval + 1])

        modality.data_ = data_norm
        # Update the histogram
        modality.update_histogram()

        return modality

