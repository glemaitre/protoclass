"""Gabor filters banks extraction from standalone modality."""
from __future__ import division

import numpy as np

from scipy.ndimage.filters import convolve

from .standalone_extraction import StandaloneExtraction


def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return (1.0 / np.pi * np.sqrt(np.log(2) / 2.0) *
            (2.0 ** b + 1) / (2.0 ** b - 1))


def gabor_filter_3d(frequency, alpha=0, beta=0, gamma=0, bandwidth=1,
                    sigma_x=None, sigma_y=None, sigma_z=None,
                    n_stds=3, scale_x=None, scale_y=None, scale_z=None):
    """Return complex 3D Gabor filter kernel.

    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.

    alpha, beta, gamma : float, optional
        Orientation in radians of the x, y, and z axis.

    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, `sigma_x`
        and `sigma_y` will decrease with increasing frequency. This value is
        ignored if `sigma_x` and `sigma_y` are set by the user.

    sigma_x, sigma_y, sigma_z : float, optional
        Standard deviation in x-, y-, and z-directions. These directions apply
        to the kernel *before* rotation. If `alpha = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.

    scale_x, scale_y, scale_z : float, optional
        In case of using bandwidth but with a wish of different scale for
        the sigma

    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations

    Returns
    -------
    g : complex array
        Complex filter kernel.

    """

    if scale_x is None:
        scale_x = 1.
    if scale_y is None:
        scale_y = 1.
    if scale_z is None:
        scale_z = 1.

    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / (frequency * scale_x)
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / (frequency * scale_y)
    if sigma_z is None:
        sigma_z = _sigma_prefactor(bandwidth) / (frequency * scale_z)

    # Define the different rotation matrix
    rot_mat_x = np.matrix([[1, 0, 0],
                           [0, np.cos(alpha), -np.sin(alpha)],
                           [0, np.sin(alpha), np.cos(alpha)]])
    rot_mat_y = np.matrix([[np.cos(beta), 0, np.sin(beta)],
                           [0, 1, 0],
                           [-np.sin(beta), 0, np.cos(beta)]])
    rot_mat_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                           [np.sin(gamma), np.cos(gamma), 0],
                           [0, 0, 1]])

    # Compute the full rotation matrix
    rot_mat = rot_mat_z * rot_mat_y * rot_mat_x

    # Find the limit of the filter
    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(beta) * np.cos(gamma)),
                     np.abs(n_stds * sigma_y * np.cos(beta) * np.sin(gamma)),
                     np.abs(n_stds * sigma_z * -np.sin(beta)),
                     1))

    y0 = np.ceil(max(np.abs(n_stds * sigma_x * (
        np.sin(alpha) * np.sin(beta) * np.cos(gamma) -
        np.cos(alpha) * np.sin(gamma))),
                     np.abs(n_stds * sigma_y * (
                         np.sin(alpha) * np.sin(beta) * np.sin(gamma) +
                         np.cos(alpha) * np.cos(gamma))),
                     np.abs(n_stds * sigma_z * np.sin(alpha) * np.cos(beta)),
                     1))

    z0 = np.ceil(max(np.abs(n_stds * sigma_x * (
        np.sin(alpha) * np.sin(gamma) +
        np.cos(alpha) * np.sin(beta) * np.cos(gamma))),
                     np.abs(n_stds * sigma_y * (
                         np.cos(alpha) * np.sin(beta) * np.sin(gamma) -
                         np.sin(alpha) * np.cos(gamma))),
                     np.abs(n_stds * sigma_z * np.cos(alpha) * np.cos(beta)),
                     1))

    x, y, z = np.mgrid[-x0:x0 + 1, -y0:y0 + 1, -z0:z0 + 1]

    # Keep the shape of the grid for later reshaping
    grid_shape = x.shape

    # Build a huge matrix with all the coordinates
    pos = np.matrix([x.reshape(-1), y.reshape(-1), z.reshape(-1)])

    # Apply the rotation
    rot_pos = rot_mat * pos

    # Split the data according to the shape of the grid
    rotx = np.reshape(np.array(rot_pos[0, :]), grid_shape)
    roty = np.reshape(np.array(rot_pos[1, :]), grid_shape)
    rotz = np.reshape(np.array(rot_pos[2, :]), grid_shape)

    # Allocate the data with complex type
    g = np.zeros(y.shape, dtype=np.complex)

    # Compute the gaussian enveloppe
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 +
                          roty ** 2 / sigma_y ** 2 +
                          rotz ** 2 / sigma_z ** 2))
    # Normalize the enveloppe
    g /= ((2 * np.pi)**(3. / 2.)) * sigma_x * sigma_y * sigma_z
    # Apply the sinusoidal
    g *= np.exp(1j * 2 * np.pi * (frequency * rotx))

    return g


class GaborBankExtraction(StandaloneExtraction):
    """Edge signal extraction from standalone modality.

    Parameters
    ----------
     base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    frequencies : ndarray, shape (n_frequency, )
        Vector containing the different frequencies of the Gabor filter bank.

    thetas : ndarray, shape (n_theta, )
        Vector containing the different rotation angles of the filter bank
        in x-y plane.

    phis : ndarray, shape (n_phi, )
        Vector containing the different rotations angles of the filer bank
        in the z plane.

    scale_sigmas : ndarray, shape (3, )
        The standard deviations x, y, and z for each direction of the
        filter bank.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from StandaloneModality class.

    roi_data_ : ndarray, shape flexible
        Corresponds to the index to consider in order to fit the data.

    """

    def __init__(self, base_modality, frequencies, alphas, betas, gammas,
                 scale_sigmas):
        super(GaborBankExtraction, self).__init__(base_modality)
        self.frequencies = frequencies
        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.scale_sigmas = scale_sigmas
        self.data_ = None

    def fit(self, modality, ground_truth=None, cat=None):
        """Compute the images images.

        Parameters
        ----------
        modality : object of type TemporalModality
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
        super(GaborBankExtraction, self).fit(modality=modality,
                                             ground_truth=ground_truth,
                                             cat=cat)

        # Create the kernel for the bank of filter
        kernels = []
        for alpha in self.alphas:
            for beta in self.betas:
                for gamma in self.gammas:
                    for frequency in self.frequencies:
                        kernels.append(gabor_filter_3d(
                            frequency,
                            alpha=alpha, beta=beta, gamma=gamma,
                            scale_x=self.scale_sigmas[0],
                            scale_y=self.scale_sigmas[1],
                            scale_z=self.scale_sigmas[2]))

        self.data_ = []
        # We can compute the different convolution
        for kernel in kernels:
            # Compute the real filtering
            self.data_.append(convolve(modality.data_, np.real(kernel)))
            # Compute the imaginary filtering
            self.data_.append(convolve(modality.data_, np.imag(kernel)))

        # Convert to a numpy array
        self.data_ = np.array(self.data_)

        return self

    def transform(self, modality, ground_truth=None, cat=None):
        """Extract the data from the given modality.

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
        ------
        data : ndarray, shape (n_sample, n_feature)
             A matrix containing the features extracted. The number of samples
             is equal to the number of positive label in the ground-truth.

        """
        super(GaborBankExtraction, self).transform(
            modality=modality,
            ground_truth=ground_truth,
            cat=cat)

        # Check that we fitted the data
        if self.data_ is None:
            raise RuntimeError('Fit the data before to extract anything.')

        data = np.zeros((self.roi_data_[0].size, self.data_.shape[0]))
        for feat_dim in range(self.data_.shape[0]):
            feat_data = self.data_[feat_dim]
            data[:, feat_dim] = feat_data[self.roi_data_]

        return data
