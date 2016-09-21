"""RDA modality class."""

import warnings

import numpy as np

import struct

from scipy.fftpack import fft
from scipy.fftpack import fftshift

from .mrsi_modality import MRSIModlality

from ..utils.validation import check_rda_filename


class RDAModality(MRSIModlality):
    """Class to handle RDA-MRSI modality.

    Parameters
    ----------
    path_data : str, optional (default=None)
         The folder in which the data are stored.

    Attributes
    ----------
    path_data_ : string
        Location of the data.

    data_ : ndarray, shape (Y, X, Z)
        The different volume of the T2W volume. The data are saved in
        Y, X, Z ordered.

    metadata_ : dict
        Dictionnary which contain the MRI sequence information. Note that the
        information are given in the original ordering (X, Y, Z), which is
        different from the organisation of `data_` which is (Y, X, Z).

    """

    def __init__(self, path_data=None):
        super(RDAModality, self).__init__(path_data=path_data)

    def update_histogram(self, nb_bins=None):
        """Method to compute some histogram."""
        raise NotImplementedError

    def read_data_from_path(self, path_data=None):
        """Read T2W images which represent a single 3D volume.

        Parameters
        ----------
        path_data : str or None, optional (default=None)
            Path to the standalone modality data.

        Returns
        -------
        self : object
           Returns self.

        """
        # Check the consistency of the path data
        if self.path_data_ is not None and path_data is not None:
            # We will overide the path and raise a warning
            warnings.warn('The data path will be overriden using the path'
                          ' given in the function.')
            self.path_data_ = check_rda_filename(path_data)
        elif self.path_data_ is None and path_data is not None:
            self.path_data_ = check_rda_filename(path_data)
        elif self.path_data_ is None and path_data is None:
            raise ValueError('You need to give a path_data from where to read'
                             ' the data.')

        # Open the file
        f = open(self.path_data_, 'r')
        # Define what is the string for the end of the header
        str_end_header = '>>> End of header <<<\r\n'

        # Read the different field
        bin_dict = {}
        while True:
            # Read one line and put it in a huge dictionnary
            # We will be insterested only by numbers
            line = f.readline()

            # Make an early stop if this is the end of the header
            if line == str_end_header:
                break

            try:
                # Try to find the split in the string
                idx_split = line.index(':')
                idx_end_line = line.index('\r')
                bin_dict[line[:idx_split]] = line[idx_split + 2:idx_end_line]

            except ValueError:
                continue

        # Build the metadata dictionary from all the header that we read
        self.metadata_ = {}
        # Get the size of the CSI grid
        self.metadata_['size'] = (int(bin_dict['NumberOfRows']),
                                  int(bin_dict['NumberOfColumns']),
                                  int(bin_dict['NumberOf3DParts']))
        # Get the origin of the CSI grid
        self.metadata_['origin'] = (float(bin_dict['PositionVector[0]']),
                                    float(bin_dict['PositionVector[1]']),
                                    float(bin_dict['PositionVector[2]']))
        # Get the direction of the CSI grid
        self.metadata_['direction'] = (float(bin_dict['RowVector[0]']),
                                       float(bin_dict['RowVector[1]']),
                                       float(bin_dict['RowVector[2]']),
                                       float(bin_dict['ColumnVector[0]']),
                                       float(bin_dict['ColumnVector[1]']),
                                       float(bin_dict['ColumnVector[2]']),
                                       float(bin_dict['VOINormalSag']),
                                       float(bin_dict['VOINormalCor']),
                                       float(bin_dict['VOINormalTra']))
        # Get the spacing of the CSI grid
        self.metadata_['spacing'] = (float(bin_dict['PixelSpacingRow']),
                                     float(bin_dict['PixelSpacingCol']),
                                     float(bin_dict['PixelSpacing3D']))
        # Get the information about TE, TR, and flip-angle
        self.metadata_['TR'] = float(bin_dict['TR'])
        self.metadata_['TE'] = float(bin_dict['TE'])
        self.metadata_['flip-angle'] = float(bin_dict['FlipAngle'])
        # Geth the size of each spectra
        self.metadata_['spectra-size'] = int(bin_dict['VectorSize'])
        # Store the size of CSI scan
        self.metadata_['CSIMatrixSizeOfScan'] = (
            int(bin_dict['CSIMatrixSizeOfScan[0]']),
            int(bin_dict['CSIMatrixSizeOfScan[1]']),
            int(bin_dict['CSIMatrixSizeOfScan[2]']))
        # Store the information about the VOI
        self.metadata_['VOIorigin'] = (float(bin_dict['VOIPositionSag']),
                                       float(bin_dict['VOIPositionCor']),
                                       float(bin_dict['VOIPositionTra']))
        # Store the information about the FOV
        self.metadata_['FOV'] = (float(bin_dict['FoVHeight']),
                                 float(bin_dict['FoVWidth']),
                                 float(bin_dict['FoV3D']))
        # Store the information about the VOI FOV
        self.metadata_['VOIFoV'] = (float(bin_dict['VOIThickness']),
                                    float(bin_dict['VOIPhaseFOV']),
                                    float(bin_dict['VOIReadoutFOV']))
        # Store the information about the rotation in the plane
        self.metadata_['VOIRotationInPlane'] = float(
            bin_dict['VOIRotationInPlane'])

        data = []
        while True:
            # Read chunck of 8 bits
            db = f.read(8)

            # Early stop if we finished
            if db == '':
                break

            data.append(struct.unpack('d', db)[0])

        data_org = np.zeros((2,
                             self.metadata_['spectra-size'],
                             self.metadata_['size'][1],
                             self.metadata_['size'][0],
                             self.metadata_['size'][2]))

        total_idx = 0
        for z in range(self.metadata_['size'][2]):
            for x in range(self.metadata_['size'][0]):
                for y in range(self.metadata_['size'][1]):
                    for sp in range(self.metadata_['spectra-size']):
                        for real_imag in range(2):
                            data_org[real_imag, sp, y, x, z] = data[total_idx]
                            total_idx += 1

        self.data_ = (data_org[0, :, :, :, :] + 1j * data_org[1, :, :, :, :])

        # Transform the signal in the frequency domain
        for y in range(self.data_.shape[2]):
            for x in range(self.data_.shape[1]):
                for z in range(self.data_.shape[3]):
                    self.data_[:, y, x, z] = fftshift(fft(
                        self.data_[:, y, x, z]))

        return self
