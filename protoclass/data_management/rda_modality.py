"""RDA modality class."""

import warnings

import numpy as np

import struct

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

        self.data_ = []
        while True:
            # Read chunck of 8 bits
            db = f.read(8)

            # Early stop if we finished
            if db == '':
                break

            self.data_.append(struct.unpack('d', db))

        return self
