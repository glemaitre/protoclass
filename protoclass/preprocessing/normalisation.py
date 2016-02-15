#title           :normalisation.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/26
#version         :0.1
#notes           :
#python_version  :2.7.6 
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Matplotlib library
import matplotlib.pyplot as plt
# Scipy library
from scipy.integrate import simps
### Scipy library for Gaussian statistics
from scipy.stats import norm
### Scipy library for Rician statistics
from scipy.stats import rice
### Scipy library for curve fitting
from scipy.optimize import curve_fit
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
# Multiprocessing library
import multiprocessing

class GaussianNormalisation(object):
    """ Class to perform Gaussian normalisation
    
    Notes
    -----
    """

    def __init__(self, norm_factor=1., mu=0., sigma=1.):
        """Constructor of the GaussianNormalisation object
        
        Parameters
        ----------
        norm_factor: double
            Amplitude of the Gaussian representation
        mu: double
            Mean of the Gaussian representation
        sigma: double
            Standard deviation of the Gaussian representation
    
        """

        self.__norm_factor = norm_factor
        self.__mu = mu
        self.__sigma = sigma

    def __init_from_data__(self, x, norm_factor):
        """Function to initialise the class members using data
        
        Parameters
        ----------
        x: ndarray
            Array containing the data from which the mean and
            standard deviation are derived
        norm_factor: double
            Amplitude of the Gaussian representation

        """

        self.__norm_factor = norm_factor
        self.__mu = np.mean(x)
        self.__sigma = np.std(x)

    def get_parameters(self):
        """Function to get the parameter of the object
        
        Returns
        -------
        params: tuple 
            Return a tuple with the amplitude, mean and std of the 
            Gaussian representation
    
        """
        
        return (self.__norm_factor, self.__mu, self.__sigma)

    def __parametrization__(self, x, norm_factor, mu, sigma):
        """Function to parametrise the Gaussian PDF
        
        Parameters
        ----------
        x : 1d-array
            Array containing the x values in order to compute the PDF
        norm_factor: double
            Amplitude of the Gaussian representation
        mu: double
            Mean of the Gaussian representation
        sigma: double
            Standard deviation of the Gaussian representation

        Returns
        -------
        pdf: 1d-array 
            Return the Gaussian PDF according the parameters
    
        """
        
        return norm.pdf(x, mu, sigma) / norm_factor

    def __compute_histogram__(self, x, x_range):
        """Function to compute the histogram of the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the data
        x_range: tuple
            Tuple containing the minimum and maxumum to consider
            to build the histogram of the data x

        Returns
        -------
        pdf: 1d-array 
            Return the PDF of x using the range provided by the user
        bin_edge: 1d-array
            Return the bins corresponding to the PDF
    
        """
        
        # Compute the histogram for the data x with unit bins
        pdf_rel, bin_edges_rel = np.histogram(x, bins=(np.max(x) - np.min(x)), density=True)

        # We need to translate the pdf depending of the range given by x_range
        ### Create an array with unit bins depending of x_range
        ### We need max - min + 1 bins
        pdf_abs = np.zeros((x_range[1] - x_range[0],))
        bin_edges_abs = np.array(range(x_range[0], x_range[1] + 1))
        ### Copy the relative pdf at the right position
        pdf_abs[np.flatnonzero(bin_edges_abs==bin_edges_rel[0])[0] : np.flatnonzero(bin_edges_abs==bin_edges_rel[-1])[0]] = pdf_rel[:]

        return (pdf_abs, bin_edges_abs)

    def fit(self, x, x_range):
        """Function to find the best member parameters to fit 
           the class parametrisation using Levenverg-Marquardt
           curve fitting
        
        Parameters
        ----------
        x: ndarray
            Array containing the data
        x_range: tuple
            Tuple containing the minimum and maxumum to consider
            to build the histogram of the data x

        Returns
        -------
        error_fit: double
            Return the fitting error
    
        """

        # Check that the value in x_range make sense
        if not ((x_range[0] <= np.min(x))&(x_range[1] >= np.max(x))):
            raise ValueError('normalisation.GaussianNormalisation: Wrong range specifications for the x.')

        # Compute the histogram of x
        x_pdf, x_pdf_bins = self.__compute_histogram__(x, x_range)

        # We need to normalise the absicisse of the histogram between 0 and 1.
        norm_factor = x_range[1]
        x = x.astype(float) / norm_factor
        x_pdf_bins = x_pdf_bins.astype(float) / norm_factor

        # Get the initial parameter for the given data
        self.__init_from_data__(x, norm_factor)
        
        # We have to fit the pdf now
        popt, pcov = curve_fit(self.__parametrization__,
                               x_pdf_bins[:-1],
                               x_pdf,
                               p0=(self.__norm_factor, self.__mu, self.__sigma))

        # Update the value of the mean and standard deviation
        ### We need to unormalise the mean and std
        ###  WE NEED TO DISCUSS TO DECIDE IF THE NORMALISATION FACTOR SHOULD BE A CONSTANT AND NOT BE OPTIMISED
        self.__norm_factor = popt[0]
        self.__mu = popt[1]
        self.__sigma = popt[2]

        print 'The parameters fitted are {}'.format(self.GetParameters())

        return pcov

    def normalise(self, x):
        """Function to normalise the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the unormalised data

        Returns
        -------
        x_normalised: ndarray 
            Return the data x normalised
    
        """

        return (x - (self.__mu * self.__norm_factor)) / (self.__sigma * self.__norm_factor)

    def denormalise(self, x):
        """Function to unormalise the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the normalised data

        Returns
        -------
        x_unormalised: ndarray 
            Return the data x unormalised
    
        """

        return x * (self.__sigma * self.__norm_factor) + (self._mu * self.__norm_factor)

class RicianNormalisation(object):
    """ Class to perform Rician normalisation
    
    Notes
    -----
    """

    def __init__(self, norm_factor=1., v=0., loc=0., sigma=1.):
        """Constructor of the GaussianNormalisation object
        
        Parameters
        ----------
        norm_factor: double
            Amplitude of the Rician representation
        v: double
            Distance to the center of the bivariate distribution
        loc: double
            Initial shift of the Rician representation
        sigma: double
            Scale of the Rician representation
    
        """

        self.__norm_factor = norm_factor
        self.__v = v
        self.__loc = loc
        self.__sigma = sigma

    def __init_from_data__(self, x, norm_factor):
        """Function to initialise the class members using data
        
        Parameters
        ----------
        x: ndarray
            Array containing the data from which the mean and
            standard deviation are derived
        norm_factor: double
            Amplitude of the Rician representation

        """
        
        self.__norm_factor = norm_factor
        self.__v = np.mean(x)
        self.__loc = np.min(x)
        self.__sigma = np.std(x)

    def get_parameters(self):
        """Function to get the parameter of the object
        
        Returns
        -------
        params: tuple 
            Return a tuple the parameters of the Rician representation
    
        """

        return (self.__norm_factor, self.__v, self.__loc, self.__sigma)

    def __parametrization__(self, x, norm_factor, v, loc, sigma):
        """Function to parametrise the Rician PDF
        
        Parameters
        ----------
        x : 1d-array
            Array containing the x values in order to compute the PDF
        norm_factor: double
            Amplitude of the Rician representation
        v: double
            Distance to center of the Rician representation
        loc: double
            Location of the Rician representation
        sigma: double
            Scale of the Rician representation

        Returns
        -------
        pdf: 1d-array 
            Return the Gaussian PDF according the parameters
    
        """

        return rice.pdf(x, v, loc, sigma) / norm_factor

    def __compute_histogram__(self, x, x_range):
        """Function to compute the histogram of the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the data
        x_range: tuple
            Tuple containing the minimum and maxumum to consider
            to build the histogram of the data x

        Returns
        -------
        pdf: 1d-array 
            Return the PDF of x using the range provided by the user
        bin_edge: 1d-array
            Return the bins corresponding to the PDF
    
        """        
        # Compute the histogram for the data x with unit bins
        pdf_rel, bin_edges_rel = np.histogram(x, bins=(np.max(x) - np.min(x)), density=True)

        # We need to translate the pdf depending of the range given by x_range
        ### Create an array with unit bins depending of x_range
        ### We need max - min + 1 bins
        pdf_abs = np.zeros((x_range[1] - x_range[0],))
        bin_edges_abs = np.array(range(x_range[0], x_range[1] + 1))
        ### Copy the relative pdf at the right position
        pdf_abs[np.flatnonzero(bin_edges_abs==bin_edges_rel[0])[0] : np.flatnonzero(bin_edges_abs==bin_edges_rel[-1])[0]] = pdf_rel[:]

        return (pdf_abs, bin_edges_abs)

    def fit(self, x, x_range):
        """Function to find the best member parameters to fit 
           the class parametrisation using Levenverg-Marquardt
           curve fitting
        
        Parameters
        ----------
        x: ndarray
            Array containing the data
        x_range: tuple
            Tuple containing the minimum and maxumum to consider
            to build the histogram of the data x

        Returns
        -------
        error_fit: double
            Return the fitting error
    
        """
        # Check that the value in x_range make sense
        if not ((x_range[0] <= np.min(x))&(x_range[1] >= np.max(x))):
            raise ValueError('normalisation.GaussianNormalisation: Wrong range specifications for the x.')

        # Compute the histogram of x
        x_pdf, x_pdf_bins = self.__compute_histogram__(x, x_range)

        # We need to normalise the absicisse of the histogram between 0 and 1.
        norm_factor = x_range[1]
        x = x.astype(float) / norm_factor
        x_pdf_bins = x_pdf_bins.astype(float) / norm_factor
        
        # Get the initial parameter for the given data
        self.__init_from_data__(x, norm_factor)
        
        # We have to fit the pdf now
        popt, pcov = curve_fit(self.__parametrization__,
                               x_pdf_bins[:-1],
                               x_pdf,
                               p0=(self.__norm_factor, self.__v, self.__loc, self.__sigma))

        # Update the value of the mean and standard deviation
        ### We need to unormalise the mean and std
        self.__norm_factor = popt[0]
        self.__v = popt[1]
        self.__loc = popt[2]
        self.__sigma = popt[3]

        print 'The parameters fitted are {}'.format(self.GetParameters())

        return pcov

    def normalise(self, x):
        """Function to normalise the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the unormalised data

        Returns
        -------
        x_normalised: ndarray 
            Return the data x normalised
    
        """

        return ( (x - rice.mean(self.__v, self.__loc, self.__sigma) * self.__norm_factor) / 
                 (self.__sigma * self.__norm_factor) )

    def denormalise(self, x):
        """Function to unormalise the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the normalised data

        Returns
        -------
        x_unormalised: ndarray 
            Return the data x unormalised
    
        """

        return ( x * (self.__sigma * self.__norm_factor) + 
                 (rice.mean(self.__v, self.__loc, self.__sigma) * self.__norm_factor) )

class LinearNormalisationByParts(object):
    """ Class to perform linear normalisation by parts
    
    Notes
    -----
    """

    def __init__(self, atlas, min_perc=2., max_perc=98.):
        """Constructor of the GaussianNormalisation object
        
        Parameters
        ----------
        atlas: array
            Array of the mean value of landmarks dedicting from the full dataset
            Refers to the function FindLandmarksDataset()

        min_perc: float (default=2.)
            Minimum percentiles to consider

        max_perc: float (default=98.)
            Maximum percentiles to consider
    
        """

        self.__atlas = atlas
        self.__percentiles = np.zeros(self.__atlas.shape)
        self.__min_perc = min_perc
        self.__max_perc = max_perc
        # Find the number of landmarks to compute depending of the atlas provided to the constructor
        self.__n_landmarks = atlas.size

    def __init_from_data__(self, x):
        """Function to initialise the class members using data
        
        Parameters
        ----------
        x: ndarray
            Array containing the data from which the landmarks have to be extracted

        """

        from protoclass.tool.dicom_manip import __VolumePercentilesFromData__

        # Compute the percentiles from the data
        self.__percentiles = __VolumePercentilesFromData__(x, self.__n_landmarks, self.__min_perc, self.__max_perc)

    def get_parameters(self):
        """Function to get the parameter of the object
        
        Returns
        -------
        params: tuple 
            Return a tuple with the value of the atlas to be fitted and the landmarks found for the curren volume
    
        """
        
        return (self.__atlas, self.__percentiles)

    def fit(self, x):
        """Function to find the best member parameters to make the mapping
        
        Parameters
        ----------
        x: ndarray
            Array containing the data

    
        """

        # Initalise the object using the data provided
        self.__init_from_data__(x)


    def __rescale_parts__(self, x, x_norm, org_inf, org_sup, pro_inf, pro_sup):
        """Function to rescale for a given parts
        
        Parameters
        ----------
        x: ndarray
            Array containing the unormalised data

        x_norm: ndarray
            Array containing part of the normalise data

        org_inf: float
            Original lower bound

        org_sup: float
            Original superior bound

        pro_inf: float
            Projected lower bound

        pro_sup: float
            Projected superior bound

        Returns
        -------
        x_norm: ndarray 
            Return the data x normalised for this parts
    
        """

        # Find the index of the data which have to be normalised in this round
        idx = np.nonzero(np.bitwise_and(x >= org_inf, x < org_sup))
        # Apply the linear rescaling
        x_norm[idx] = pro_inf + np.multiply((x[idx] - org_inf), np.divide(pro_sup - pro_inf, org_sup - org_inf))

        return x_norm

    def normalise(self, x):
        """Function to normalise the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the unormalised data

        x_norm: ndarray
            Array containing part of the normalise data

        Returns
        -------
        x_norm: ndarray 
            Return the data x normalised for this parts
    
        """

        x_norm = x.copy()
        # We need to go through the n_landmarks - 1 parts
        for ld in range(self.__n_landmarks - 1):
            x_norm = self.__rescale_parts__(x, x_norm, 
                                            self.__percentiles[ld], self.__percentiles[ld+1],
                                            self.__atlas[ld], self.__atlas[ld+1])

        return x_norm

    def denormalise(self, x):
        """Function to unormalise the data
        
        Parameters
        ----------
        x: ndarray
            Array containing the normalised data

        Returns
        -------
        x_unormalised: ndarray 
            Return the data x unormalised
    
        """

        x_norm = x.copy()
        # We need to go through the n_landmarks - 1 parts
        for ld in range(self.__n_landmarks - 1):
            x_norm = self.__rescale_parts__(x, x_norm, 
                                            self.__atlas[ld], self.__atlas[ld+1],
                                            self.__percentiles[ld], self.___percentiles[ld+1])

        return x_norm
