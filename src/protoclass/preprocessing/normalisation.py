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
# Scipy library
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

    def __init__(self, mu=0., sigma=1.):
        self.__mu = mu
        self.__sigma = sigma

    def __InitFromData__(self, x, x_max):
        # TODO: put more information about the definition
        # GOAL: Initialisation of the mean and std using the data

        self.__mu = np.mean(x) / (x_max)
        self.__sigma = np.std(x) / (x_max)

    def GetParameters(self):
        # TODO: put more information about the definition
        # GOAL: Get the parameters

        return (self.__mu, self.__sigma)

    def __Parametrization__(self, x, mu, sigma):
        # TODO: put more information about the definition
        # GOAL: Gaussian fitting pdf

        return norm.pdf(x, mu, sigma)

    def __ComputeHistogram__(self,x, x_range):
        # TODO: put more information about the definition
        # GOAL: compute this histogram of x taking care about the data range
        
        # Compute the histogram for the data x with unit bins
        pdf_rel, bin_edges_rel = np.histogram(x, bins=(np.max(x) - np.min(x)), density=True)

        # We need to translate the pdf depending of the range given by x_range
        ### Create an array with unit bins depending of x_range
        ### We need max - min + 1 bins
        pdf_abs = np.zeros((x_range[1] - x_range[0],))
        bin_edges_abs = np.array(range(x_range[0], x_range[1] + 1))
        ### Copy the relative pdf at the right position
        pdf_abs[np.flatnonzero(bin_edges_abs==bin_edges_rel[0])[0] : np.flatnonzero(bin_edges_abs==bin_edges_rel[-1])[0]] = pdf_rel[:]

        return pdf_abs

    def Fit(self, x, x_range):
        # TODO: put more information about the definition
        # GOAL: fit the pdf of the data x. We need to compute the histogram first

        # Check that the value in x_range make sense
        if not ((x_range[0] <= np.min(x))&(x_range[1] >= np.max(x))):
            raise ValueError('normalisation.GaussianNormalisation: Wrong range specifications for the x.')

        # Compute the histogram of x
        x_pdf = self.__ComputeHistogram__(x, x_range)
        
        # Get the initial parameter for the given data
        self.__InitFromData__(x, x_range[1])
        print self.GetParameters()

        # We have to fit the pdf now
        popt, pcov = curve_fit(self.__Parametrization__,
                               np.linspace(0, 1., len(x_pdf)),
                               x_pdf,
                               p0=(self.__mu, self.__sigma))

        # Update the value of the mean and standard deviation
        self.__mu = popt[0]
        self.__sigma = popt[1]
        print self.GetParameters()

        return pcov
