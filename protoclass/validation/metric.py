#title           :metric.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/18
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Scipy library
import scipy as sp
# Scikit-learn library
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef

def BuildConfusionFromVolume(true_label, pred_label):
    """Function to build the confusion matrix.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    cm: ndarray
        The resulting confusion matrix.
    """

    return confusion_matrix(true_label, pred_label)


def LabelsToSensitivitySpecificity(true_label, pred_label):
    """Function to compute the sensitivity and specificty.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    sens, spec: double
        The resulting sensitivity and specificity.
        The sensitivity is also known as true positive rate, hit rate, or recall.
        The specificity is also known as true negative rate.
    """

    # Compute the confusion matrix
    cm = BuildConfusionFromVolume(true_label, pred_label)

    # Compute the sensitivity and specificity
    if (cm[1, 1] > 0):
        sens = float(cm[1, 1]) / float(cm[1, 1] + cm[1, 0])
    else:
        sens = 0
    if (cm[0, 0] > 0):
        spec = float(cm[0, 0]) / float(cm[0, 0] + cm[0, 1])
    else:
        spec = 0

    return (sens, spec)


def LabelsToPrecisionNegativePredictiveValue(true_label, pred_label):
    """Function to compute the precision and the negative predictive value.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    prec, npv: double
        The resulting precision and negative predictive value.
        The precision is also kown as the positive predictive value.
    """

    # Compute the confusion matrix
    cm = BuildConfusionFromVolume(true_label, pred_label)

    # Compute the sensitivity and specificity
    if (cm[1, 1] > 0):
        prec = float(cm[1, 1]) / float(cm[1, 1] + cm[0, 1])
    else:
        prec = 0
    if (cm[0, 0] > 0):
        npv = float(cm[0, 0]) / float(cm[0, 0] + cm[1, 0])
    else:
        npv = 0

    return (prec, npv)


def LabelsToGeometricMean(true_label, pred_label):
    """Function to compute the geometric mean.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    gmean: double
        The resulting geometric mean of the accuracies measured.

    References
    ----------
    .. [1] Kubat, M. and Matwin, S. "Addressing the curse of imbalanced training sets: 
           one-sided selection" ICML (1997)
    """

    # Compute the confusion matrix
    cm = BuildConfusionFromVolume(true_label, pred_label)

    # Compute the sensitivity and specificity
    sens = float(cm[1, 1]) / float(cm[1, 1] + cm[1, 0])
    spec = float(cm[0, 0]) / float(cm[0, 0] + cm[0, 1])

    # Compute the geometric mean
    gmean = np.sqrt(sens * spec)

    return gmean


def LabelsToAccuracy(true_label, pred_label):
    """Function to compute the accuracy score.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    acc: double
        The resulting accuracy.
    """

    return accuracy_score(true_label, pred_label);

def LabelsToCostValue(true_label,pred_label): 
    """Function to compute cost value parameter.
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    cval: double
        The resulting cost value.
    """
    sens, spec = LabelsToSensitivitySpecificity(true_label, pred_label)
    c10 = 1.5
    c01 = 1 
    cval = (c10 * (1.0 - sens) + c01 * (1.0 - spec)) / (c10+c01)
    return cval


def LabelsToF1score(true_label, pred_label):
    """Function to compute the F1 score.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    f1score: double
        The resulting F1 score.
    """

    return f1_score(true_label, pred_label);


def LabelsToMatthewCorrCoef(true_label, pred_label):
    """Function to compute the Matthew correlation coefficient.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    Returns
    -------
    mcc: double
        The resulting Matthew correlation coefficient.
    """

    return matthews_corrcoef(true_label, pred_label)
	


def LabelsToGeneralizedIndexBalancedAccuracy(true_label, pred_label, M='gmean', alpha=0.1, squared=True):
    """Function to compute the the generalized index of balanced accuracy.
    Parameters
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    M: str (default: 'gmean')
        Name of the metric to consider
        ``sens``:
            use the sensitivity.
        ``'spec'``:
            use the specificity.
        ``'prec'``:
            use the precision.
        ``'npv'``:
            use the negative predictive value.
        ``'gmean'``:
            use the geometric mean.
        ``'acc'``:
            use the accuracy score.
	``'cost'``:
	    use the cost value.
        ``'f1score'``:
            use the F1 score.
        ``'mcc'``:
            use the Matthew correlation coefficient.
    alpha: float (default: 0.1)
        Dominance weight
    squared: bool (default: True)
        If the metric M should be squared
    Returns
    -------
    iba: double
        The resulting generalized index of balanced accuracy.
        
     
	 

    References
    ----------
    .. [1] Garcia, V. and Mollineda, R.A. and Sanchez, J.S. "Theoretical analysis of a performance measure for 
           imbalanced data" ICPR (2010)
    """

    if (M == 'sens'):
        # Compute the sensitivity
        met, _ = LabelsToSensitivitySpecificity(true_label, pred_label)
    elif (M == 'spec'):
        # Compute the specificity
        _, met = LabelsToSensitivitySpecificity(true_label, pred_label)
    elif (M == 'prec'):
        # Compute the precision
        met, _ = LabelsToPrecisionNegativePredictiveValue(true_label, pred_label)
    elif (M == 'npv'):
        # Compute the precision
        _, met = LabelsToPrecisionNegativePredictiveValue(true_label, pred_label)
    elif (M == 'gmean'):
        # Compute the negative predictive value
        met = LabelsToGeometricMean(true_label, pred_label)
    elif (M == 'acc'):
        # Compute the accuracy
        met = LabelsToAccuracy(true_label, pred_label)
    elif (M == 'cost'):
        # Compute the accuracy
        met = LabelsToCostValue(true_label, pred_label)
    elif (M == 'f1score'):
        # Compute the F1 Score
        met = LabelsToF1score(true_label, pred_label)
    elif (M == 'mcc'):
        # Compute the Matthew correlation coefficient
        met  = LabelsToMatthewCorrCoef(true_label, pred_label)
    else:
        raise ValueError('protoclass.metric.GIBA: The metric that you attend to correct is not implemented.')

    # Check the value of alpha is meaningful
    if not ((alpha >= 0.) and (alpha <= 1.)):
        raise ValueError('protoclass.metric.GIBA: The value of alpha sould be set between 0 and 1.')

    # Check if we should square the metric
    if (squared == True):
        met = met ** 2

    # Compute the dominance 
    ### We need the sensitivity and specificity
    sens, spec = LabelsToSensitivitySpecificity(true_label, pred_label)
    ### Compute the dominance as the difference of the sensitiy and specificity
    dom = sens - spec

    # Compute the generalized index of balanced accuracy
    iba = (1 + alpha * dom) * met

    return iba
