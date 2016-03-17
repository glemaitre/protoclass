""" Methods to compute classification statistics from the
prediction and labels.
"""
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


def build_confusion_from_volume(true_label, pred_label):
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


def labels_to_sensitivity_specificity(true_label, pred_label):
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
        The sensitivity is also known as true positive rate, hit rate,
        or recall. The specificity is also known as true negative rate.
    """

    # Compute the confusion matrix
    cm = build_confusion_from_volume(true_label, pred_label)

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


def labels_to_precision_negative_predictive_value(true_label, pred_label):
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
    cm = build_confusion_from_volume(true_label, pred_label)

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


def labels_to_geometric_mean(true_label, pred_label):
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
    .. [1] Kubat, M. and Matwin, S. "Addressing the curse of
    imbalanced training sets: one-sided selection" ICML (1997)
    """

    # Compute the confusion matrix
    cm = build_confusion_from_volume(true_label, pred_label)

    # Compute the sensitivity and specificity
    sens = float(cm[1, 1]) / float(cm[1, 1] + cm[1, 0])
    spec = float(cm[0, 0]) / float(cm[0, 0] + cm[0, 1])

    # Compute the geometric mean
    gmean = np.sqrt(sens * spec)

    return gmean


def labels_to_accuracy(true_label, pred_label):
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

    return accuracy_score(true_label, pred_label)


def cost_with_bias(sens, spec, bias_pos, bias_neg):
    """Function to compute cost given the sensitivity and specificity.
    ----------
    sens: float
        Sensitivity.
    spec: float
        Specificity.
    bias_pos: float
        Constant influencing the bias of the positive class.
    bias_neg: float
        Constant influecing the bias of the negative class.
    Returns
    -------
    cval: double
        The resulting cost value.
    """

    return (bias_pos * (1.0 - sens) +
            bias_neg * (1.0 - spec)) / (bias_pos + bias_neg)


def labels_to_cost_value(true_label, pred_label, bias_pos=1.5, bias_neg=1.):
    """Function to compute cost given the ground-truth label
    and predicted label.
    ----------
    true_label: ndarray
        Ground-truth array.
    pred_label: ndarray
        Prediction label given by the machine learning method.
    bias_pos: float
        Constant influencing the bias of the positive class.
    bias_neg: float
        Constant influecing the bias of the negative class.
    Returns
    -------
    cval: double
        The resulting cost value.
    """

    sens, spec = labels_to_sensitivity_specificity(true_label, pred_label)

    return cost_with_bias(sens, spec, bias_pos, bias_neg)


def labels_to_f1_score(true_label, pred_label):
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

    return f1_score(true_label, pred_label)


def labels_to_matthew_corrcoef(true_label, pred_label):
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


def labels_to_generalized_index_balanced_accuracy(true_label,
                                                  pred_label,
                                                  M='gmean',
                                                  alpha=0.1,
                                                  squared=True):
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
    .. [1] Garcia, V. and Mollineda, R.A. and Sanchez, J.S. "Theoretical
    analysis of a performance measure for imbalanced data" ICPR (2010)
    """

    if M == 'sens':
        # Compute the sensitivity
        met, _ = labels_to_sensitivity_specificity(true_label, pred_label)
    elif M == 'spec':
        # Compute the specificity
        _, met = labels_to_sensitivity_specificity(true_label, pred_label)
    elif M == 'prec':
        # Compute the precision
        met, _ = labels_to_precision_negative_predictive_value(true_label,
                                                               pred_label)
    elif M == 'npv':
        # Compute the precision
        _, met = labels_to_precision_negative_predictive_value(true_label,
                                                               pred_label)
    elif M == 'gmean':
        # Compute the negative predictive value
        met = labels_to_geometric_mean(true_label, pred_label)
    elif M == 'acc':
        # Compute the accuracy
        met = labels_to_accuracy(true_label, pred_label)
    elif M == 'cost':
        # Compute the accuracy
        met = labels_to_cost_value(true_label, pred_label)
    elif M == 'f1score':
        # Compute the F1 Score
        met = labels_to_f1_score(true_label, pred_label)
    elif M == 'mcc':
        # Compute the Matthew correlation coefficient
        met = labels_to_matthew_corrcoef(true_label, pred_label)
    else:
        raise ValueError('protoclass.metric.GIBA: The metric that you'
                         ' attend to correct is not implemented.')

    # Check the value of alpha is meaningful
    if not ((alpha >= 0.) and (alpha <= 1.)):
        raise ValueError('protoclass.metric.GIBA: The value of alpha'
                         ' sould be set between 0 and 1.')

    # Check if we should square the metric
    if squared:
        met = met ** 2

    # Compute the dominance
    # We need the sensitivity and specificity
    sens, spec = labels_to_sensitivity_specificity(true_label, pred_label)
    # Compute the dominance as the difference of the sensitiy and specificity
    dom = sens - spec

    # Compute the generalized index of balanced accuracy
    iba = (1 + alpha * dom) * met

    return iba
