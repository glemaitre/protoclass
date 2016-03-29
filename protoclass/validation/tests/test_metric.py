""" Testing the metrics for classification """

import numpy as np

from numpy.testing import assert_equal
from numpy.testing import assert_raises

from protoclass.validation.metric import labels_to_accuracy
from protoclass.validation.metric import labels_to_cost_value
from protoclass.validation.metric import labels_to_f1_score
from protoclass.validation.metric import labels_to_generalized_index_balanced_accuracy
from protoclass.validation.metric import labels_to_geometric_mean
from protoclass.validation.metric import labels_to_matthew_corrcoef
from protoclass.validation.metric import labels_to_precision_negative_predictive_value
from protoclass.validation.metric import labels_to_sensitivity_specificity
from protoclass.validation.metric import cost_with_bias


# Create a fake true and prediction label
true_label = np.array([0] * 10 + [1] * 10)
pred_label = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)


def test_sens_spec():
    """ Test the sensitivity and specificity. """
    # Compute the sensitivity and specificity
    sens, spec = labels_to_sensitivity_specificity(true_label, pred_label)
    assert_equal(sens, 0.5)
    assert_equal(spec, 0.5)


def test_sens_spec_0():
    """ Test the specific case when the sensitivity and specificity can be
    equal to zero. """
    pred_label_corr = np.array([0] * 20)
    sens, spec = labels_to_sensitivity_specificity(true_label,
                                                   pred_label_corr)
    assert_equal(sens, 0.)
    assert_equal(spec, 1.)

    pred_label_corr = np.array([1] * 20)
    sens, spec = labels_to_sensitivity_specificity(true_label,
                                                   pred_label_corr)
    assert_equal(sens, 1.)
    assert_equal(spec, 0.)


def test_npv_precision():
    """ Test precision and NPV. """
    # Compute the precision and negative predictive value
    prec, npv = labels_to_precision_negative_predictive_value(true_label,
                                                              pred_label)
    assert_equal(prec, 0.5)
    assert_equal(npv, 0.5)


def test_npv_precision_0():
    """ Test the specific case when the precision and npv can be
    equal to zero. """
    pred_label_corr = np.array([0] * 20)
    prec, npv = labels_to_precision_negative_predictive_value(true_label,
                                                              pred_label_corr)

    assert_equal(prec, 0.)
    assert_equal(npv, .5)

    pred_label_corr = np.array([1] * 20)
    prec, npv = labels_to_precision_negative_predictive_value(true_label,
                                                              pred_label_corr)
    assert_equal(prec, .5)
    assert_equal(npv, 0.)


def test_geometric_mean():
    """ Test the geometric mean. """
    # Compute the geometric mean
    gmean = labels_to_geometric_mean(true_label, pred_label)
    assert_equal(gmean, 0.5)


def test_accuracy():
    """ Test the accuracy. """
    # Compute the accuracy
    acc = labels_to_accuracy(true_label, pred_label)
    assert_equal(acc, 0.5)


def test_f1_score():
    """ Test the f1 score. """
    # Compute the F1 score
    f1score = labels_to_f1_score(true_label, pred_label)
    assert_equal(f1score, 0.5)


def test_mcc():
    """ Test the Matthew's score. """
    # Compute the Matthew correlation coefficient
    mcc = labels_to_matthew_corrcoef(true_label, pred_label)
    assert_equal(mcc, 0.)


def test_iba():
    """ Test the generalized index balanced. """
    # If the other metric can be computed fine, nothing can go wrong
    # apart of the implementation of this function

    # Create an array of the method to try
    metrics = ['sens', 'spec', 'prec', 'npv', 'gmean',
               'acc', 'cost', 'f1score', 'mcc']

    res_metrics = [.5, .5, .5, .5, .5, .5, .5, .5, .0]

    for res_curr, m_curr in zip(res_metrics, metrics):
        iba = labels_to_generalized_index_balanced_accuracy(true_label,
                                                            pred_label,
                                                            M=m_curr,
                                                            squared=False)
        assert_equal(iba, res_curr)
        iba = labels_to_generalized_index_balanced_accuracy(true_label,
                                                            pred_label,
                                                            M=m_curr,
                                                            squared=True)
        assert_equal(iba, res_curr ** 2)


def test_iba_no_metric():
    """ Test either if an error is raised when an unknown metric is given. """

    assert_raises(ValueError, labels_to_generalized_index_balanced_accuracy,
                  true_label, pred_label, M='unknown', squared=True)


def test_iba_wrong_alpha():
    """ Test either if an error is raised when alpha is not in the range
    it should be. """

    assert_raises(ValueError, labels_to_generalized_index_balanced_accuracy,
                  true_label, pred_label, alpha=-1.)

    assert_raises(ValueError, labels_to_generalized_index_balanced_accuracy,
                  true_label, pred_label, alpha=1.5)


def test_cost():
    """ Check the cost calculation. """
    # Check the cost with sensitivity and specificity with a class bias
    cval1 = labels_to_cost_value(true_label, pred_label,
                                 bias_pos=1.5, bias_neg=1.)
    assert_equal(cval1, 0.5)


def test_cost_metric():
    """ Test the cost metric with bias. """
    # Check the cost function itself with sensitivity and specificity
    # with a class bias
    sens, spec = labels_to_sensitivity_specificity(true_label, pred_label)
    cval2 = cost_with_bias(sens, spec, bias_pos=1.5, bias_neg=1.)
    assert_equal(cval2, 0.5)
