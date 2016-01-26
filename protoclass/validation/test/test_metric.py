#title           :test_metric.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/18
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

import numpy as np
from numpy import testing
from numpy.testing import assert_equal

from protoclass.validation.metric import LabelsToSensitivitySpecificity, LabelsToPrecisionNegativePredictiveValue, LabelsToGeometricMean, LabelsToAccuracy, LabelsToF1score, LabelsToMatthewCorrCoef, LabelsToGeneralizedIndexBalancedAccuracy, LabelsToCostValue, CostWithBias


def test_metrics():

    # Create a fake true and prediction label
    true_label = np.array([0] * 10 + [1] * 10)
    pred_label = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)

    # Compute the sensitivity and specificity
    sens, spec = LabelsToSensitivitySpecificity(true_label, pred_label)
    assert_equal(sens, 0.5)
    assert_equal(spec, 0.5)

    # Compute the precision and negative predictive value
    prec, npv = LabelsToPrecisionNegativePredictiveValue(true_label, pred_label)
    assert_equal(prec, 0.5)
    assert_equal(npv, 0.5)

    # Compute the geometric mean
    gmean = LabelsToGeometricMean(true_label, pred_label)
    assert_equal(gmean, 0.5)

    # Compute the accuracy
    acc = LabelsToAccuracy(true_label, pred_label)
    assert_equal(acc, 0.5)

    # Compute the F1 score
    f1score = LabelsToF1score(true_label, pred_label)
    assert_equal(f1score, 0.5)

    # Compute the Matthew correlation coefficient
    mcc = LabelsToMatthewCorrCoef(true_label, pred_label)
    assert_equal(mcc, 0.)

    # If the other metric can be computed fine, nothing can go wrong apart of the implementation of this function
    iba = LabelsToGeneralizedIndexBalancedAccuracy(true_label, pred_label)
    assert_equal(iba, 0.25)

    # Check the cost with sensitivity and specificity with a class bias
    cval1 = LabelsToCostValue(true_label, pred_label, bias_pos=1.5, bias_neg=1.)
    assert_equal(cval1, 0.5)

    # Check the cost function itself with sensitivity and specificity with a class bias
    cval2 = CostWithBias(sens, spec, bias_pos=1.5, bias_neg=1.)
    assert_equal(cval2, cval2)
    

if __name__ == '__main__':
    testing.run_module_suite()
