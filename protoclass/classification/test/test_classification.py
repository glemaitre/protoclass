#title           :test_classiciation.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/12
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
### Module to performed parallel processing
import multiprocessing
# OS library
import os
from os.path import join, isdir, isfile
# sys library
import sys

from protoclass.classification.classification import Classify

# Generate a vector 
label = np.array([-1]*1000 + [1]*200)
data = np.random.random((label.shape[0], 5))

label2 = np.array([-1]*100 + [1]*20)
data2 = np.random.random((label2.shape[0], 5))

balancing_criterion = 'balance-cascade'
kind_smote = 'svm'
version_nearmiss = 3
classifier_str = 'naive-bayes'
pred, roc = Classify(data, label, data2, label2, classifier_str=classifier_str, class_prior_override=(.1, .9))# ,
                     # balancing_criterion=balancing_criterion)
print pred
print roc
