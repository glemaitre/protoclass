#title           :classiciation_imbalanced_study.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/31
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

# Scikit-learn library
from sklearn.datasets import make_classification
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler

from protoclass.classification.classification import Classify

# # Get the path to file
# filename_data = sys.argv[1]
# print 'Opening the following file: {}'.format(filename_data)

# # Read the data
# data, label = OpenDataLabel(filename_data)
data, label = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=10)

# normalise the data
scaler_min_max = MinMaxScaler(feature_range=(-1., 1.), copy=False)
scaler_min_max.fit_transform(data)

# Create 10-fold for the cross validation
n_folds = 10
n_samples = data.shape[0]
kf = KFold(n=n_samples, n_folds=n_folds, shuffle=True)

config = [{'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'random-over-sampling'},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'smote', 'kind_smote' : 'regular'},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'smote', 'kind_smote' : 'borderline1'},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'smote', 'kind_smote' : 'borderline2'},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'smote', 'kind_smote' : 'svm'},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'random-under-sampling', 'replacement' : True},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'tomek_links'},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'clustering'},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'nearmiss', 'version_nearmiss' : 1, 'size_ngh': 3},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'nearmiss', 'version_nearmiss' : 2, 'size_ngh': 3},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'nearmiss', 'version_nearmiss' : 3, 'size_ngh': 3, 'ver3_samp_ngh' : 3},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'cnn', 'size_ngh' : 3, 'n_seeds_S' :1},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'one-sided-selection', 'size_ngh' : 1, 'n_seeds_S' :1},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'ncr', 'size_ngh' : 3},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'easy-ensemble', 'n_subsets' :  10},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'balance-cascade', 'n_max_subset' : 100, 'balancing_classifier' : 'knn', 'bootstrap' : True},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'smote-enn', 'size_ngh' : 3},
          {'classifier_str' : 'random-forest', 'n_estimators' : 10, 'balancing_criterion' : 'smote-tomek'}]

rocs = []
pred_labels = []
# Apply the classification for each fold
k = 1
for training_idx, testing_idx in kf:
    
    # CHECK THAT THE NUMBER IN THE TESTING CAN CHANGE OR NOT 
    # IT COULD BE PROBLEMATIC WHEN CONVERTING TO NUMPY ARRAY

    print 'Iteration #{}'.format(k)
    # Extract the data
    ### Training
    training_data = data[training_idx, :]
    training_label = label[training_idx]
    ### Testing 
    testing_data = data[testing_idx, :]
    testing_label = label[testing_idx]

    config_roc = []
    config_pred_label = []
    for c in config:
        pred_label, roc = Classify(training_data, training_label, testing_data, testing_label, **c)
        config_roc.append(roc)
        config_pred_label.append(pred_label)

    rocs.append(config_roc)
    pred_labels.append(config_pred_label)

    k += 1

# Convert the data to store to numpy data
rocs = np.array(rocs)
pred_labels = np.array(pred_labels)

# Reshape the array to have the first index corresponding to the
# configuration, the second index to the iteration of the k-fold
# and the last index to the data themselve.
rocs = np.swapaxes(rocs, 0, 1)
pred_labels = np.swapaxes(pred_labels, 0, 1)

# Save the results somewhere
path_to_save = sys.argv[3]

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

saving_filename = 'result_' + str(filename_data)
saving_path = join(path_to_save, saving_filename)
np.savez(saving_path, pred_labels=pred_labels, rocs=rocs)
