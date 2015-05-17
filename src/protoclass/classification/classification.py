#title           :classification.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/13
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
from collections import namedtuple

import multiprocessing

import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def Classify(training_data, training_label, testing_data, testing_label, classifier_str='random-forest', **kwargs):
    
    # Check which classifier to select to classify the data
    if classifier_str == 'random-forest':
        # Classify using random forest
        pred_prob, pred_label = ClassifyRandomForest(training_data, training_label, testing_data, **kwargs)

    # Test the reliability of the classifier
    ### Compute the ROC curve
    roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])
    fpr, tpr, thresh = roc_curve(testing_label, pred_prob[:, 1])
    auc = roc_auc_score(testing_label, pred_prob[:, 1])
    roc = roc_auc(fpr, tpr, thresh, auc)

    return (pred_label, roc)

def ClassifyRandomForest(training_data, training_label, testing_data, **kwargs):
    
        # Import Random Forest from scikit learn
        from sklearn.ensemble import RandomForestClassifier

        # Unpack the keywords to create the classifier
        param_criterion = kwargs.pop('criterion', 'gini')
        param_max_depth = kwargs.pop('max_depth', None)
        param_min_samples_split = kwargs.pop('min_samples_split', 2)
        param_min_samples_leaf = kwargs.pop('min_samples_leaf', 1)
        param_min_weight_fraction_leaf = kwargs.pop('min_weight_fraction_leaf', 0.0)
        param_max_features = kwargs.pop('max_features', 'auto')
        param_max_leaf_nodes = kwargs.pop('max_leaf_nodes', None)
        param_bootstrap = kwargs.pop('bootstrap', True)
        param_oob_score = kwargs.pop('oob_score', False)
        param_n_jobs = kwargs.pop('n_jobs', multiprocessing.cpu_count())
        param_random_state = kwargs.pop('random_state', None)
        param_verbose = kwargs.pop('verbose', 1)
        param_min_density = kwargs.pop('min_density', None)
        param_compute_importances = kwargs.pop('compute_importances', None)

        # If the number of estimators is not specified, it will be find by cross-validation
        if 'n_estimators' in kwargs:
            param_n_estimators = kwargs.pop('n_estimators', 10)
        else:
            param_n_estimators = {'n_estimators':np.array(np.round(np.logspace(1., 2.7, n_log_space))).astype(int)}

            crf = RandomForestClassifier(criterion=param_criterion, max_depth=param_max_depth, min_samples_split=param_min_samples_split, min_samples_leaf=param_min_samples_leaf, max_features=param_max_features, bootstrap=param_bootstrap, oob_score=param_oob_score, n_jobs=param_n_jobs, random_state=param_random_state, verbose=param_verbose, min_density=param_min_density, compute_importances=param_compute_importances)
        
        # Construct the Random Forest classifier
        crf = RandomForestClassifier(n_estimators=param_n_estimators, criterion=param_criterion, max_depth=param_max_depth, min_samples_split=param_min_samples_split, min_samples_leaf=param_min_samples_leaf, max_features=param_max_features, bootstrap=param_bootstrap, oob_score=param_oob_score, n_jobs=param_n_jobs, random_state=param_random_state, verbose=param_verbose, min_density=param_min_density, compute_importances=param_compute_importances)

        # Train the classifier
        sample_weight = kwargs.pop('sample_weight', None)
        crf.fit(training_data, training_label, sample_weight)

        # Test the classifier
        pred_prob = crf.predict_proba(testing_data)
        pred_label = crf.predict(testing_data)

        return (pred_prob, pred_label)
