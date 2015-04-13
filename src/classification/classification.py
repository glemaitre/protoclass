#title           :classification.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/04/13
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the file needed
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def classifiy(training_data, training_label, testing_data, testing_label, classifier_str='random-forest', **kargs):
    
    # Check which classifier to select to classify the data
    if classifier_str == 'random-forest':
        # Classify using random forest
        pred_prob, pred_label = classify_random_forest(training_data, training_label, testing_data, **kargs)

    # Test the reliability of the classifier
    

def classify_random_forest(training_data, training_label, testing_data, classifier_str='random-forest', **kargs):
    
        # Import Random Forest from scikit learn
        from sklearn.ensemble import RandomForestClassifier

        # Unpack the keywords to create the classifier
        n_estimators = kwargs.pop('n_estimators', 10)
        criterion = kwargs.pop('criterion', 'gini')
        max_depth = kwargs.pop('max_depth', None)
        min_samples_split = kwargs.pop('min_samples_split', 2)
        min_samples_leaf = kwargs.pop('min_samples_leaf', 1)
        min_weight_fraction_leaf = kwargs.pop('min_weight_fraction_leaf', 0.0)
        max_features = kwargs.pop('max_features', 'auto')
        max_leaf_nodes = kwargs.pop('max_leaf_nodes', None)
        bootstrap = kwargs.pop('bootstrap', True)
        oob_score = kwargs.pop('oob_score', False)
        n_jobs = kwargs.pop('n_jobs', 1)
        random_state = kwargs.pop('random_state', None)
        verbose = kwargs.pop('verbose', 0)
        warm_start = kwargs.pop('warm_start', False)
        class_weight = kwargs.pop('class_weight', None)
        
        # Construct the Random Forest classifier
        crf = RandomForestClassifier(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, class_weight)

        # Train the classifier
        sample_weight = kwargs.pop('sample_weight', None)
        crf.fit(training_data, training_label, sample_weight)

        # Test the classifier
        pred_prob = crf.predict_proba(testing_data)
        pred_label = crf.predict(testing_data)

        return (pred_prob, pred_label)
