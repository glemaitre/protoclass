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

from random import sample
from collections import Counter

def Classify(training_data, training_label, testing_data, testing_label, classifier_str='random-forest', balancing_criterion=None, **kwargs):

    if ('class_weight' in kwargs) and (balancing_criterion == 'class_prior'):
        print 'The keyword class_weight has been overwritten due to the keyword balancing_criterion'
        del kwargs['class_weight']
        class_weight = 'auto'
    else:
        class_weight = kwargs.pop('class_weight', None)

    # Apply the balancing
    if balancing_criterion == 'random-sampling':
        training_data, training_label = BalancingTraining(training_data, training_label, balancing_criterion=balancing_criterion)
    elif balancing_criterion == 'class-prior':
        class_weight = 'auto'
    elif balancing_criterion == 'random-samples-boosting':
        n_bootstrap_balancing = kwargs.pop('n_bootstrap_balancing', 100)
        boot_training_data = []
        boot_training_label = []
        for b in range(n_bootstrap_balancing):
            # Generate a training set which should be balanced
            tmp_training_data, tmp_training_label = BalancingTraining(training_data, training_label, balancing_criterion='random-sampling')
            boot_training_data.append(tmp_training_data)
            boot_training_label.append(tmp_training_label)

    # Check which classifier to select to classify the data
    if (classifier_str == 'random-forest') and (balancing_criterion == 'random-samples-boosting'):
        # In this cases, we will have several training set to use
        boot_pred_prob = []
        boot_pred_label = []
        for d, l in zip(boot_training_data, boot_training_label):
            # Classify using random forest
            tmp_pred_prob, tmp_pred_label = ClassifyRandomForest(d, l, testing_data, class_weight=class_weight, **kwargs)
            boot_pred_prob.append(tmp_pred_prob)
            boot_pred_label.append(tmp_pred_label)

            # TO CHECK WHAT IS BETTER HERE FOR THE PROBABILITY
            # Make the average of the probability
            pred_prob = np.mean(np.array(boot_pred_prob), axis=0)
            # Make the majority voting
            pred_label = np.sum(np.array(boot_pred_label), axis=0)
            pred_label[pred_label >= 0] = 1
            pred_label[pred_label < 0] = -1

    elif classifier_str == 'random-forest':
        # Classify using random forest
        pred_prob, pred_label = ClassifyRandomForest(training_data, training_label, testing_data, class_weight=class_weight, **kwargs)

    # Test the reliability of the classifier
    ### Compute the ROC curve
    roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])
    fpr, tpr, thresh = roc_curve(testing_label, pred_prob[:, 1])
    auc = roc_auc_score(testing_label, pred_prob[:, 1])
    roc = roc_auc(fpr, tpr, thresh, auc)

    return (pred_label, roc)

def BalancingTraining(training_data, training_label, **kwargs):

    strategy = kwargs.pop('balancing_criterion', 'random-sampling')

    if strategy == 'random-sampling':
        return BalancingRandomSampling(training_data, training_label)
            
def BalancingRandomSampling(training_data, training_label):

    # Count the occurence in the training_label
    count = Counter(training_label)
    
    # Find the least represented class
    sel_idx = []
    for keys, values in count.iteritems():
        
        if keys == min(count, key=count.get):
            # Append all the indexes
            sel_idx = sel_idx + np.nonzero(training_label == keys)[0].tolist()
        else:
            # Append a subset of indexes
            sel_idx = sel_idx + sample(np.nonzero(training_label == keys)[0], count[min(count, key=count.get)])

    return (training_data[sel_idx, :], training_label[sel_idx])

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
        param_warm_start = kwargs.pop('warm_start', False)
        param_class_weight = kwargs.pop('class_weight', None)
        n_log_space = kwargs.pop('n_log_space', 10)
        
        # If the number of estimators is not specified, it will be find by cross-validation
        if 'n_estimators' in kwargs:
            param_n_estimators = kwargs.pop('n_estimators', 10)

            # Construct the Random Forest classifier
            crf = RandomForestClassifier(n_estimators=param_n_estimators, criterion=param_criterion, max_depth=param_max_depth, min_samples_split=param_min_samples_split, min_samples_leaf=param_min_samples_leaf, max_features=param_max_features, bootstrap=param_bootstrap, oob_score=param_oob_score, n_jobs=param_n_jobs, random_state=param_random_state, verbose=param_verbose, warm_start=param_warm_start, class_weight=param_class_weight)

        else:
            # Import the function to perform the grid search
            from sklearn import grid_search
            
            # Create the parametor grid
            param_n_estimators = {'n_estimators':np.array(np.round(np.logspace(1., 2.7, n_log_space))).astype(int)}

            # Create the random forest without the parameters to find during the grid search
            crf_gs = RandomForestClassifier(criterion=param_criterion, max_depth=param_max_depth, min_samples_split=param_min_samples_split, min_samples_leaf=param_min_samples_leaf, max_features=param_max_features, bootstrap=param_bootstrap, oob_score=param_oob_score, n_jobs=param_n_jobs, random_state=param_random_state, verbose=param_verbose, warm_start=param_warm_start, class_weight=param_class_weight)

            # Create the different random forest to try specifying the grid search parametors
            crf = grid_search.GridSearchCV(crf_gs, param_n_estimators)

        # Train the classifier
        crf.fit(training_data, training_label)

        # Test the classifier
        pred_prob = crf.predict_proba(testing_data)
        pred_label = crf.predict(testing_data)

        return (pred_prob, pred_label)
