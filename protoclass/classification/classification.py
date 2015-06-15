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

# Add the unbalanced toolbox the python path
import sys
sys.path.append(r'./protoclass/classification/UnbalancedDataset/')

# Import over-sampling method
from unbalanced_dataset.over_sampling import OverSampler
from unbalanced_dataset.over_sampling import SMOTE

# Import under-sampling method
from unbalanced_dataset.under_sampling import UnderSampler
from unbalanced_dataset.under_sampling import TomekLinks
from unbalanced_dataset.under_sampling import ClusterCentroids
from unbalanced_dataset.under_sampling import NearMiss
from unbalanced_dataset.under_sampling import CondensedNearestNeighbour
from unbalanced_dataset.under_sampling import OneSidedSelection
from unbalanced_dataset.under_sampling import NeighbourhoodCleaningRule

# Import ensemble sampling method
from unbalanced_dataset.ensemble_sampling import EasyEnsemble
from unbalanced_dataset.ensemble_sampling import BalanceCascade

# Import known pipeline method
from unbalanced_dataset.pipeline import SMOTEENN
from unbalanced_dataset.pipeline import SMOTETomek 

from random import sample
from collections import Counter

roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])

def Classify(training_data, training_label, testing_data, testing_label, classifier_str='random-forest', balancing_criterion=None, **kwargs):

    #########################################################################
    ### GET THE PARAMETERS
    #########################################################################
    if ('class_weight' in kwargs) and (balancing_criterion == 'class_prior'):
        print 'The keyword class_weight has been overwritten due to the keyword balancing_criterion'
        del kwargs['class_weight']
        class_weight = 'auto'
    else:
        class_weight = kwargs.pop('class_weight', None)


    #########################################################################
    ### BALANCE THE DATA IF NEEDED
    #########################################################################

    # Define the ration to use in case that we want to balance the data
    count_label = Counter(training_label)
    count_min_class = float(count_label[min(count_label, key=count_label.get)])
    count_max_class = float(count_label[max(count_label, key=count_label.get)])
    ratio_oversampling = (count_max_class - count_min_class) / count_min_class
    
    # Over-sampling
    if balancing_criterion == 'random-over-sampling':
        os = OverSampler(ratio=ratio_oversampling)
        training_data, training_label = os.fit_transform(training_data, training_label)
    elif balancing_criterion == 'smote':
        k_smote = kwargs.pop('k_smote', 5)
        m_smote = kwargs.pop('m_smote', 10)
        out_step_smote = kwargs.pop('out_step_smote', 0.5)
        kind_smote = kwargs.pop('kind_smote', 'regular')
        sm = SMOTE(ratio=ratio_oversampling, k=k_smote, m=m_smote, 
                   out_step=out_step_smote, kind=kind_smote)
        training_data, training_label = sm.fit_transform(training_data, training_label)
    
    # Under-sampling
    elif balancing_criterion == 'random-under-sampling':
        replacement = kwargs.pop('replacement', True)
        us = UnderSampler(replacement=replacement)
        training_data, training_label = us.fit_transform(training_data, training_label)
    elif balancing_criterion == 'tomek-links':
        tl = TomekLinks()
        training_data, training_label = tl.fit_transform(training_data, training_label)
    elif balancing_criterion == 'clustering':
        cc = ClusterCentroids()
        training_data, training_label = cc.fit_transform(training_data, training_label)
    elif balancing_criterion == 'nearmiss':
        version_nearmiss = kwargs.pop('version_nearmiss', 1)
        size_ngh = kwargs.pop('size_ngh', 3)
        ver3_samp_ngh = kwargs.pop('ver3_samp_ngh', 3)
        # Add some option to extract NN kwargs
        nm = NearMiss(version=version_nearmiss, size_ngh=size_ngh, 
                      ver3_samp_ngh=ver3_samp_ngh)
        training_data, training_label = nm.fit_transform(training_data, training_label)
    elif balancing_criterion == 'cnn':
        size_ngh = kwargs.pop('size_ngh', 3)
        n_seeds_S = kwargs.pop('n_seeds_S', 1)
        # Add some option to extract NN kwargs
        cnn = CondensedNearestNeighbour(size_ngh=size_ngh, n_seeds_S=n_seeds_S)
        training_data, training_label = cnn.fit_transform(training_data, training_label)
    elif balancing_criterion == 'one-sided-selection':
        size_ngh = kwargs.pop('size_ngh', 1)
        n_seeds_S = kwargs.pop('n_seeds_S', 1)
        # Add some option to extract NN kwargs
        oss = OneSidedSelection(size_ngh=size_ngh, n_seeds_S=n_seeds_S)
        training_data, training_label = oss.fit_transform(training_data, training_label)
    elif balancing_criterion == 'ncr':
        size_ngh = kwargs.pop('size_ngh', 3)
        # Add some option to extract NN kwargs
        ncr = NeighbourhoodCleaningRule(size_ngh=size_ngh)
        training_data, training_label = ncr.fit_transform(training_data, training_label)
        
    # Ensemble-sampling
    elif balancing_criterion == 'easy-ensemble':
        n_subsets = kwargs.pop('n_subsets', 10)
        ee = EasyEnsemble(n_subsets=n_subsets)
        boot_training_data, boot_training_label = ee.fit_transform(training_data, training_label)
    elif balancing_criterion == 'balance-cascade':
        balancing_classifier = kwargs.pop('balancing_classifier', 'knn')
        n_max_subset = kwargs.pop('n_max_subset', None)
        bootstrap = kwargs.pop('bootstrap', True)
        bc = BalanceCascade(classifier=balancing_classifier,
                            n_max_subset=n_max_subset, bootstrap=bootstrap)
        boot_training_data, boot_training_label = bc.fit_transform(training_data, training_label)
    
    # Pipeline-sampling
    elif balancing_criterion == 'smote-enn':
        k_smote = kwargs.pop('k_smote', 5)
        size_ngh = kwargs.pop('size_ngh', 3)
        sme = SMOTEENN(ratio=ratio_oversampling, k=k_smote, size_ngh=size_ngh)
        training_data, training_label = sme.fit_transform(training_data, training_label)
    elif balancing_criterion == 'smote-tomek':
        k_smote = kwargs.pop('k_smote', 5)
        smt = SMOTEENN(ratio=ratio_oversampling, k=k_smote)
        training_data, training_label = smt.fit_transform(training_data, training_label)

        
    #########################################################################
    ### APPLY CLASSIFICATION
    #########################################################################
    if (classifier_str == 'random-forest') and ((balancing_criterion == 'easy-ensemble') or 
                                                (balancing_criterion == 'balance-cascade')):
        # In this cases, we will have several training set to use
        boot_pred_prob = []
        boot_pred_label = []
        for d, l in zip(boot_training_data, boot_training_label):
            # Classify using random forest
            crf = TrainRandomForest(d, l, class_weight=class_weight, **kwargs)
            tmp_pred_prob, tmp_pred_label = TestRandomForest(crf, testing_data)
            boot_pred_prob.append(tmp_pred_prob)
            boot_pred_label.append(tmp_pred_label)

            # TO CHECK WHAT IS BETTER HERE - MAJORITY VOTING FOR THE MOMENT
            # Make the average of the probability
            pred_prob = np.mean(np.array(boot_pred_prob), axis=0)
            # Make the majority voting
            pred_label = np.sum(np.array(boot_pred_label), axis=0)
            pred_label[pred_label >= 0] = 1
            pred_label[pred_label < 0] = -1

    elif classifier_str == 'random-forest':
        # Classify using random forest
        crf = TrainRandomForest(training_data, training_label, class_weight=class_weight, **kwargs)
        pred_prob, pred_label = TestRandomForest(crf, testing_data)


    #########################################################################
    ### CHECK THE PERFORMANCE OF THE CLASSIFIER
    #########################################################################
    ### Compute the ROC curve
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

def TrainRandomForest(training_data, training_label, **kwargs):
    
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

        return crf

def TestRandomForest(crf, testing_data):
    
        # Import Random Forest from scikit learn
        from sklearn.ensemble import RandomForestClassifier

        # Test the classifier
        pred_prob = crf.predict_proba(testing_data)
        pred_label = crf.predict(testing_data)

        return (pred_prob, pred_label)
