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

# Import over-sampling method
from unbalanced_dataset import OverSampler, SMOTE

# Import under-sampling method
from unbalanced_dataset import UnderSampler, TomekLinks, ClusterCentroids, NearMiss, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule

# Import ensemble sampling method
from unbalanced_dataset import EasyEnsemble, BalanceCascade

# Import known pipeline method
from unbalanced_dataset import SMOTEENN, SMOTETomek 

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

    gs_n_jobs = kwargs.pop('gs_n_jobs', multiprocessing.cpu_count())


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

    # Case of Ensemble or Cascade Under sampling with ....
    
    if ((balancing_criterion == 'easy-ensemble') or 
        (balancing_criterion == 'balance-cascade')):

        # In this cases, we will have several training set to use
        boot_pred_prob = []
        boot_pred_label = []
        for d, l in zip(boot_training_data, boot_training_label):

            # ... Random Forest classifier
            if (classifier_str == 'random-forest'):
                crf = TrainRandomForest(d, l, class_weight=class_weight, gs_n_jobs=gs_n_jobs, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestRandomForest(crf, testing_data)

            # ... Decision Tree classifier
            elif (classifier_str == 'decision-tree'):
                cdt = TrainTree(d, l, class_weight=class_weight, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestTree(cdt, testing_data)

            # ... Logistic Regression classifier
            elif (classifier_str == 'logistic-regression'):
                clr = TrainLogisticRegression(d, l, class_weight=class_weight, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestLogisticRegression(clr, testing_data)

            # ... Perceptron classifier
            elif (classifier_str == 'perceptron'):
                cp = TrainPerceptron(d, l, class_weight=class_weight, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestPerceptron(cp, testing_data)

            # ... LDA classifier
            elif (classifier_str == 'lda'):
                clda = TrainLDA(d, l, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestLDA(clda, testing_data)

            # ... QDA classifier
            elif (classifier_str == 'qda'):
                cqda = TrainQDA(d, l, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestQDA(cqda, testing_data)

            # ... Linear SVM classifier
            elif (classifier_str == 'linear-svm'):
                clsvm = TrainLinearSVM(d, l, class_weight=class_weight, gs_n_jobs=gs_n_jobs, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestLinearSVM(clsvm, testing_data)

            # ... Kernel SVM classifier
            elif (classifier_str == 'kernel-svm'):
                cksvm = TrainKernelSVM(d, l, class_weight=class_weight, gs_n_jobs=gs_n_jobs, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestKernelSVM(cksvm, testing_data)

            # ... KNN classifier
            elif (classifier_str == 'knn'):
                cknn = TrainKNN(d, l, gs_n_jobs=gs_n_jobs, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestKNN(cknn, testing_data)

            # ... AdaBoost classifier
            elif (classifier_str == 'adaboost'):
                cadb = TrainAdaBoost(d, l, gs_n_jobs=gs_n_jobs, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestAdaBoost(cadb, testing_data)

            # ... Gradient Boosting classifier
            elif (classifier_str == 'gradient-boosting'):
                cgb = TrainGradientBoosting(d, l, gs_n_jobs=gs_n_jobs, **kwargs)
                tmp_pred_prob, tmp_pred_label = TestGradientBoosting(cgb, testing_data)

            else:
                raise ValueError('protoclass.classification: Classifier unknown')

            # Append the different results
            boot_pred_prob.append(tmp_pred_prob)
            boot_pred_label.append(tmp_pred_label)

        # TO CHECK WHAT IS BETTER HERE - MAJORITY VOTING FOR THE MOMENT
        # Make the average of the probability
        pred_prob = np.mean(np.array(boot_pred_prob), axis=0)
        # Make the majority voting
        pred_label = np.sum(np.array(boot_pred_label), axis=0)
        pred_label[pred_label >= 0] = 1
        pred_label[pred_label < 0] = -1

    # Case of any other resampling or not any sampling with ...
    # ... Random Forest classifier
    elif (classifier_str == 'random-forest'):
        # Train and classify
        crf = TrainRandomForest(training_data, training_label, class_weight=class_weight, gs_n_jobs=gs_n_jobs, **kwargs)
        pred_prob, pred_label = TestRandomForest(crf, testing_data)

    # ... Decision Tree classifier
    elif (classifier_str == 'decision-tree'):
        # Train and classify
        cdt = TrainTree(training_data, training_label, class_weight=class_weight, **kwargs)
        pred_prob, pred_label = TestTree(cdt, testing_data)

    # ... Logistic Regression classifier
    elif (classifier_str == 'logistic-regression'):
        # Train and classify
        clr = TrainLogisticRegression(training_data, training_label, class_weight=class_weight, **kwargs)
        pred_prob, pred_label = TestLogisticRegression(clr, testing_data)

    # ... Perceptron classifier
    elif (classifier_str == 'perceptron'):
        # Train and classify
        cp = TrainPerceptron(training_data, training_label, class_weight=class_weight, **kwargs)
        pred_prob, pred_label = TestPerceptron(cp, testing_data)

    # ... LDA classifier
    elif (classifier_str == 'lda'):
        # Train and classify
        clda = TrainLDA(training_data, training_label, **kwargs)
        pred_prob, pred_label = TestLDA(clda, testing_data)

    # ... QDA classifier
    elif (classifier_str == 'qda'):
        # Train and classify
        cqda = TrainQDA(training_data, training_label, **kwargs)
        pred_prob, pred_label = TestQDA(cqda, testing_data)

    # ... Linear SVM classifier
    elif (classifier_str == 'linear-svm'):
        # Train and classify
        clsvm = TrainLinearSVM(training_data, training_label, class_weight=class_weight, gs_n_jobs=gs_n_jobs, **kwargs)
        pred_prob, pred_label = TestLinearSVM(clsvm, testing_data)

    # ... Kernel SVM classifier
    elif (classifier_str == 'kernel-svm'):
        # Train and classify
        cksvm = TrainKernelSVM(training_data, training_label, class_weight=class_weight, gs_n_jobs=gs_n_jobs, **kwargs)
        pred_prob, pred_label = TestKernelSVM(cksvm, testing_data)

    # ... KNN classifier
    elif (classifier_str == 'knn'):
        # Train and classify
        cknn = TrainKNN(training_data, training_label, gs_n_jobs=gs_n_jobs, **kwargs)
        pred_prob, pred_label = TestKNN(cknn, testing_data)

    # ... AdaBoost classifier
    elif (classifier_str == 'adaboost'):
        # Train and classify
        cadb = TrainAdaBoost(training_data, training_label, gs_n_jobs=gs_n_jobs, **kwargs)
        pred_prob, pred_label = TestAdaBoost(cadb, testing_data)

    # ... Gradient Boosting classifier
    elif (classifier_str == 'gradient-boosting'):
        # Train and classify
        cgb = TrainGradientBoosting(training_data, training_label, gs_n_jobs=gs_n_jobs, **kwargs)
        pred_prob, pred_label = TestGradientBoosting(cgb, testing_data)

    else:
        raise ValueError('protoclass.classification: Unknown classifier')


    #########################################################################
    ### CHECK THE PERFORMANCE OF THE CLASSIFIER
    #########################################################################
    ### Compute the ROC curve
    # Case to compute using the probability return by the classification
    if ((classifier_str == 'random-forest')       or
        (classifier_str == 'decision-tree')       or
        (classifier_str == 'logistic-regression') or   
        (classifier_str == 'lda')                 or
        (classifier_str == 'qda')                 or
        (classifier_str == 'adaboost')            or
        (classifier_str == 'gradient-boosting')   or
        (classifier_str == 'knn')                    ):
        fpr, tpr, thresh = roc_curve(testing_label, pred_prob[:, 1])
        auc = roc_auc_score(testing_label, pred_prob[:, 1])
    # Case to compute using the decision function return by the classification
    elif ((classifier_str == 'perceptron') or
          (classifier_str == 'linear-svm') or
          (classifier_str == 'kernel-svm')    ):
        fpr, tpr, thresh = roc_curve(testing_label, pred_prob)
        auc = roc_auc_score(testing_label, pred_prob)

    # Put the element inside a structure
    roc = roc_auc(fpr, tpr, thresh, auc)

    return (pred_label, roc)

############################## RANDOM FOREST CLASSIFICATION ##############################

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
    param_n_jobs = kwargs.pop('n_jobs', -1)
    param_random_state = kwargs.pop('random_state', None)
    param_verbose = kwargs.pop('verbose', 1)
    param_warm_start = kwargs.pop('warm_start', False)
    param_class_weight = kwargs.pop('class_weight', None)
    n_log_space = kwargs.pop('n_log_space', 10)
    gs_n_jobs = kwargs.pop('gs_n_jobs', multiprocessing.cpu_count())

    # If the number of estimators is not specified, it will be find by cross-validation
    if 'n_estimators' in kwargs:
        param_n_estimators = kwargs.pop('n_estimators', 10)

        # Construct the Random Forest classifier
        crf = RandomForestClassifier(n_estimators=param_n_estimators, criterion=param_criterion, 
                                     max_depth=param_max_depth, min_samples_split=param_min_samples_split, 
                                     min_samples_leaf=param_min_samples_leaf, max_features=param_max_features, 
                                     bootstrap=param_bootstrap, oob_score=param_oob_score, n_jobs=param_n_jobs, 
                                     random_state=param_random_state, verbose=param_verbose, 
                                     warm_start=param_warm_start, class_weight=param_class_weight)

    else:
        # Import the function to perform the grid search
        from sklearn import grid_search

        # Create the parametor grid
        param_n_estimators = {'n_estimators':np.array(np.round(np.logspace(1., 2.7, n_log_space))).astype(int)}

        # Create the random forest without the parameters to find during the grid search
        crf_gs = RandomForestClassifier(criterion=param_criterion, max_depth=param_max_depth, 
                                        min_samples_split=param_min_samples_split, 
                                        min_samples_leaf=param_min_samples_leaf, max_features=param_max_features, 
                                        bootstrap=param_bootstrap, oob_score=param_oob_score, n_jobs=param_n_jobs, 
                                        random_state=param_random_state, verbose=param_verbose, 
                                        warm_start=param_warm_start, class_weight=param_class_weight)

        # Create the different random forest to try specifying the grid search parametors
        crf = grid_search.GridSearchCV(crf_gs, param_n_estimators, n_jobs=gs_n_jobs)

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

############################## DECISION TREE CLASSIFICATION ##############################

def TrainTree(training_data, training_label, **kwargs):
    
    # Import Random Forest from scikit learn
    from sklearn.tree import DecisionTreeClassifier

    # Unpack the keywords to create the classifier
    criterion = kwargs.pop('criterion', 'gini')
    splitter = kwargs.pop('splitter', 'best')
    max_depth = kwargs.pop('max_depth', None)
    min_samples_split = kwargs.pop('min_samples_split', 2)
    min_samples_leaf = kwargs.pop('min_samples_leaf', 1)
    min_weight_fraction_leaf = kwargs.pop('min_weight_fraction_leaf', 0.0)
    max_features = kwargs.pop('max_features', 'auto')
    random_state = kwargs.pop('random_state', None)
    max_leaf_nodes = kwargs.pop('max_leaf_nodes', None)
    class_weight = kwargs.pop('class_weight', None)

    # Call the constructor
    cdt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                 random_state=random_state, max_leaf_nodes=max_leaf_nodes, class_weight=class_weight)

    # Train the classifier
    cdt.fit(training_data, training_label)

    return cdt

def TestTree(cdt, testing_data):
    
    # Import decision tree from scikit learn
    from sklearn.tree import DecisionTreeClassifier

    # Test the classifier
    pred_prob = cdt.predict_proba(testing_data)
    pred_label = cdt.predict(testing_data)

    return (pred_prob, pred_label)

############################## LOGISTIC REGRESSION CLASSIFIER ##############################

def TrainLogisticRegression(training_data, training_label, **kwargs):
    
    # Import Logistic Regression from scikit learn
    from sklearn.linear_model import LogisticRegression

    # Unpack the keywords to create the classifier
    penalty = kwargs.pop('penalty', 'l2')
    dual = kwargs.pop('dual', False)
    tol = kwargs.pop('tol', 0.0001)
    C = kwargs.pop('C', 1.0)
    fit_intercept = kwargs.pop('fit_intercept', True)
    intercept_scaling = kwargs.pop('intercept_scaling', 1)
    class_weight = kwargs.pop('class_weight', None)
    random_state = kwargs.pop('random_state', None)
    solver = kwargs.pop('solver', 'liblinear')
    max_iter = kwargs.pop('max_iter', 100)
    multi_class = kwargs.pop('multi_class', 'ovr')
    verbose = kwargs.pop('verbose', 0)

    # Call the constructor with the proper input arguments
    clr = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, 
                             intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state,
                             solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose)

    # Train the classifier
    clr.fit(training_data, training_label)

    return clr

def TestLogisticRegression(clr, testing_data):
    
    # Import Logistic Regression from scikit learn
    from sklearn.linear_model import LogisticRegression

    # Test the classifier
    pred_prob = clr.predict_proba(testing_data)
    pred_label = clr.predict(testing_data)

    return (pred_prob, pred_label)

############################## PERCEPTRON CLASSIFIER ##############################

def TrainPerceptron(training_data, training_label, **kwargs):
    
    # Import Perceptron from scikit learn
    from sklearn.linear_model import Perceptron

    # Unpack the keywords to create the classifier
    penalty = kwargs.pop('penalty', None)
    alpha = kwargs.pop('alpha', 0.0001)
    fit_intercept = kwargs.pop('fit_intercept', True)
    n_iter = kwargs.pop('n_iter', 5)
    shuffle = kwargs.pop('suffle', True)
    verbose = kwargs.pop('verbose', 0)
    eta0 = kwargs.pop('eta0', 1.0)
    n_jobs = kwargs.pop('n_jobs', -1)
    random_state = kwargs.pop('random_state', 0)
    class_weight = kwargs.pop('class_weight', None)
    warm_start = kwargs.pop('warm_start', False)

    # Call the constructor with the proper input arguments
    cp = Perceptron(penalty=penalty, alpha=alpha, fit_intercept=fit_intercept, n_iter=n_iter, shuffle=shuffle, 
                             verbose=verbose, eta0=eta0, n_jobs=n_jobs, random_state=random_state, 
                             class_weight=class_weight, warm_start=warm_start)

    # Train the classifier
    cp.fit(training_data, training_label)

    return cp

def TestPerceptron(cp, testing_data):
    
    # Import Perceptron from scikit learn
    from sklearn.linear_model import Perceptron

    # Test the classifier
    pred_prob = cp.decision_function(testing_data)
    pred_label = cp.predict(testing_data)

    return (pred_prob, pred_label)

############################## LDA CLASSIFIER ##############################

def TrainLDA(training_data, training_label, **kwargs):
    
    # Import LDA from scikit learn
    from sklearn.lda import LDA

    # Unpack the keywords to create the classifier
    solver = kwargs.pop('solver', 'svd')
    shrinkage = kwargs.pop('shrinkage', None)
    priors = kwargs.pop('priors', None)
    n_components = kwargs.pop('n_components', None)
    store_covariance = kwargs.pop('store_covariance', False)
    tol = kwargs.pop('tol', 0.0001)

    # Call the constructor with the proper input arguments
    clda = LDA(solver=solver, shrinkage=shrinkage, priors=priors, n_components=n_components, 
              store_covariance=store_covariance, tol=tol)

    # Train the classifier
    clda.fit(training_data, training_label)

    return clda

def TestLDA(clda, testing_data):
    
    # Import LDA from scikit learn
    from sklearn.lda import LDA

    # Test the classifier
    pred_prob = clda.predict_proba(testing_data)
    pred_label = clda.predict(testing_data)

    return (pred_prob, pred_label)

############################## QDA CLASSIFIER ##############################

def TrainQDA(training_data, training_label, **kwargs):
    
    # Import LDA from scikit learn
    from sklearn.qda import QDA

    # Unpack the keywords to create the classifier
    priors = kwargs.pop('priors', None)
    reg_param = kwargs.pop('reg_param', 0.0)

    # Call the constructor with the proper input arguments
    cqda = QDA(priors=priors, reg_param=reg_param)

    # Train the classifier
    cqda.fit(training_data, training_label)

    return cqda

def TestQDA(cqda, testing_data):
    
    # Import QDA from scikit learn
    from sklearn.qda import QDA

    # Test the classifier
    pred_prob = cqda.predict_proba(testing_data)
    pred_label = cqda.predict(testing_data)

    return (pred_prob, pred_label)

############################## LINEAR SVM CLASSIFIER ##############################

def TrainLinearSVM(training_data, training_label, **kwargs):
    
    # Import Linear SVM from scikit learn
    from sklearn.svm import LinearSVC

    # Unpack the keywords to create the classifier
    penalty = kwargs.pop('penalty', 'l2')
    loss = kwargs.pop('loss', 'squared_hinge')
    dual = kwargs.pop('dual', True)
    tol = kwargs.pop('tol', 0.0001)
    multi_class = kwargs.pop('multi_class', 'ovr')
    fit_intercept = kwargs.pop('fit_intercept', True)
    intercept_scaling = kwargs.pop('intercept_scaling', 1)
    class_weight = kwargs.pop('class_weight', None)
    verbose = kwargs.pop('verbose', 0)
    random_state = kwargs.pop('random_state', 0)
    max_iter = kwargs.pop('max_iter', 1000)
    gs_n_jobs = kwargs.pop('gs_n_jobs', multiprocessing.cpu_count())
    
    # Make a grid_search if the parameter C is not given
    if ('C' in kwargs):
        C = kwargs.pop('C', 1.0)

        # Call the constructor with the proper input arguments
        clsvm = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class, 
                          fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, 
                          verbose=verbose, random_state=random_state, max_iter=max_iter)

    else:
        # Import the grid search module
        from sklearn import grid_search

        # Create the parameter grid
        C = {'C': np.logspace(-5., 15., num=11, endpoint=True, base=2.0)}
        
        # Create the linear SVM object without the C parameters
        clsvm_gs = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, multi_class=multi_class, 
                          fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, 
                          verbose=verbose, random_state=random_state, max_iter=max_iter)

        # Create the different linear SVM object using the grid search
        clsvm = grid_search.GridSearchCV(clsvm_gs, C, n_jobs=gs_n_jobs)

    # Train the classifier
    clsvm.fit(training_data, training_label)

    return clsvm

def TestLinearSVM(clsvm, testing_data):
    
    # Import Linear SVM from scikit learn
    from sklearn.svm import LinearSVC

    # Test the classifier
    pred_prob = clsvm.decision_function(testing_data)
    pred_label = clsvm.predict(testing_data)

    return (pred_prob, pred_label)

############################## KERNEL SVM CLASSIFIER ##############################

def TrainKernelSVM(training_data, training_label, **kwargs):
    
    # Import Kernel SVM from scikit learn
    from sklearn.svm import SVC

    # Unpack the keywords to create the classifier
    #C = kwargs.pop('C', 1.0)
    kernel = kwargs.pop('kernel', 'rbf')
    #degree = kwargs.pop('degree', 3)
    #gamma = kwargs.pop('gamma', 0.0)
    coef0 = kwargs.pop('coef0', 0.0)
    probability = kwargs.pop('probability', False)
    shrinking = kwargs.pop('shrinking', True)
    tol = kwargs.pop('tol', 0.0001)
    cache_size = kwargs.pop('cache_size', 200)
    class_weight = kwargs.pop('class_weight', None)
    verbose = kwargs.pop('verbose', False)
    max_iter = kwargs.pop('max_iter', -1)
    random_state = kwargs.pop('random_state', None)
    gs_n_jobs = kwargs.pop('gs_n_jobs', multiprocessing.cpu_count())
    
    if (verbose):
        print 'SVM is the {} kernel'.format(kernel)
    
    # Check which kernel is needed
    if ((kernel == 'rbf')    or 
        (kernel == 'sigmoid')   ):

        # Check if the parameter C is given
        if ('C' in kwargs):
            C = kwargs.pop('C', 1.0)
            # Check if the parameter gamma is specified
            if ('gamma' in kwargs):
                gamma = kwargs.pop('gamma', 0.0)

                if (verbose):
                    print 'SVM training without grid search'

                # Call the constructor with the proper input arguments
                cksvm = SVC(C=C, gamma=gamma, coef0=coef0, kernel=kernel, probability=probability, shrinking=shrinking, 
                            tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, 
                            max_iter=max_iter, random_state=random_state)

            # We need to make a grid search for the parameter gamma
            else:
                # Import the grid search module
                from sklearn import grid_search

                # Create the parameter grid
                gamma = {'gamma': np.logspace(-15, 3., num=10, endpoint=True, base=2.0)}
                
                if (verbose):
                    print 'SVM training with grid-search for the parameter gamma'

                # Create the linear SVM object without the C parameters
                cksvm_gs = SVC(C=C, coef0=coef0, kernel=kernel, probability=probability, shrinking=shrinking, tol=tol, 
                            cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                            random_state=random_state)

                # Create the different linear SVM object using the grid search
                cksvm = grid_search.GridSearchCV(cksvm_gs, gamma, n_jobs=gs_n_jobs)

        # We need a grid search for the parameter C
        else:
            # We call dict_svm in case we need to add some other parameters
            dict_svm = {'C': np.logspace(-5., 15., num=11, endpoint=True, base=2.0)}

            # Check if the parameter gamma is specified
            if ('gamma' in kwargs):
                # Import the grid search module
                from sklearn import grid_search

                gamma = kwargs.pop('gamma', 0.0)

                if (verbose):
                    print 'SVM training with grid-search for the parameter C'

                # Create the linear SVM object without the C parameters
                cksvm_gs = SVC(gamma=gamma, coef0=coef0, kernel=kernel, probability=probability, shrinking=shrinking, tol=tol, 
                               cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                               random_state=random_state)

                # Create the different linear SVM object using the grid search
                cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)

            # We need also to make a grid-search for the param gamma
            else:
                # Import the grid search module
                from sklearn import grid_search

                dict_svm['gamma'] = np.logspace(-15, 3., num=10, endpoint=True, base=2.0)
                
                if (verbose):
                    print 'SVM training with grid-search for the parameter C and gamma'

                # Create the linear SVM object without the C parameters
                cksvm_gs = SVC(kernel=kernel, coef0=coef0, probability=probability, shrinking=shrinking, tol=tol, 
                               cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                               random_state=random_state)

                # Create the different linear SVM object using the grid search
                cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)

    elif (kernel == 'poly'):

        # Check if the parameter C is given
        if ('C' in kwargs):
            C = kwargs.pop('C', 1.0)
            # Check if the parameter gamma is specified
            if ('gamma' in kwargs):
                gamma = kwargs.pop('gamma', 0.0)

                # Check if the parameter degree is specified
                if ('degree' in kwargs):
                    degree = kwargs.pop('degree', 3)

                    if (verbose):
                        print 'SVM training without grid search'

                    # Call the constructor with the proper input arguments
                    cksvm = SVC(C=C, degree=degree, gamma=gamma, coef0=coef0, kernel=kernel, probability=probability, 
                                shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                verbose=verbose, max_iter=max_iter, random_state=random_state)
                else:
                    
                    # Import the grid search module
                    from sklearn import grid_search

                    # Search for degree of order 2 to 6
                    degree = {'degree': np.arange(2, 7, 1, dtype=int)}

                    if (verbose):
                        print 'SVM training with grid-search for the parameter degree'

                    # Create the linear SVM object without the C parameters
                    cksvm_gs = SVC(C=C, gamma=gamma, coef0=coef0, kernel=kernel, probability=probability, 
                                   shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                   verbose=verbose, max_iter=max_iter, random_state=random_state)

                    # Create the different linear SVM object using the grid search
                    cksvm = grid_search.GridSearchCV(cksvm_gs, degree, n_jobs=gs_n_jobs)

            # We need to make a grid search for the parameter gamma
            else:
                # Create the parameter grid
                dict_svm = {'gamma': np.logspace(-15, 3., num=10, endpoint=True, base=2.0)}

                # Check if the paramter degree was assigned
                if ('degree' in kwargs):
                    degree = kwargs.pop('degree', 3)
                    
                    if (verbose):
                        print 'SVM training with grid-search for the parameter gamma'

                    # Import the grid search module
                    from sklearn import grid_search

                    # Create the linear SVM object without the C parameters
                    cksvm_gs = SVC(C=C, degree=degree, coef0=coef0, kernel=kernel, probability=probability, 
                                   shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                   verbose=verbose, max_iter=max_iter, random_state=random_state)

                    # Create the different linear SVM object using the grid search
                    cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)

                else:
                    dict_svm['degree'] = np.arange(2, 7, 1, dtype=int)

                    if (verbose):
                        print 'SVM training with grid-search for the parameter gamma and degree'

                    # Import the grid search module
                    from sklearn import grid_search

                    # Create the linear SVM object without the C parameters
                    cksvm_gs = SVC(C=C, coef0=coef0, kernel=kernel, probability=probability, 
                                   shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                   verbose=verbose, max_iter=max_iter, random_state=random_state)

                    # Create the different linear SVM object using the grid search
                    cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)

        # We need a grid search for the parameter C
        else:
            # We call dict_svm in case we need to add some other parameters
            dict_svm = {'C': np.logspace(-5., 15., num=11, endpoint=True, base=2.0)}

            # Check if the parameter gamma is specified
            if ('gamma' in kwargs):
                gamma = kwargs.pop('gamma', 0.0)

                # Check if the parameter degree is specified
                if ('degree' in kwargs):
                    
                    # Import the grid search module
                    from sklearn import grid_search

                    degree = kwargs.pop('degree', 3)

                    if (verbose):
                        print 'SVM training with grid-search for the parameter C'

                    # Create the linear SVM object without the C parameters
                    cksvm_gs = SVC(degree=degree, gamma=gamma, coef0=coef0, kernel=kernel, probability=probability, 
                                   shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                   verbose=verbose, max_iter=max_iter, random_state=random_state)

                    # Create the different linear SVM object using the grid search
                    cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)
                else:
                    
                    # Import the grid search module
                    from sklearn import grid_search

                    # Search for degree of order 2 to 6
                    dict_svm['degree'] = np.arange(2, 7, 1, dtype=int)

                    if (verbose):
                        print 'SVM training with grid-search for the parameter degree and C'

                    # Create the linear SVM object without the C parameters
                    cksvm_gs = SVC(gamma=gamma, coef0=coef0, kernel=kernel, probability=probability, 
                                   shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                   verbose=verbose, max_iter=max_iter, random_state=random_state)

                    # Create the different linear SVM object using the grid search
                    cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)

            # We need to make a grid search for the parameter gamma
            else:
                # Create the parameter grid
                dict_svm['gamma'] = np.logspace(-15, 3., num=10, endpoint=True, base=2.0)

                # Check if the paramter degree was assigned
                if ('degree' in kwargs):
                    degree = kwargs.pop('degree', 3)
                    
                    if (verbose):
                        print 'SVM training with grid-search for the parameter C and gamma'

                    # Import the grid search module
                    from sklearn import grid_search

                    # Create the linear SVM object without the C parameters
                    cksvm_gs = SVC(degree=degree, coef0=coef0, kernel=kernel, probability=probability, 
                                   shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                   verbose=verbose, max_iter=max_iter, random_state=random_state)

                    # Create the different linear SVM object using the grid search
                    cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)

                else:
                    dict_svm['degree'] = np.arange(2, 7, 1, dtype=int)

                    if (verbose):
                        print 'SVM training with grid-search for the parameter C, gamma, and degree'

                    # Import the grid search module
                    from sklearn import grid_search

                    # Create the linear SVM object without the C parameters
                    cksvm_gs = SVC(coef0=coef0, kernel=kernel, probability=probability, 
                                   shrinking=shrinking, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                   verbose=verbose, max_iter=max_iter, random_state=random_state)

                    # Create the different linear SVM object using the grid search
                    cksvm = grid_search.GridSearchCV(cksvm_gs, dict_svm, n_jobs=gs_n_jobs)

    # Linear kernel
    elif (kernel == 'linear'):
        # Make a grid_search if the parameter C is not given
        if ('C' in kwargs):
            C = kwargs.pop('C', 1.0)

            # Call the constructor with the proper input arguments
            cksvm = SVC(C=C, kernel=kernel, probability=probability, shrinking=shrinking, tol=tol, 
                        cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                        random_state=random_state)
        else:
            # Import the grid search module
            from sklearn import grid_search

            # Create the parameter grid
            C = {'C': np.logspace(-5., 15., num=11, endpoint=True, base=2.0)}

            # Create the linear SVM object without the C parameters
            cksvm_gs = SVC(kernel=kernel, probability=probability, shrinking=shrinking, tol=tol, 
                        cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                        random_state=random_state)

            # Create the different linear SVM object using the grid search
            cksvm = grid_search.GridSearchCV(cksvm_gs, C)

    else:
        raise ValueError('protoclass.classification.classification.KernelSVM: Wrong kernel type.')


    # Train the classifier
    cksvm.fit(training_data, training_label)

    return cksvm

def TestKernelSVM(cksvm, testing_data):
    
    # Import Linear SVM from scikit learn
    from sklearn.svm import SVC

    # Test the classifier
    pred_prob = cksvm.decision_function(testing_data)
    pred_label = cksvm.predict(testing_data)

    return (pred_prob, pred_label)

############################## K-NEAREST NEIGHBOUR CLASSIFIER ##############################

def TrainKNN(training_data, training_label, **kwargs):
    
    # Import KNN from scikit learn
    from sklearn.neighbors import KNeighborsClassifier

    # Unpack the keywords to create the classifier
    weights = kwargs.pop('weights', 'uniform')
    algorithm = kwargs.pop('algorithm', 'auto')
    leaf_size = kwargs.pop('leaf_size', 30)
    p = kwargs.pop('p', 2)
    metric = kwargs.pop('metric', 'minkowski')
    metric_params = kwargs.pop('metric_params', None)
    gs_n_jobs = kwargs.pop('gs_n_jobs', multiprocessing.cpu_count())

    # If the number of estimators is not specified, it will be find by cross-validation
    if 'n_neighbors' in kwargs:
        n_neighbors = kwargs.pop('n_neighbors', 5)

        # Construct the KNN classifier
        cknn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
                                    p=p, metric=metric, metric_params=metric_params)

    else:
        # Import the function to perform the grid search
        from sklearn import grid_search

        # Create the parametor grid
        n_neighbors = {'n_neighbors': np.arange(3, 9, 2, dtype=int)}

        # Create the KNN without the parameters to find during the grid search
        
        cknn_gs = KNeighborsClassifier(weights=weights, algorithm=algorithm, leaf_size=leaf_size,
                                    p=p, metric=metric, metric_params=metric_params)

        # Create the different random forest to try specifying the grid search parametors
        cknn = grid_search.GridSearchCV(cknn_gs, n_neighbors, n_jobs=gs_n_jobs)

    # Train the classifier
    cknn.fit(training_data, training_label)

    return cknn

def TestKNN(cknn, testing_data):
    
    # Import KNN from scikit learn
    from sklearn.neighbors import KNeighborsClassifier

    # Test the classifier
    pred_prob = cknn.predict_proba(testing_data)
    pred_label = cknn.predict(testing_data)

    return (pred_prob, pred_label)

############################## ADABOOST CLASSIFICATION ##############################

def TrainAdaBoost(training_data, training_label, **kwargs):
    
    # Import Random Forest from scikit learn
    from sklearn.ensemble import AdaBoostClassifier

    # Unpack the keywords to create the classifier
    base_estimator_name = kwargs.pop('base_estimator_name', 'decision-tree')
    algorithm = kwargs.pop('algorithm', 'SAMME.R')
    random_state= kwargs.pop('random_state', None)
    gs_n_jobs = kwargs.pop('gs_n_jobs', multiprocessing.cpu_count())
    verbose = kwargs.pop('verbose', False)

    # Create the weak learner
    if (base_estimator_name == 'decision-tree'):
        from sklearn.tree import DecisionTreeClassifier
        cbe = DecisionTreeClassifier()
    else:
        raise ValueError('protoclass.classification.classificatio.ababoost: base estimator not supported.')


    # If the number of estimators is not specified, it will be find by cross-validation
    if ('n_estimators' in kwargs):
        n_estimators = kwargs.pop('n_estimators', 10)

        # Check that the learning rate is provided
        if ('learning_rate' in kwargs):
            learning_rate = kwargs.pop('learning_rate', 1.0)

            if verbose:
                print 'AdaBoost performed with no grid-search'

            # Call the constructor
            cadb = AdaBoostClassifier(cbe, n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm,
                            random_state=random_state)

        else:
            # Create a dictionnary with the different learning rate to try
            dict_adb = {'learning_rate': np.linspace(.1, 1., 10)}

            # Import the function to perform the grid search
            from sklearn import grid_search

            if verbose:
                print 'AdaBoost performed with a grid-search on the learning-rate'

            # Call the constructor
            cadb_gs = AdaBoostClassifier(cbe, n_estimators=n_estimators, algorithm=algorithm, random_state=random_state)

            # Create the grid search classifier
            cadb = grid_search.GridSearchCV(cadb_gs, dict_adb, n_jobs=gs_n_jobs)
    else:
        # Create a dictionnary for the number of estimators
        dict_adb = {'n_estimators': np.linspace(1, 1000, 10, dtype=int)}

        # Check that the learning rate is provided
        if ('learning_rate' in kwargs):
            learning_rate = kwargs.pop('learning_rate', 1.0)

            # Import the function to perform the grid search
            from sklearn import grid_search

            if verbose:
                print 'AdaBoost performed with a grid-search on the numbers of estimators'

            # Call the constructor
            cadb_gs = AdaBoostClassifier(cbe, learning_rate=learning_rate, algorithm=algorithm, 
                                        random_state=random_state)

            # Create the grid search classifier
            cadb = grid_search.GridSearchCV(cadb_gs, dict_adb, n_jobs=gs_n_jobs)

        else:
            # Create a dictionnary with the different learning rate to try
            dict_adb['learning_rate'] = np.linspace(.1, 1., 10)

            # Import the function to perform the grid search
            from sklearn import grid_search

            if verbose:
                print 'AdaBoost performed with a grid-search on the learning-rate and the number of estimators'

            # Call the constructor
            cadb_gs = AdaBoostClassifier(cbe, algorithm=algorithm, random_state=random_state)

            # Create the grid search classifier
            cadb = grid_search.GridSearchCV(cadb_gs, dict_adb, n_jobs=gs_n_jobs)

    # Train the classifier
    cadb.fit(training_data, training_label)

    return cadb

def TestAdaBoost(cadb, testing_data):
    
    # Import AdaBoost from scikit learn
    from sklearn.ensemble import AdaBoostClassifier

    # Test the classifier
    pred_prob = cadb.predict_proba(testing_data)
    pred_label = cadb.predict(testing_data)

    return (pred_prob, pred_label)

############################## GRADIENT BOOSTING CLASSIFICATION ##############################

def TrainGradientBoosting(training_data, training_label, **kwargs):
    
    # Import Random Forest from scikit learn
    from sklearn.ensemble import GradientBoostingClassifier

    # Unpack the keywords to create the classifier
    loss = kwargs.pop('loss', 'deviance')
    subsample = kwargs.pop('subsample', 1.0)
    min_samples_split = kwargs.pop('min_samples_split', 2)
    min_samples_leaf = kwargs.pop('min_samples_leaf', 1)
    min_weight_fraction_leaf = kwargs.pop('min_weight_fraction_leaf', 0.0)
    max_depth = kwargs.pop('max_depth', 3)
    init = kwargs.pop('init', None)
    random_state = kwargs.pop('random_state', None)
    max_features = kwargs.pop('max_features', None)
    max_leaf_nodes = kwargs.pop('max_leaf_nodes', None)
    warm_start = kwargs.pop('warm_start', False)
    gs_n_jobs = kwargs.pop('gs_n_jobs', multiprocessing.cpu_count())
    verbose = kwargs.pop('verbose', False)

    # If the number of estimators is not specified, it will be find by cross-validation
    if ('n_estimators' in kwargs):
        n_estimators = kwargs.pop('n_estimators', 10)

        # Check that the learning rate is provided
        if ('learning_rate' in kwargs):
            learning_rate = kwargs.pop('learning_rate', 1.0)

            if verbose:
                print 'Gradient Boosting performed with no grid-search'

            # Call the constructor
            cgb = GradientBoostingClassifier(loss=loss, subsample=subsample, min_samples_split=min_samples_split, 
                                             min_samples_leaf=min_samples_leaf, 
                                             min_weight_fraction_leaf=min_weight_fraction_leaf,
                                             max_depth=max_depth, init=init, random_state=random_state, 
                                             max_features=max_features, max_leaf_nodes=max_leaf_nodes, 
                                             warm_start=warm_start, learning_rate=learning_rate, 
                                             n_estimators=n_estimators)

        else:
            # Create a dictionnary with the different learning rate to try
            dict_adb = {'learning_rate': np.linspace(.1, 1., 10)}

            # Import the function to perform the grid search
            from sklearn import grid_search

            if verbose:
                print 'Gradient Boosting performed with a grid-search on the learning-rate'

            # Call the constructor
            cgb_gs = GradientBoostingClassifier(loss=loss, subsample=subsample, min_samples_split=min_samples_split, 
                                                min_samples_leaf=min_samples_leaf, 
                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                max_depth=max_depth, init=init, random_state=random_state, 
                                                max_features=max_features, max_leaf_nodes=max_leaf_nodes, 
                                                warm_start=warm_start, n_estimators=n_estimators)

            # Create the grid search classifier
            cgb = grid_search.GridSearchCV(cadb_gs, dict_adb, n_jobs=gs_n_jobs)
    else:
        # Create a dictionnary for the number of estimators
        dict_adb = {'n_estimators': np.linspace(1, 1000, 10, dtype=int)}

        # Check that the learning rate is provided
        if ('learning_rate' in kwargs):
            learning_rate = kwargs.pop('learning_rate', 1.0)

            # Import the function to perform the grid search
            from sklearn import grid_search

            if verbose:
                print 'Gradient Boosting performed with a grid-search on the numbers of estimators'

            # Call the constructor
            cgb_gs = GradientBoostingClassifier(loss=loss, subsample=subsample, min_samples_split=min_samples_split, 
                                                min_samples_leaf=min_samples_leaf, 
                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                max_depth=max_depth, init=init, random_state=random_state, 
                                                max_features=max_features, max_leaf_nodes=max_leaf_nodes, 
                                                warm_start=warm_start, learning_rate=learning_rate)

            # Create the grid search classifier
            cgb = grid_search.GridSearchCV(cgb_gs, dict_adb, n_jobs=gs_n_jobs)

        else:
            # Create a dictionnary with the different learning rate to try
            dict_adb['learning_rate'] = np.linspace(.1, 1., 10)

            # Import the function to perform the grid search
            from sklearn import grid_search

            if verbose:
                print 'Gradient Boosting performed with a grid-search on the learning-rate and the number of estimators'

            # Call the constructor
            cgb_gs = GradientBoostingClassifier(loss=loss, subsample=subsample, min_samples_split=min_samples_split, 
                                                min_samples_leaf=min_samples_leaf, 
                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                max_depth=max_depth, init=init, random_state=random_state, 
                                                max_features=max_features, max_leaf_nodes=max_leaf_nodes, 
                                                warm_start=warm_start)

            # Create the grid search classifier
            cgb = grid_search.GridSearchCV(cgb_gs, dict_adb, n_jobs=gs_n_jobs)

    # Train the classifier
    cgb.fit(training_data, training_label)

    return cgb

def TestGradientBoosting(cgb, testing_data):
    
    # Import Gradient Boosting from scikit learn
    from sklearn.ensemble import GradientBoostingClassifier

    # Test the classifier
    pred_prob = cgb.predict_proba(testing_data)
    pred_label = cgb.predict(testing_data)

    return (pred_prob, pred_label)
