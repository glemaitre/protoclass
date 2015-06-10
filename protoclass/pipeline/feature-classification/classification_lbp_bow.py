#title           :classification_lbp_now.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/06/07
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
# Panda library
import pandas as pd
# OS library
import os
from os.path import join
# SYS library
import sys

from protoclass.extraction.codebook import *
from protoclass.classification.classification import Classify

# Read the csv file with the ground truth
#gt_csv_filename = '/DATA/OCT/data_organized/data.csv'
gt_csv_filename = '/work/le2i/gu5306le/OCT/data.csv'
gt_csv = pd.read_csv(gt_csv_filename)

gt = gt_csv.values

data_filename = gt[:, 0]

# Get the good extension
radius = 1
data_filename = np.array([f + '_nlm_lbp_' + str(radius) + '_hist.npz' for f in data_filename])

label = gt[:, 1]
label = ((label + 1.) / 2.).astype(int)

from collections import Counter

count_gt = Counter(label)

if (count_gt[0] != count_gt[1]):
    raise ValueError('Not balanced data.')
else:
    # Split data into positive and negative
    # TODO TACKLE USING PERMUTATION OF ELEMENTS
    filename_normal = data_filename[label == 0]
    filename_dme = data_filename[label == 1]

    data_folder = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_data_npz'
    codebook_filename = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_codebook/codebook.pkl'

    get_lbp_data = lambda f: np.load(join(data_folder, f))['vol_lbp_hist']
    from sklearn.externals import joblib

    codebook_list = joblib.load(codebook_filename)
    
    results_cv = []
    for idx_test, (pat_test_norm, pat_test_dme) in enumerate(zip(filename_normal, filename_dme)):

        # Take the testing out and keep the rest for training
        pat_train_norm = np.delete(filename_normal, idx_test)
        pat_train_dme = np.delete(filename_dme, idx_test)

        results_by_codebook = []
        for current_cbook in codebook_list[idx_test]:
            # Collect the current training data
            training_normal = [current_cbook.get_BoF_descriptor(get_lbp_data(f))[0] for f in pat_train_norm]
            training_dme = [current_cbook.get_BoF_descriptor(get_lbp_data(f))[0] for f in pat_train_dme]

            # Compose the training ( data & labels )
            training_data = np.array(training_normal+training_dme)
            training_label = np.array([0]*len(training_normal) + [1]*len(training_dme), dtype=int)

            # Compose the testing
            testing_data = np.array([current_cbook.get_BoF_descriptor(get_lbp_data(filename_normal[idx_test]))[0]
                                     , current_cbook.get_BoF_descriptor(get_lbp_data(filename_dme[idx_test]))[0]])

            # Run the classification for this specific data
            pred_label, roc = Classify(training_data,
                                       training_label,
                                       testing_data,
                                       np.array([0, 1], dtype=int),
                                       classifier_str='random-forest',
                                       n_estimators=100,
                                       n_jobs=20)

            results_by_codebook.append((pred_label, roc))

        results_cv.append(results_by_codebook)

    # We have to store the final codebook
    path_to_save = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_BoW_results'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    from sklearn.externals import joblib
    joblib.dump(codebook_list, join(path_to_save, 'bow.pkl'))
