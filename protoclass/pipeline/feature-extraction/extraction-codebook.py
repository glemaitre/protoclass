#title           :extraction_codebook.py
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

# Read the csv file with the ground truth
#gt_csv_filename = '/DATA/OCT/data_organized/data.csv'
gt_csv_filename = '/work/le2i/gu5306le/OCT/data.csv'
gt_csv = pd.read_csv(gt_csv_filename)

gt = gt_csv.values

data_filename = gt[:, 0]

# Get the good extension
radius = 2
data_filename = np.array([f + '_nlm_lbp_' + str(radius) + '_hist_now.npz' for f in data_filename])

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

    data_folder = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_now_data_npz'

    codebook_list = []
    for idx_test, (pat_test_norm, pat_test_dme) in enumerate(zip(filename_normal, filename_dme)):

        pat_train_norm = np.delete(filename_normal, idx_test)
        pat_train_dme = np.delete(filename_dme, idx_test)

        # Load the training data
        # Define the name of the volume

        # TODO LAMBDA FUNCTION
        # get_lbp_data = lambda f: np.load(join(data_folder,f))['vol_lbp']

        # open the training data
        vol_name = 'vol_lbp_hist'
        training_data = np.concatenate((np.concatenate([np.load(join(data_folder, f))[vol_name] for f in pat_train_norm], axis=0),
                                        np.concatenate([np.load(join(data_folder, f))[vol_name] for f in pat_train_dme], axis=0)),
                                       axis=0)

        # open the testing data
        print join(data_folder, pat_test_norm)
        print join(data_folder, pat_test_dme)
        testing_data = np.concatenate((np.load(join(data_folder, pat_test_norm))[vol_name], np.load(join(data_folder, pat_test_dme))[vol_name]), axis=0)

        # Create the codebook using the training data
        num_cores = 5
        list_n_words = [2, 4, 8, 16, 32, 64, 80, 100, 150, 200, 250, 300, 350, 400]
        cbook = [CodeBook(n_words=w, init='random', n_jobs=num_cores, n_init=1, cluster_core='random-words') for w in list_n_words]

        # Fit each code book for the data currently open
        for idx_cb, c in enumerate(cbook):
            print 'Fitting for dictionary with {} words'.format(list_n_words[idx_cb])
            c.fit(training_data)

        # Concatenate the codebook
        codebook_list.append(cbook)

    # We have to store the final codebook
    path_to_save = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_now_codebook_random'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    from sklearn.externals import joblib
    joblib.dump(codebook_list, join(path_to_save, 'codebook.pkl'))
