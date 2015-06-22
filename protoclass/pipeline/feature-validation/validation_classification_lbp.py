import numpy as np
from sklearn.externals import joblib

# print ''
# print 'LBP histogram'
# print ''

# for radius in range(1, 5):
#     path_result = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_results/hist.pkl'

#     # Load the results from the given path
#     result = joblib.load(path_result)
#     result_array = np.array(result)

#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in result_array[:, 0]:
        
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- RADIUS  {} -----'.format(radius)
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[0]),
#                                                      normal / float(result_array.shape[0]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[0]),
#                                                  normal,
#                                                  float(result_array.shape[0]))
#     print ''

# print ''
# print 'LBP histogram + PCA'
# print ''
    
# for radius in range(1, 5):
#     path_result = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_pca_results/hist.pkl'

#     # Load the results from the given path
#     result = joblib.load(path_result)
#     result_array = np.array(result)

#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in result_array[:, 0]:
        
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- RADIUS  {} -----'.format(radius)
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[0]),
#                                                      normal / float(result_array.shape[0]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[0]),
#                                                  normal,
#                                                  float(result_array.shape[0]))
#     print ''

# print ''
# print 'LBP combination'
# print ''

# path_result = '/work/le2i/gu5306le/OCT/lbp_hist_comb_results/hist.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# normal = 0.
# dme = 0.
# # Extract the results on the testing set
# for cv in result_array[:, 0]:
        
#     if cv[0] == 0:
#         normal += 1.
#     if cv[1] == 1:
#         dme += 1.

# print 'The statistic are the following:'
# print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[0]),
#                                                  normal / float(result_array.shape[0]))
# print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                              float(result_array.shape[0]),
#                                              normal,
#                                              float(result_array.shape[0]))
# print ''

# print ''
# print 'LBP combination + PCA'
# print ''

# path_result = '/work/le2i/gu5306le/OCT/lbp_hist_comb_pca_results/hist.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# normal = 0.
# dme = 0.
# # Extract the results on the testing set
# for cv in result_array[:, 0]:
        
#     if cv[0] == 0:
#         normal += 1.
#     if cv[1] == 1:
#         dme += 1.

# print 'The statistic are the following:'
# print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[0]),
#                                                  normal / float(result_array.shape[0]))
# print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                              float(result_array.shape[0]),
#                                              normal,
#                                              float(result_array.shape[0]))
# print ''

# print ''
# print 'LBP histogram + BOW'
# print ''
    
# nw = [2, 4, 8, 16, 32, 64, 100]

# for radius in range(1, 5):

#     path_result = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_BoW_results/bow.pkl'

#     # Load the results from the given path
#     result = joblib.load(path_result)
#     result_array = np.array(result)

#     # Swap the two first axis
#     result_array = np.rollaxis(result_array, 0, 2)

#     # Go throught the different words
#     for idx_w, w in enumerate(result_array):
    
#         normal = 0.
#         dme = 0.
#         # Extract the results on the testing set
#         for cv in w[:, 0]:
            
#             if cv[0] == 0:
#                 normal += 1.
#             if cv[1] == 1:
#                 dme += 1.

#         print '----- RADIUS  {} -----'.format(radius)
#         print '----- # WORDS {} -----'.format(nw[idx_w])
#         print 'The statistic are the following:'
#         print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                          normal / float(result_array.shape[1]))
#         print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                      float(result_array.shape[1]),
#                                                      normal,
#                                                      float(result_array.shape[1]))
#         print ''

# print ''
# print 'LBP histogram + BOW + combine'
# print ''
        
# nw = [2, 4, 8, 16, 32]

# path_result = '/work/le2i/gu5306le/OCT/lbp_hist_BoW_comb_results/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''

        
# print ''
# print 'LBP histogram + sliding window + BOW'
# print ''

    
# nw = [2, 4, 8, 16, 32]

# for radius in range(1, 5):

#     path_result = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_BoW_now_results/bow.pkl'

#     # Load the results from the given path
#     result = joblib.load(path_result)
#     result_array = np.array(result)

#     # Swap the two first axis
#     result_array = np.rollaxis(result_array, 0, 2)

#     # Go throught the different words
#     for idx_w, w in enumerate(result_array):
    
#         normal = 0.
#         dme = 0.
#         # Extract the results on the testing set
#         for cv in w[:, 0]:
            
#             if cv[0] == 0:
#                 normal += 1.
#             if cv[1] == 1:
#                 dme += 1.

#         print '----- RADIUS  {} -----'.format(radius)
#         print '----- # WORDS {} -----'.format(nw[idx_w])
#         print 'The statistic are the following:'
#         print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                          normal / float(result_array.shape[1]))
#         print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                      float(result_array.shape[1]),
#                                                      normal,
#                                                      float(result_array.shape[1]))
#         print ''

# print ''
# print 'LBP histogram + sliding window + BOW Complementsx'
# print ''

    
# nw = [40, 60, 80, 100]

# radius = 2

# path_result = '/work/le2i/gu5306le/OCT/lbp_r_' + str(radius) + '_hist_BoW_now_results2/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
        
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- RADIUS  {} -----'.format(radius)
#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''


# print ''
# print 'LBP histogram + sliding window + BOW + combine'
# print ''
        
# nw = [2, 4, 8, 16, 32]

# path_result = '/work/le2i/gu5306le/OCT/lbp_hist_BoW_comb_results/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''

# print ''
# print 'LBP histogram + BOW + combine'
# print ''
        
# nw = [200]

# path_result = '/work/le2i/gu5306le/OCT/lbp_r_2_hist_BoW_now_results_200/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''

# print 'DUKE DATSET'
# print ''
# print 'LBP histogram + BOW + sliding window'
# print ''

# nw = [2, 4, 8, 16, 32]

# path_result = '/work/le2i/gu5306le/dukeOCT/dataset/lbp_r_1_hist_BoW_now_results/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''

# print ''
# print 'LBP histogram + BOW + sliding window'
# print ''

# nw = [2, 4, 8, 16, 32]

# path_result = '/work/le2i/gu5306le/dukeOCT/dataset/lbp_r_2_hist_BoW_now_results/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''

# print ''
# print 'LBP histogram + BOW + sliding window'
# print ''

# nw = [2, 4, 8, 16, 32]

# path_result = '/work/le2i/gu5306le/dukeOCT/dataset/lbp_r_3_hist_BoW_now_results/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(palette="Set2")

# print ''
# print 'LBP histogram + Random BOW + sliding window'
# print ''

# nw = [2, 4, 8, 16, 32, 64, 80, 100, 150, 200, 250, 300, 350, 400]

# specificity_try = []
# sensitivity_try = []
# accuracy_try = []
# for mtry in range(1, 6):

#     path_result = '/work/le2i/gu5306le/OCT/lbp_r_3_hist_BoW_now_random_results_'+str(mtry)+'/bow.pkl'

#     # Load the results from the given path
#     result = joblib.load(path_result)
#     result_array = np.array(result)

#     # Swap the two first axis
#     result_array = np.rollaxis(result_array, 0, 2)

#     # Go throught the different words
#     specificity_word = []
#     sensitivity_word = []
#     accuracy_word = []
#     for idx_w, w in enumerate(result_array):
    
#         normal = 0.
#         dme = 0.
#         # Extract the results on the testing set
#         for cv in w[:, 0]:
            
#             if cv[0] == 0:
#                 normal += 1.
#             if cv[1] == 1:
#                 dme += 1.

#         # Append the results for this word
#         specificity_word.append(normal / float(result_array.shape[1]))
#         sensitivity_word.append(dme / float(result_array.shape[1]))
#         accuracy_word.append((normal + dme) / float(result_array.shape[1] * 2.))

#         print '----- # WORDS {} -----'.format(nw[idx_w])
#         print 'The statistic are the following:'
#         print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                          normal / float(result_array.shape[1]))
#         print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                      float(result_array.shape[1]),
#                                                      normal,
#                                                      float(result_array.shape[1]))
#         print ''

    
#     specificity_try.append(specificity_word)
#     sensitivity_try.append(sensitivity_word)
#     accuracy_try.append(accuracy_word)

# specificity_try = np.array(specificity_try)
# sensitivity_try = np.array(sensitivity_try)
# accuracy_try = np.array(accuracy_try)

# spe_inter = []
# sen_inter = []
# acc_inter = []
# from scipy import interpolate
# for mtry in range(5):
#     def intdata(x, y):
#         f = interpolate.interp1d(x, y, kind="cubic")
#         xnew = np.linspace(2, 400, 399)
#         return f(xnew)

#     spe_inter.append(intdata(np.array(nw), specificity_try[mtry, :]))
#     sen_inter.append(intdata(np.array(nw), sensitivity_try[mtry, :]))
#     acc_inter.append(intdata(np.array(nw), accuracy_try[mtry, :]))

# #color_map = dict(slow="indianred", average="darkseagreen", fast="steelblue")
# sns.tsplot(spe_inter, color="indianred", condition="Specificity")
# sns.tsplot(sen_inter, color="darkseagreen", condition="Sensitivity")
# sns.tsplot(acc_inter, color="steelblue", condition="Accuracy")
# sns.tsplot(specificity_try, time=np.array(nw), color="indianred", err_style="ci_bars", interpolate=False)
# sns.tsplot(sensitivity_try, time=np.array(nw), color="darkseagreen", err_style="ci_bars", interpolate=False)
# sns.tsplot(accuracy_try, time=np.array(nw), color="steelblue", err_style="ci_bars", interpolate=False)
# plt.xlabel('Number of words')
# plt.show()

# print ''
# print 'FINAL - LBP histogram + sliding window + BOW - Radius 1'
# print ''
        
# nw = [32]

# path_result = '/work/le2i/gu5306le/OCT/lbp_r_1_hist_BoW_now_results_final/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''


# print ''
# print 'FINAL - LBP histogram + sliding window + BOW - Radius 2'
# print ''
        
# nw = [32]

# path_result = '/work/le2i/gu5306le/OCT/lbp_r_2_hist_BoW_now_results_final/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''


# print ''
# print 'FINAL - LBP histogram + sliding window + BOW - Radius 3'
# print ''
        
# nw = [32]

# path_result = '/work/le2i/gu5306le/OCT/lbp_r_3_hist_BoW_now_results_final/bow.pkl'

# # Load the results from the given path
# result = joblib.load(path_result)
# result_array = np.array(result)

# # Swap the two first axis
# result_array = np.rollaxis(result_array, 0, 2)

# # Go throught the different words
# for idx_w, w in enumerate(result_array):
    
#     normal = 0.
#     dme = 0.
#     # Extract the results on the testing set
#     for cv in w[:, 0]:
            
#         if cv[0] == 0:
#             normal += 1.
#         if cv[1] == 1:
#             dme += 1.

#     print '----- # WORDS {} -----'.format(nw[idx_w])
#     print 'The statistic are the following:'
#     print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
#                                                      normal / float(result_array.shape[1]))
#     print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
#                                                  float(result_array.shape[1]),
#                                                  normal,
#                                                  float(result_array.shape[1]))
#     print ''

nw = [2, 4, 8, 16, 32]

path_result = '/work/le2i/gu5306le/dukeOCT/dataset/lbp_hist_BoW_now_comb_results/bow.pkl'

# Load the results from the given path
result = joblib.load(path_result)
result_array = np.array(result)

# Swap the two first axis
result_array = np.rollaxis(result_array, 0, 2)

# Go throught the different words
for idx_w, w in enumerate(result_array):
    
    normal = 0.
    dme = 0.
    # Extract the results on the testing set
    for cv in w[:, 0]:
            
        if cv[0] == 0:
            normal += 1.
        if cv[1] == 1:
            dme += 1.

    print '----- # WORDS {} -----'.format(nw[idx_w])
    print 'The statistic are the following:'
    print 'Sensitivity: {} - Specificity: {}'.format(dme / float(result_array.shape[1]),
                                                     normal / float(result_array.shape[1]))
    print 'DME: {} / {} - Normal: {}/ {}'.format(dme,
                                                 float(result_array.shape[1]),
                                                 normal,
                                                 float(result_array.shape[1]))
    print ''
