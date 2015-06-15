"""Testing for Code Book"""

# Get the correct unittest library
try:
    import unittest2 as unittest
except ImportError:
    import unittest

from protoclass.extraction.codebook import *

## -----
import sys

import numpy as np

from sklearn.datasets.samples_generator import make_blobs

from sklearn.utils.testing import assert_equal
# from sklearn.utils.testing import assert_array_equal
# from sklearn.utils.testing import assert_array_almost_equal
# from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises
# from sklearn.utils.testing import assert_raises_regexp
# from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less
# from sklearn.utils.testing import assert_warns
# from sklearn.utils.testing import if_not_mac_os

# from sklearn.utils.extmath import row_norms
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import KMeans, k_means
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.cluster.k_means_ import _labels_inertia
# from sklearn.cluster.k_means_ import _mini_batch_step
# from sklearn.externals.six.moves import cStringIO as StringIO


class Test_data_convert(unittest.TestCase):

    def test_random_codebook(self):
        # non centered, sparse centers to check the
        centers = np.array([
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
            ])
        n_samples = 100
        n_clusters, n_features = centers.shape
        X, true_labels = make_blobs(n_samples=n_samples, centers=centers,
                                    cluster_std=1., random_state=42)

        cbook = CodeBook(n_words=3, init='random', n_init=1, max_iter=1)
        cbook = cbook.fit(X)


    # def _check_fitted_model(self):
    def test_fitted_model(self):

        # non centered, sparse centers to check the
        centers = np.array([
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
            ])
        n_samples = 100
        n_clusters, n_features = centers.shape
        X, true_labels = make_blobs(n_samples=n_samples, centers=centers,
                                    cluster_std=1., random_state=42)

        cbook = CodeBook(n_words=3, max_iter=1)
        cbook = cbook.fit(X) # TODO: Is it neaded to reasign? or it can be just cbook.fit(X)

        # check that the number of clusters centers and distinct labels match
        # the expectation
        centers = cbook.get_dictionary()
        assert_equal(centers.shape, (n_clusters, n_features))

        labels = cbook.predict(X)
        assert_equal(np.unique(labels).shape[0], n_clusters)

        # check that the labels assignment are perfect (up to a permutation)
        assert_equal(v_measure_score(true_labels, labels), 1.0)
        assert_greater(cbook.cluster_core.inertia_, 0.0)

        # check that the descriptor looks like the homogenous PDF used
        # to create the original samples
        cbook_hist = cbook.get_BoF_descriptor(X)
        expected_value = float(1)/cbook.n_words
        for bin_value in cbook_hist[0]:
            assert_less(round(bin_value-expected_value,3), 0.01)
            
        #TODO: recode this as assert_almost_equal but with higher tolerance
        #      since assert_almost_equal is assert(round(a-b,7)==0)

    # def test_kmeans_dtype():
    #     rnd = np.random.RandomState(0)
    #     X = rnd.normal(size=(40, 2))
    #     X = (X * 10).astype(np.uint8)
    #     km = KMeans(n_init=1).fit(X)
    #     pred_x = assert_warns(RuntimeWarning, km.predict, X)
    #     assert_array_equal(km.labels_, pred_x)

    # @if_not_mac_os()
    # def test_k_means_plus_plus_init_2_jobs():
    #     if _has_blas_lib('openblas'):
    #         raise SkipTest('Multi-process bug with OpenBLAS (see issue #636)')

    #     km = KMeans(init="k-means++", n_clusters=n_clusters, n_jobs=2,
    #                 random_state=42).fit(X)
    #     _check_fitted_model(km)

    # def test_predict():
    #     km = KMeans(n_clusters=n_clusters, random_state=42)

    #     km.fit(X)

    #     # sanity check: predict centroid labels
    #     pred = km.predict(km.cluster_centers_)
    #     assert_array_equal(pred, np.arange(n_clusters))

    #     # sanity check: re-predict labeling for training set samples
    #     pred = km.predict(X)
    #     assert_array_equal(pred, km.labels_)

    #     # re-predict labels for training set using fit_predict
    #     pred = km.fit_predict(X)
    #     assert_array_equal(pred, km.labels_)


    # def test_score():
    #     km1 = KMeans(n_clusters=n_clusters, max_iter=1, random_state=42)
    #     s1 = km1.fit(X).score(X)
    #     km2 = KMeans(n_clusters=n_clusters, max_iter=10, random_state=42)
    #     s2 = km2.fit(X).score(X)
    #     assert_greater(s2, s1)

    # def test_input_dtypes():
    #     X_list = [[0, 0], [10, 10], [12, 9], [-1, 1], [2, 0], [8, 10]]
    #     X_int = np.array(X_list, dtype=np.int32)
    #     X_int_csr = sp.csr_matrix(X_int)
    #     init_int = X_int[:2]

    #     fitted_models = [
    #         KMeans(n_clusters=2).fit(X_list),
    #         KMeans(n_clusters=2).fit(X_int),
    #         KMeans(n_clusters=2, init=init_int, n_init=1).fit(X_list),
    #         KMeans(n_clusters=2, init=init_int, n_init=1).fit(X_int),
    #         # mini batch kmeans is very unstable on such a small dataset hence
    #         # we use many inits
    #         MiniBatchKMeans(n_clusters=2, n_init=10, batch_size=2).fit(X_list),
    #         MiniBatchKMeans(n_clusters=2, n_init=10, batch_size=2).fit(X_int),
    #         MiniBatchKMeans(n_clusters=2, n_init=10, batch_size=2).fit(X_int_csr),
    #         MiniBatchKMeans(n_clusters=2, batch_size=2,
    #                         init=init_int, n_init=1).fit(X_list),
    #         MiniBatchKMeans(n_clusters=2, batch_size=2,
    #                         init=init_int, n_init=1).fit(X_int),
    #         MiniBatchKMeans(n_clusters=2, batch_size=2,
    #                         init=init_int, n_init=1).fit(X_int_csr),
    #     ]
    #     expected_labels = [0, 1, 1, 0, 0, 1]
    #     scores = np.array([v_measure_score(expected_labels, km.labels_)
    #                     for km in fitted_models])
    #     assert_array_equal(scores, np.ones(scores.shape[0]))


    # def test_transform():
    #     km = KMeans(n_clusters=n_clusters)
    #     km.fit(X)
    #     X_new = km.transform(km.cluster_centers_)

    #     for c in range(n_clusters):
    #         assert_equal(X_new[c, c], 0)
    #         for c2 in range(n_clusters):
    #             if c != c2:
    #                 assert_greater(X_new[c, c2], 0)


    # def test_fit_transform():
    #     X1 = KMeans(n_clusters=3, random_state=51).fit(X).transform(X)
    #     X2 = KMeans(n_clusters=3, random_state=51).fit_transform(X)
    #     assert_array_equal(X1, X2)

    # def test_k_means_function():
    #     # test calling the k_means function directly
    #     # catch output
    #     old_stdout = sys.stdout
    #     sys.stdout = StringIO()
    #     try:
    #         cluster_centers, labels, inertia = k_means(X, n_clusters=n_clusters,
    #                                                    verbose=True)
    #     finally:
    #         sys.stdout = old_stdout
    #     centers = cluster_centers
    #     assert_equal(centers.shape, (n_clusters, n_features))

    #     labels = labels
    #     assert_equal(np.unique(labels).shape[0], n_clusters)

    #     # check that the labels assignment are perfect (up to a permutation)
    #     assert_equal(v_measure_score(true_labels, labels), 1.0)
    #     assert_greater(inertia, 0.0)

    #     # check warning when centers are passed
    #     assert_warns(RuntimeWarning, k_means, X, n_clusters=n_clusters,
    #                  init=centers)

    #     # to many clusters desired
    #     assert_raises(ValueError, k_means, X, n_clusters=X.shape[0] + 1)
def main():
    unittest.main()

if __name__ == '__main__':
    main()
