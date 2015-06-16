"""
The :mod:`codebook` module includes utilities to create a Bag-of-(visual)Fetures,
that takes advantage of :mod:`sklearn`.
"""

# Authors: Joan Massich
#          Guillaume Lemaitre
#
# License: MIT

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

# import warnings

# import scipy.sparse as sp

# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.utils.extmath import row_norms, squared_norm
# from sklearn.utils.sparsefuncs_fast import assign_rows_csr
# from sklearn.utils.sparsefuncs import mean_variance_axis
# from sklearn.utils.fixes import astype
# from sklearn.utils import check_array
# from sklearn.utils import check_random_state
# from sklearn.utils import as_float_array
# from sklearn.utils import gen_batches
# from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.random import choice
# from sklearn.externals.joblib import Parallel
# from sklearn.externals.joblib import delayed


class CodeBook(BaseEstimator, ClusterMixin, TransformerMixin):
    """Code Book creation and manimpulation for Bag-of-(visual)Fetures.

    Parameters
    ----------

    n_words : int, optional, default: 36
        The number of clusters to form as well as the number of
        words (centroids) to generate.

    cluster_core : sklearn.cluster, default: KMeans
        Clustering technique used to quantisize the feature space to
        generate the code book.
        #TODO: its default should be described by _default_clustering()

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.


    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_words > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances

        False : never precompute distances

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    Attributes
    ----------
    cook_book_ : array, [n_words, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Notes
    ------
    The k-means problem is solved using Lloyd's algorithm.

    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    See also
    --------

    dictionary_code
    """
    #TODO: test n_words default = 36
    #TODO: does make sense these paremters: max_iter, n_init
    #TODO: change the cluter_core from cluster_code=None to
    #      cluster_code=_default_cluster(), doing all the apropiated
    #      changes. Check that BaseEstimator asks for strict declaration
    #
    #      def _default_cluster(self, n_words=36,
    #                   init='k-means++', n_init=10, max_iter=300,
    #                   tol=1e-4, precompute_distances='auto',
    #                   verbose=0, random_state=None, copy_x=True, n_jobs=1):
    #          """Default space clustering strategy to determine the code book"""
    #          from sklearn.cluster import KMeans
    #          return KMeans(n_clusters=n_words, ...)
    #
    #       Then self.set_param can also be used to setup the parameters for the
    #       current classification methodology

    def __init__(self, n_words=36, cluster_core=None, init='k-means++',
                 n_init=10, max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1):

        if hasattr(init, '__array__'):
            n_words = init.shape[0]
            init = np.asarray(init, dtype=np.float64)

        self.n_words = n_words
        self.cluster_core_name = cluster_core
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs

        if self.cluster_core_name == 'random-words':
            self.n_init = 1
            self.max_iter = 1
            print 'The number of iterations and try as been fixed to 1.'

        if ( (self.cluster_core_name is     None      ) or 
             (self.cluster_core_name == 'random-words')    ):
            from sklearn.cluster import KMeans
            self.cluster_core = KMeans(n_clusters=self.n_words, init=self.init,
                                       max_iter=self.max_iter, tol=self.tol,
                                       precompute_distances=self.precompute_distances,
                                       n_init=self.n_init, verbose=self.verbose,
                                       random_state=self.random_state,
                                       copy_x=self.copy_x, n_jobs=self.n_jobs)
    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than n_words"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] < self.n_words:
            raise ValueError("n_samples=%d should be >= n_words=%d" % (
                X.shape[0], self.n_words))
        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr')
        n_samples, n_features = X.shape
        expected_n_features = self.cook_book_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))
        if X.dtype.kind != 'f':
            warnings.warn("Got data type %s, converted to float "
                          "to avoid overflows" % X.dtype,
                          RuntimeWarning, stacklevel=2)
            X = X.astype(np.float)

        return X

    def fit(self, X, y=None):
        """Compute the clustering of the space.
        #TODO: right now only for K_means, however a dispatcher is
               needed so that other clustering stragegies are called
               indisticntly

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        self.cluster_core = self.cluster_core.fit(X, y)
        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        #return self.fit(X).labels_
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        # 
        # X = self._check_fit_data(X)
        # return self.fit(X)._transform(X)
        raise NotImplementedError

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        # check_is_fitted(self, 'cook_book_')

        # X = self._check_test_data(X)
        # return self._transform(X)
        raise NotImplementedError

    def _transform(self, X):
        """guts of transform method; no input validation"""
        # return euclidean_distances(X, self.cook_book_)
        raise NotImplementedError


    def predict(self, X):
        """Predicts the index value of the closest word within the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the closest word within the code book.
        """
        return self.cluster_core.predict(X)

    def get_dictionary(self):
        """Retrieves the words forming the code book

        Returns
        -------
        dictionary : array, shape [n_words, n_features]
            Code book elements (words of the dictionary) represented
            in the feature space
        """
        #TODO: check that the coodebook is fitted
        return self.cluster_core.cluster_centers_

    def get_BoF_descriptor(self, X):

        # norm = lambda x: x.astype(float)/np.linalg.norm(x)
        # return norm(np.bincount(self.predict(X)))
        return np.histogram(self.predict(X),
                            bins=range(self.n_words+1),
                            density=True)

    def get_BoF_pramide_descriptor(self, X):
        """ Split the image (or volume) in a piramide manner and get
        a descriptor for each level (and part). Concatenate the output.
        TODO: build proper documentaiton

        """
        def split_data_by2(X):
            # TODO: rewrite this in a nice manner that uses len(X.shape)
            # TODO: this can rise ERROR if length of X is odd
            parts = [np.split(x, 2, axis=2) for x in [np.split(x, 2, axis=1) for x in
             np.slit(X, 2, axis=0) ]]
            return parts

        def get_occurrences(X):
            return np.histogram(X, bins=range(self.n_words+1))

        def build_piramide(X, level=2):
            if level is 0:
                return get_occurrences(X)
            else:
                return [get_occurrences(X)] + [build_piramide(Xpart, level-1)
                       for Xpart in split_data_by2(X)]

        return build_piramide(self.predict(X))

    def get_params(self, deep=True):
        return self.cluster_core.get_params()

    def set_params(self, **params):
        self.cluster_core.set_params(**params)
