"""Standard time normalization to normalize temporal modality."""

import numpy as np
from numpy.matlib import repmat

from scipy.ndimage.filters import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from skimage import img_as_float
from skimage.graph import route_through_array

from .temporal_normalization import TemporalNormalization

from ..utils import find_nearest

class StandardTimeNormalization(TemporalNormalization):
    """Standard normalization to normalize temporal modality.

    Parameters
    ----------
    base_modality : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    Attributes
    ----------
    base_modality_ : object
        The base modality on which the normalization will be applied. The base
        modality should inherate from TemporalModality class.

    fit_params_ : dict of str : float
        There is the following keys:

        - 'shift' is the shift found thorugh graph walking.

    """

    def __init__(self, base_modality):
        super(StandardTimeNormalization, self).__init__(base_modality)
        # Initialize the fitting boolean
        self.is_fitted_ = False

    @staticmethod
    def _build_graph(heatmap, param_alpha, verbose=True):
        """Build a graph representation from the heatmap.

        Parameters
        ----------
        heatmap : ndarray, shape (n_serie, nb_bins)
            The heatmap from which the graph will be built.

        param_alpha : float
            Weight for the vertical and horizontal vertices.

        verbose : bool
            Display the completion or not.

        Returns
        -------
        graph : ndarray, shape (n_serie * nb_bins, n_serie * nb_bins)
            Fully connected graph computed from the heatmap.

        """

        # We can build the graph associated with the heatmap
        graph = np.empty((heatmap.size, heatmap.size), dtype=float)
        for y in np.arange(graph.shape[0]):
            # Come back to the pixel index
            px_idx = np.unravel_index(y, heatmap.shape)
            if (px_idx[0] >= (heatmap.shape[0] - 1) or
                px_idx[1] >= (heatmap.shape[1] - 1)):
                continue
            # Get the pixel value
            edge_val = heatmap[px_idx]
            # Assign the verteces
            # Find the position of the verteces inside the graph
            p_1 = np.ravel_multi_index((px_idx[0] + 1, px_idx[1]),
                                       heatmap.shape)
            p_2 = np.ravel_multi_index((px_idx[0], px_idx[1] + 1),
                                       heatmap.shape)
            # Assign the edge value
            graph[y, p_1] = param_alpha * edge_val
            graph[y, p_2] = (1. - param_alpha) * edge_val

        graph = coo_matrix(graph)

        if verbose:
            print 'Graph built ...'

        return graph

    @staticmethod
    def _walk_through_graph(graph, heatmap, start_end_vertices, method='shortest-path', verbose=True):
        """Find path through the path with different a-priori.

        Parameters
        ----------
        graph : ndarray, shape (n_serie * nb_bins, n_serie * nb_bins)
            Fully connected graph computed from the heatmap.

        heatmap : ndarray, shape (n_serie, nb_bins)
            The heatmap from which the graph will be built.

        start_end_vertices : tuple, shape ((start_vertice, end_vertice))
            Tuple containing the starting and ending vertices to enter and exit
            the graph.

        method : str, optional (default='shortest-path')
            Method to use to walk through the graph. The following
            possibilities:
            - 'shortest-path' : find shortest-path using scipy implementation.
            - 'route-through-graph' : find the path using MCP algorithm from
                skimage.

        verbose : bool
            Show the processing stage.

        Returns
        -------
        path : ndarray, shape (n_serie, 2)
            The path found to walk through the graph.

        """
        # Split the starting and ending vertices
        start_tuple = start_end_vertices[0]
        end_tuple = start_end_vertices[1]

        if method == 'shortest-path':
            # Define a function to go from px to vertices
            def px2v(px, im_shape):
                return np.ravel_multi_index(px, im_shape)

            # Define a function to go from vertices to px
            def v2px(v, im_shape):
                return np.unravel_index(v, im_shape)

            # Compute the shortest path for the whole graph
            d, p = shortest_path(graph, return_predecessors=True)
            # Initialize the path
            path_list = [end_tuple]
            # Find the shortest path thorugh an iterative process
            while end_tuple != start_tuple:
                # Convert coord from px to v
                s_v = px2v(start_tuple, heatmap.shape)
                e_v = px2v(end_tuple, heatmap.shape)

                # Find the predecessor
                pred_v = p[s_v, e_v]

                # Convert into pixel
                pred_px = v2px(pred_v, heatmap.shape)
                path_list.append(pred_px)

                # Update the last point of the path
                end_tuple = pred_px
        elif method == 'route-through-graph':
            # Call the function from skimage
            indices, weight = route_through_array(heatmap,
                                                  start_tuple,
                                                  end_tuple, 
                                                  geometric=True)
            path_list = np.array(indices)
        else:
            raise NotImplementedError

        # Convert the list into array
        path_list = np.array(path_list)

        # Clean the path serie by serie to have a unique value
        cleaned_path = []
        for t_serie in range(heatmap.shape[0]):
            # Find all the intensities corresponding to this serie
            poi = path_list[np.nonzero(path_list[:, 0] == t_serie)]

            # Compute the median of the second column
            med_path = np.median(poi[:, 1])
            cleaned_path.append([t_serie, med_path])

        if verbose:
            print 'Walked through the graph ...'

        # Convert list to array
        return np.round(np.array(cleaned_path))

    @staticmethod
    def _shift_heatmap(heatmap, shift):
        """Roll the heatmap depending on a given shift.

        Parameters
        ----------
        heatmap : ndarray, shape (n_series, nb_bins)
            The heatmap to be shifted.

        shift : ndarray, shape (n_series, )
            The shift to apply.

        Returns
        -------
        heatmap_shifted : ndarray, shape (n_series, nb_bins)
            The shifted heatmap.

        """
        # Check that we have the same number of series
        if heatmap.shape[0] != shift.shape[0]:
            raise ValueError('Inconsitent size for the data.')

        # Find the index where to align every sequence
        middle_idx = int(heatmap.shape[1] / 2.)
        # Shift the heatmap accordling
        heatmap_shifted = heatmap.copy()
        for idx_serie, heatmap_serie in enumerate(heatmap):
            # Define the shift to perform
            shift_rel = int(middle_idx - shift[idx_serie])
            heatmap_shifted[idx_serie] = np.roll(heatmap_serie, shift_rel)

        return heatmap_shifted

    def _find_shift(self, heatmap, bins_heatmap, param_std, param_exp, param_alpha, param_max_iter, verbose=True):
        """Find the shift through iterative shortest-path.

        Parameters
        ----------
        heatmap : ndarray, shape (n_serie, nb_bins)
            Heatmap from the DCE modality.

        bins_heatmap : ndarray, shape (nb_bins, )
            Associated bins with the heatmap.

        param_std : float
            Parameter for the Gaussian filter.

        param_exp : float
            Parameter to compute the exponential heatmap.

        param_alpha : float
            Parameter for vertices weight of the graph.

        param_max_iter : int
            Parameter for the max number of iteration to walk through
            the graph.

        Returns
        -------
        shift : ndarray, shape (n_serie, )
            The vector of intensity from which each serie has to be shifted.

        idx_shift : ndarray, shape (n_serie, )
            Vector with the corresponding index in `bins_heatmap`.

        """
        # Smooth the heatmap using a Gaussian filter
        heatmap_inv_exp = gaussian_filter1d(heatmap,
                                            param_std, axis=1)
        # Inverse the map such that we can take the shortest path
        heatmap_inv_exp = 1. - (heatmap_inv_exp /
                                np.ndarray.max(heatmap_inv_exp))
        # Compute the exponential map to emplify the differences
        heatmap_inv_exp = img_as_float(np.exp(param_exp *
                                              heatmap_inv_exp))

        if verbose:
            print 'Heatmap built ...'

        # Initialization of the output
        shift = np.zeros(heatmap.shape[0], )

        graph = self._build_graph(heatmap_inv_exp, param_alpha, verbose)

        # Find the starting and ending point in the graph - the median is used
        # The median can be estimated from the heatmap
        # Compute the normalize cumulative sum of the first serie
        cumsum_hist = np.cumsum(heatmap[0, :])
        cumsum_hist /= cumsum_hist[-1]
        # Find the median
        idx_start = np.argmax(cumsum_hist > .5)

        # Compute the normalize cumulative sum of the last serie
        cumsum_hist = np.cumsum(heatmap[-1, :])
        cumsum_hist /= cumsum_hist[-1]
        # Find the median
        idx_end = np.argmax(cumsum_hist > .5)

        # Create the starting and ending tuple
        start_tuple = (0, idx_start)
        end_tuple = (heatmap.shape[0] - 1, idx_end)
        start_end_tuple = (start_tuple, end_tuple)

        # Compute the shortest path
        method = 'shortest-path'
        absolute_shift = self._walk_through_graph(graph, heatmap_inv_exp, 
                                                  start_end_tuple, 
                                                  method, verbose)
        # Keep only the uselful information of the shift
        absolute_shift = np.ravel(absolute_shift[:, 1])

        # do-while loop
        middle_idx = 0
        itr_loop = 1
        while True:
            if verbose:
                print 'Iteration #{}'.format(itr_loop)
            # Increment the shifting of the previous iteration
            shift += (absolute_shift - middle_idx)
            # Shift the heatmap accordingly
            heatmap_inv_exp = self._shift_heatmap(heatmap_inv_exp,
                                                  absolute_shift)
            # Apply again the shortest path
            graph = self._build_graph(heatmap_inv_exp, param_alpha, verbose)

            # Create the starting and ending tuple
            middle_idx = int(heatmap_inv_exp.shape[1] / 2.)
            start_tuple = (0, middle_idx)
            end_tuple = (heatmap.shape[0] - 1, middle_idx)
            start_end_tuple = (start_tuple, end_tuple)

            # Compute the shortest path
            method = 'route-through-graph'
            absolute_shift = self._walk_through_graph(graph, heatmap_inv_exp, 
                                                      start_end_tuple, 
                                                      method, verbose)
            # Keep only the uselful information of the shift
            absolute_shift = np.ravel(absolute_shift[:, 1])

            # Keep track of the iteration
            itr_loop += 1
            # Breaking condition - We don't move
            if (np.sum(absolute_shift - middle_idx) == 0 or 
                itr_loop > param_max_iter):
                break

        # Store the shift in the dictionary
        # Take the intensity value, not the bin
        return bins_heatmap[shift.astype(int)], shift.astype(int)

    def _compute_rmse(self, heatmap, bins_heatmap, shift, idx_shift, verbose=True):
        """Compute the root mean squared error from the shift estimator.

        Parameters
        ----------
        heatmap : ndarray, shape (n_serie, nb_bins)
            Heatmap from the DCE modality.

        bins_heatmap : ndarray, shape (nb_bins, )
            Associated bins with the heatmap.

        shift : ndarray, shape (n_serie, )
            The vector of intensity from which each serie has to be shifted.

        idx_shift : ndarray, shape (n_serie, )
            Vector with the corresponding index in `bins_heatmap`.

        Returns
        -------
        rmse_estimator : ndarray, shape (n_serie, )
            The vector of intensity from which each serie has to be shifted.

        """
        # Shift the original heatmap
        heatmap = self._shift_heatmap(heatmap, idx_shift)
        # Shift the bins accordingly
        bins_heatmap_serie = repmat(bins_heatmap, heatmap.shape[0], 1)
        bins_heatmap_serie = self._shift_heatmap(bins_heatmap_serie,
                                                 idx_shift)

        # Compute the RMSE for each serie
        rmse_estimator = []
        for idx_serie in range(heatmap.shape[0]):
            # Compute sum i^2 p(i)
            # + .5 since we consider the center of each bins
            # The zero is the center of the histogram, shift the zero
            i_array = (bins_heatmap_serie[idx_serie] - shift[idx_serie])
            i2pi_array = (i_array ** 2) * heatmap[idx_serie, :]
            rmse_estimator.append(np.sqrt(np.sum(i2pi_array)))

        return rmse_estimator

    def fit(self, modality, ground_truth=None, cat=None, params='default', verbose=True):
        """Find the parameters needed to apply the normalization.

        Parameters
        ----------
        modality : object
            Object inherated from TemporalModality.

        ground-truth : object of type GTModality or None, optional
            (default=None)
            The ground-truth of GTModality. If None, the whole data will be
            considered.

        cat : str or None, optional (default=None)
            String corresponding at the ground-truth of interest. Cannot be
            None if ground-truth is not None.

        params : str or dict of str: float, optional (default='default')
            The initial estimation of the parameters:

            - If 'default', the value following value will be affected:
            {'std' : 50., 'exp' : 25., 'alpha' : .9, 'max_iter' : 5}
            - If dict, a dictionary with the keys 'std', 'exp', 'alpha', 
            and 'max_iter' should be specified. The corresponding value of
            these parameters should be float. They will be the initial value
            during fitting.

            'std' corresponds to the standard deviation when applying the 
            Gaussian filter; 'exp' corresponds to the factor in the
            exponential; 'alpha' corresponds to the parameters to penalize
            vertical and horizontal weight in the graph; 'max_iter' is the
            maximum number of walks through the graph.

        verbose : bool, optional (default=True)
            Whether to show the fitting process information.

        Returns
        -------
        self : object
             Return self.

        """
        # By calling the parents function, self.roi_data_ will be affected
        # and can be used in all cases.
        super(StandardTimeNormalization, self).fit(modality=modality,
                                                   ground_truth=ground_truth,
                                                   cat=cat)

        # Check the gaussian parameters argument
        if isinstance(params, basestring):
            if params == 'default':
                # Give the default values for each parameter
                self.params_ = {'std' : 50., 'exp' : 25.,
                                'alpha' : .9, 'max_iter' : 5}
            else:
                raise ValueError('The string for the object params is'
                                 ' unknown.')
        elif isinstance(params, dict):
            # Check that mu and sigma are inside the dictionary
            valid_presets = ('std', 'exp', 'alpha', 'max_iter')
            for val_param in valid_presets:
                if val_param not in params.keys():
                    raise ValueError('At least the parameter {} is not specify'
                                      ' in the dictionary.'.format(val_param))
            # For each key, check if this is a known parameters
            self.params_ = {}
            for k_param in params.keys():
                if k_param in valid_presets:
                    # The key is valid, build our dictionary
                    if isinstance(params[k_param], float):
                        self.params_[k_param] = params[k_param]
                    else:
                        raise ValueError('The parameter std, exp, alpha, or'
                                         ' max_iter should be some float.')
                else:
                    raise ValueError('Unknown parameter inside the dictionary.'
                                     ' `std`, `exp`, `alpha`, and `max_iter`'
                                     ' are the only two solutions and need to'
                                     ' be float.')
        else:
            raise ValueError('The type of the object params does not fulfill'
                             ' any requirement.')

        # Initialize the parameter dictionary
        self.fit_params_ = {}

        # Compute the heatmap
        heatmap, bins_heatmap = modality.build_heatmap(self.roi_data_)
        
        # Find the shift in the data
        self.fit_params_['shift'], self.shift_idx_ = self._find_shift(heatmap,
                                                                      bins_heatmap,
                                                                      self.params_['std'],
                                                                      self.params_['exp'],
                                                                      self.params_['alpha'],
                                                                      self.params_['max_iter'],
                                                                      verbose)

        # Compute the associated RMSE of the estimator found
        self.rmse = self._compute_rmse(heatmap, bins_heatmap,
                                       self.fit_params_['shift'],
                                       self.shift_idx_,
                                       verbose)

        # Fitting performed
        self.is_fitted_ = True

        return self

    def normalize(self, modality):
        """Normalize the given modality using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.

        """
        super(StandardTimeNormalization, self).normalize(modality=modality)

        return self

    def denormalize(self, modality):
        """Denormalize the given modality using the fitted parameters.

        Parameters
        ----------
        modality: object of type StandaloneModality
            The modality object from which the data need to be normalized.

        Returns
        -------
        modality: object of type StandaloneModality
            The modality object in which the data will be normalized.

        """
        super(StandardTimeNormalization, self).denormalize(modality=modality)

        return self

