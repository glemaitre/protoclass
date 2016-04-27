"""Standard time normalization to normalize temporal modality."""

from scipy.ndimage.filters import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from skimage import img_as_float

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

    """

    def __init__(self, base_modality):
        super(StandardTimeNormalization, self).__init__(base_modality)
        # Initialize the fitting boolean
        self.is_fitted_ = False

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
            {'std' : 50., 'exp' : 25., 'alpha' : .9}
            - If dict, a dictionary with the keys 'std', 'exp',and 'alpha'
            should be specified. The corresponding value of these parameters
            should be float. They will be the initial value during fitting.

        verbose : bool, optional (default=True)
            Whether to show the fitting process information.

        Returns
        -------
        self : object
             Return self.

        """
        # By calling the parents function, self.roi_data_ will be affected
        # and can be used in all cases.
        super(StandardTimeNormalization, self).fit(modality=modality)

                # Check the gaussian parameters argument
        if isinstance(params, basestring):
            if params == 'default':
                # Give the default values for each parameter
                self.params_ = {'std' : 50., 'exp' : 25., 'alpha' : .9}
            else:
                raise ValueError('The string for the object params is'
                                 ' unknown.')
        elif isinstance(params, dict):
            # Check that mu and sigma are inside the dictionary
            valid_presets = ('std', 'exp', 'alpha')
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
                        raise ValueError('The parameter std, exp, or alpha'
                                         ' should be some float.')
                else:
                    raise ValueError('Unknown parameter inside the dictionary.'
                                     ' `std`, `exp`, and `alpha` are the'
                                     ' only two solutions and need to be'
                                     ' float.')
        else:
            raise ValueError('The type of the object params does not fulfill'
                             ' any requirement.')

        # Compute the heatmap
        heatmap, bins_heatmap = modality.build_heatmap(self.roi_data_)
        # Smooth the heatmap using a Gaussian filter
        heatmap = gaussian_filter1d(heatmap, self.params_['std'], axis=1)
        # Inverse the map such that we can take the shortest path
        heatmap = 1. - (heatmap / np.ndarray.max(heatmap))
        # Compute the exponential map to emplify the differences
        heatmap = img_as_float(np.exp(self.params_['exp'] * heatmap))

        if verbose:
            print 'Heatmap built ...'

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
            graph[y, p_1] = self.params_['alpha'] * edge_val
            graph[y, p_2] = (1. - self.params_['alpha']) * edge_val

        graph = coo_matrix(graph)

        if verbose:
            print 'Graph built ...'

        # Find the starting and ending point in the graph - the median is used
        # Get data from the first serie
        data_serie = modality.data_[0, :, :, :]
        # Compute the median from the given roi
        median_start = np.median(data_serie[self.roi_data_].reshape(-1))
        # Find the nearest index associated with this value
        _, idx_start = find_nearest(bins_heatmap, median_start)

        # Get data from the last serie
        data_serie = modality.data_[-1, :, :, :]
        # Compute the median from the given roi
        median_end = np.median(data_serie[self.roi_data_].reshape(-1))
        # Find the nearest index associated with this value
        _, idx_end = find_nearest(bins_heatmap, median_end)

        # Create the starting and ending tuple
        start_tuple = (0, idx_start)
        end_tuple = (modality.n_serie_ - 1, idx_end)

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

        # Convert the list into array
        path_list = np.array(path_list)

        # Clean the path serie by serie to have a unique value
        cleaned_path = []
        for t_serie in range(modality.n_serie_ - 1):
            # Find all the intensities corresponding to this serie
            poi = path_list[np.nonzero(path_list[:, 0] == t_serie)]

            # Compute the median of the second column
            med_path = np.median(poi[:, 1])
            cleaned_path.append([t_serie, med_path])

        # Convert list to array
        cleaned_path = np.round(np.array(cleaned_path))

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

