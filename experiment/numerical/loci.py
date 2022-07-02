"""
Implementation of the LOCI outlier detection algorithm
Based on the paper:
Papadimitriou, S., Kitagawa, H., Gibbons, P.B. and Faloutsos, C.,
2003, March. Loci: Fast outlier detection using the local correlation integral.
In Data Engineering, 2003. Proceedings. 19th International Conference on (pp. 315-326). IEEE.
"""
import numpy as np

from typing import Union
from scipy.spatial.distance import pdist, squareform


def run_loci(data: np.ndarray, alpha: float = 0.5, k: int = 3,
        method: str = "distance-matrix"):
    """Run the LOCI algorithm on the specified dataset.
    Runs the LOCI algorithm for the specified datset with the specified
    parameters, returns a LOCI object, from which outlier indices can
    be accessed via the outlier_indices property.
    Parameters
    ----------
    data: np.ndarray
        Shape - [number of datapoints, number of dimensions]
    alpha: float, optional
        Default is 0.5 as per the paper. See the paper for full details.
    k: int, optional
        Default is 3 as per the paper. See the paper for full details.
    method: str, optional
        Determines how the distances and neighbourhood counts are computed.
        Currently supports:
            - "distance-matrix", which uses the
                scipy.spatial.distance.pdist function
            - "distance-matrix-binary", same as "distance-matrix", but uses
                a sorted distance matrix & binary search to find the
                neighbourhood counts
    """
    if method == "distance-matrix":
        loci_i = LOCIMatrix(data, alpha, k)
        loci_i.run()
    elif method == "distance-matrix-binary":
        loci_i = LOCIMatrixBinarySearch(data, alpha, k)
        loci_i.run()
    else:
        raise Exception("Invalid method specified.")

    return loci_i


class LOCI(object):
    """Abstract class for the implementation of the LOCI outlier detection
    algorithm. This class should not be instanciated directly.
    Use the run_loci function above to run the algorithm.
    This class should only be used as a parent class and the four
    required methods have to be implemented by the child classes.
    Parameters
    ----------
    data: np.ndarray
    alpha: float, optional
    k: int, optional
    See the loci function for more details on the parameters.
    Attributes
    ----------
    _data: np.ndarray
    _alpha: float
    _k: int
    _max_dist: float
    _n_points: int
    _outlier_indices: np.ndarray
    """

    def __init__(self, data: np.ndarray, alpha: float = 0.5, k: int = 3):
        self._data = data
        self._alpha = alpha
        self._k = k

        self._max_dist = None
        self._n_points = self._data.shape[0]

        self._outlier_indices = None

    @property
    def outlier_indices(self):
        return self._outlier_indices

    def run(self):
        """Executes the LOCI algorithm"""
        self._max_dist = self._get_max_distance(self._data)
        r_max = self._max_dist / self._alpha

        outlier_indices = []
        for p_ix in range(self._n_points):
            critical_values = self._get_critical_values(p_ix, r_max)
            print('Point %d, %d critical values' % (p_ix, len(critical_values)))

            for it, r in enumerate(critical_values):
                n_values = self._get_alpha_n(
                    self._get_sampling_N(p_ix, r), r)

                cur_alpha_n = self._get_alpha_n(p_ix, r)

                n_hat = np.mean(n_values)
                mdef = 1 - (cur_alpha_n / n_hat)
                sigma_mdef = np.std(n_values) / n_hat

                if n_hat >= 20:
                    if mdef > (self._k * sigma_mdef):
                        outlier_indices.append(p_ix)
                        break

        self._outlier_indices = np.array(outlier_indices)
        return True

    def _get_critical_values(self, p_ix: int, r_max: float, r_min: float = 0):
        """Returns the critical and alpha-critical as a
        sorted numpy array for the given point p.
        """
        raise NotImplementedError()

    def _get_sampling_N(self, p_ix: int, r: float):
        """Returns the indices of the sampling neighbourhood of point p
        and radius r.
        """
        raise NotImplementedError()

    def _get_alpha_n(self, indices: Union[int, np.ndarray], r):
        """Returns the counting neighbourhood n (i.e. number of points in
        the counting neighbourhood), for the specified point/s and
        radius r (and instance constant alpha).
        """
        raise NotImplementedError()

    def _get_max_distance(self, data: np.ndarray):
        """Determines the maximum distance between any
        two points in the data set.
        """
        raise NotImplementedError()


class LOCIMatrix(LOCI):
    """Child class of LOCI, implements the required methods using
    a distance matrix computed in the constructor.
    Attributes
    ----------
    _dist_matrix: np.ndarray
        The distance matrix, has shape [n_data_points, n_data_points]
    """

    def __init__(self, data: np.ndarray, alpha: float = 0.5, k: int = 3):
        super().__init__(data, alpha, k)

        self._dist_matrix = squareform(pdist(self._data, metric="euclidean"))
        self._sorted_dist_matrix = np.sort(self._dist_matrix, axis=1)

    def _get_critical_values(self, p_ix: int, r_max: float, r_min: float = 0):
        distances = self._dist_matrix[p_ix, :]
        mask = (r_min < distances) & (distances <= r_max)

        return np.unique(np.sort(np.concatenate((distances[mask], distances[mask] / self._alpha))))

    def _get_sampling_N(self, p_ix: int, r: float):
        p_distances = self._dist_matrix[p_ix, :]

        return np.nonzero(p_distances <= r)[0]

    def _get_alpha_n(self, indices: Union[int, np.ndarray], r):
        if type(indices) is int:
            return np.count_nonzero(
                self._dist_matrix[indices, :] < (r * self._alpha))
        else:
            return np.count_nonzero(
                self._dist_matrix[indices, :] < (r * self._alpha), axis=1)

    def _get_max_distance(self, data: np.ndarray):
        return self._dist_matrix.max()


class LOCIMatrixBinarySearch(LOCIMatrix):
    """Child class of LOCIMatrix, uses a sorted distance matrix and
    binary search to get neighbourhood counts.
    Attributes
    ----------
    _sorted_dist_matrix: np.ndarray
        The distance matrix, sorted along the rows (i.e. the rows are sorted),
        same shape as _dist_matrix
    """

    def __init__(self, data: np.ndarray, alpha: float = 0.5, k: int = 3):
        super().__init__(data, alpha, k)

        self._sorted_dist_matrix = np.sort(self._dist_matrix, axis=1)

    def _get_alpha_n(self, indices: Union[int, np.ndarray], r):
        indices = [indices] if type(indices) is int else indices

        result = [np.searchsorted(self._sorted_dist_matrix[p_ix, :],
                  [r * self._alpha], side="left")[0] + 1 for p_ix in indices]

        return np.array(result)


from model.german_rf import get_model
import sys
sys.path.append('../..')
from lib.tree_extractor import path_extractor
clf, (X_train, y_train, X_test, y_test, data_table), dataset, model, parameters = get_model()
paths = path_extractor(clf, 'random forest', (X_train, y_train))

target = 'credit_risk'
X = data_table.drop(target, axis=1).values
y = data_table[target].values
from lib.tree_extractor import assign_samples
assign_samples(paths, (X, y))

features = data_table.columns[1:]
new_feature = {}
feature_pos = {}
for index, feature in enumerate(features):
    if ' - ' in feature:
        name, p = feature.split(' - ')
        p = int(p)
        if name not in new_feature:
            new_feature[name] = []
        while p >= len(new_feature[name]):
            new_feature[name].append(-1)
        new_feature[name][p] = index
    else:
        new_feature[feature] = [index]

feature_range = {}
for key in new_feature:
    if key in data_table.columns:
        feature_range[key] = [data_table[key].min(), data_table[key].max() + 1]
    else:
        feature_range[key] = [0, len(new_feature[key])]
    for i, j in enumerate(new_feature[key]):
        feature_pos[j] = (key, i)

for index, path in enumerate(paths):
    path['index'] = index

paths = [path for path in paths if np.sum(path['sample']) > 0]

mat = np.array([p['sample'] for p in paths]).astype('float')

for i, path in enumerate(paths):
    sum = np.sqrt(np.sum(mat[i]))
    if sum > 0:
        mat[i] /= sum

loci_res = run_loci(mat)
outlier_indices = loci_res.outlier_indices

