from sklearn.metrics import pairwise_distances
import numpy as np
import math

from typing import Union
from scipy.spatial.distance import pdist, squareform

def run_loci(data: np.ndarray, alpha: float = 0.5, k: int = 3):
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
    """
    loci_i = ExtendLOCI(data, alpha, k)
    loci_i.run()
    return loci_i

class ExtendLOCI():
    """
    data: np.ndarray
    alpha: float, optional
    k: int, optional
    See the loci function for more details on the parameters.
    Attributes
    ----------
    data: np.ndarray
    alpha: float
    k: int
    max_dist: float
    n_points: int
    outlier_indices: np.ndarray
    dist_matrix: np.ndarray
        The distance matrix, has shape [n_data_points, n_data_points]

    The minimum sampling radius r determined based on the number of objects in the sampling neighborhood.
    We always use a smallest sampling neighborhood with nmin = 20 neighbors;
    in practice, this is small enough but not too small to introduce statistical errors in MDEF and σ MDEF values.
    """

    def __init__(self, data: np.ndarray, alpha: float = 0.5, k: int = 3):
        self.data = data
        self.alpha = alpha
        self.k = k

        self.max_dist = None
        self.n_points = self.data.shape[0]
        self.outlier_indices = None
        self.dist_matrix = pairwise_distances(X = self.data, metric='cosine')
        self.sorted_neighbors = np.argsort(self.dist_matrix, axis=1)
        self.sorted_dist = np.sort(self.dist_matrix, axis=1)
        self.outer_ptr = np.zeros(self.n_points).astype(int)
        self.inner_ptr = np.zeros(self.n_points).astype(int)

    def update_outer_pointer(self, r):
        ptr = self.outer_ptr
        for i in range(self.n_points):
            while ptr[i] < self.n_points and self.sorted_dist[i, ptr[i]] <= r:
                ptr[i] += 1

    def update_inner_pointer(self, r):
        ptr = self.inner_ptr
        for i in range(self.n_points):
            while ptr[i] < self.n_points and self.sorted_dist[i, ptr[i]] <= r:
                ptr[i] += 1

    def run(self):
        """Executes the LOCI algorithm"""
        self.result = []
        for p_ix in range(self.n_points):
            self.result.append({
                'records': [],
            })
        sqrt_n = int(math.sqrt(self.n_points))
        quarter_sqrt_n = int(sqrt_n // 4)
        alpha = self.alpha
        r_max = self.sorted_dist[:, -1].max() / alpha
        r_min = self.sorted_dist[:, quarter_sqrt_n].min()
        print('alpha', alpha)
        self.rs = []
        self.scores = [[] for _ in range(self.n_points)]

        outlier_indices = []
        outlier_score = np.zeros(self.n_points)
        n_steps = 100
        for i in range(1, n_steps):
            r = i / n_steps * (r_max - r_min) + r_min
            self.rs.append(r)
            self.update_outer_pointer(r)
            self.update_inner_pointer(alpha * r)
            for p_ix in range(self.n_points):
                neighbors = self._get_sampling_N(p_ix)
                n_values = self._get_alpha_n(neighbors)
                cur_alpha_n = self._get_alpha_n(p_ix)

                n_hat = np.mean(n_values)
                mdef = 1 - (cur_alpha_n / n_hat)
                sigma_mdef = np.std(n_values) / n_hat
                self.result[p_ix]['records'].append((r, mdef, sigma_mdef))

                score = 0
                if cur_alpha_n >= 5 and sigma_mdef > 0:
                    score = mdef / sigma_mdef
                    if score > outlier_score[p_ix]:
                        outlier_score[p_ix] = score
                    if mdef > (self.k * sigma_mdef):
                        if p_ix not in outlier_indices:
                            outlier_indices.append(p_ix)
                self.scores[p_ix].append(score)

        self.outlier_indices = np.array(outlier_indices)
        self.outlier_score = outlier_score
        self.scores = np.array(self.scores)
        return True

    def _get_sampling_N(self, p_ix: int):
        return self.sorted_neighbors[p_ix][:self.outer_ptr[p_ix]]

    def _get_alpha_n(self, indices: Union[int, np.ndarray]):
        return self.inner_ptr[indices]