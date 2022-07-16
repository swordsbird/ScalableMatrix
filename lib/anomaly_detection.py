
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import MinCovDet
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


from sklearn.metrics import pairwise_distances

import numpy as np
import math

from typing import Union

class LOCIMatrixNew():
    ''' the inner radius is fixed, and the outer radius is variable. '''
    def __init__(self, data: np.ndarray, output = None, output_alpha = 0.5, r = 10, metric = 'euclidean', n_ticks = 100):
        self.data = data
        self.r = r
        self.n_points = self.data.shape[0]
        self.indice = None
        self.dist_matrix = pairwise_distances(X = self.data, metric=metric)
        if output is not None:
            output_dist = np.array([output[i] != output for i in range(self.n_points)])
            max_dist = self.dist_matrix.max()
            self.dist_matrix /= max_dist
            self.dist_matrix = self.dist_matrix * (1 - output_alpha) + output_dist * output_alpha
        self.sorted_neighbors = np.argsort(self.dist_matrix, axis=1)
        self.sorted_dist = np.sort(self.dist_matrix, axis=1)
        self.outer_ptr = np.zeros(self.n_points).astype(int)
        self.inner_ptr = np.zeros(self.n_points).astype(int)
        self.n_ticks = n_ticks

    def update_outer_pointer(self, r, fast = False):
        ptr = self.outer_ptr
        for i in range(self.n_points):
            if fast:
                ptr[i] = np.searchsorted(self.sorted_dist[i], r, side='right')
            else:
                while ptr[i] < self.n_points and self.sorted_dist[i, ptr[i]] <= r:
                    ptr[i] += 1

    def update_inner_pointer(self, r, fast = False):
        ptr = self.inner_ptr
        for i in range(self.n_points):
            while ptr[i] < self.n_points and self.sorted_dist[i, ptr[i]] <= r:
                ptr[i] += 1

    def check_consistency(self, r, outputs):
        ret = []
        self.outer_ptr = np.zeros(self.n_points).astype(int)
        self.update_outer_pointer(r, fast=True)
        outputs = np.array(outputs)
        for p_ix in range(self.n_points):
            neighbors = self._get_sampling_N(p_ix)
            total = len(neighbors)
            count = np.sum(outputs[neighbors] == outputs[p_ix])
            ret.append(1.0 * count / total)
        return ret

    def run(self):
        """Executes the LOCI algorithm"""
        r_max = self.sorted_dist.max()
        r_min = self.r
        self.rs = []
        self.scores = [[] for _ in range(self.n_points)]
        print('r range: %.3f - %.3f' % (r_min, r_max))

        for i in range(self.n_ticks):
            r = i / self.n_ticks * (r_max - r_min) + r_min
            self.rs.append(r)
            self.update_outer_pointer(self.r)
            self.update_inner_pointer(r)
            for p_ix in range(self.n_points):
                neighbors = self._get_sampling_N(p_ix)
                n_values = self._get_alpha_n(neighbors)
                cur_alpha_n = self._get_alpha_n(p_ix)

                n_hat = np.mean(n_values)
                mdef = 1 - (cur_alpha_n / n_hat)
                sigma_mdef = np.std(n_values) / n_hat
                score = 0
                if len(neighbors) >= 20:
                    if sigma_mdef > 0:
                        score = mdef / sigma_mdef
                self.scores[p_ix].append(score)

        self.indice = np.array([self.n_ticks // 10 for _ in range(self.n_points)])
        self.scores = np.array(self.scores)
        self.outlier_score = np.array([self.scores[i, self.indice[i]] for i in range(self.n_points)])
        return True

    def _get_sampling_N(self, p_ix: int):
        return self.sorted_neighbors[p_ix][:self.outer_ptr[p_ix]]

    def _get_alpha_n(self, indices: Union[int, np.ndarray]):
        return self.inner_ptr[indices]

class LOCIMatrix():
    def __init__(self, data: np.ndarray, alpha: float = 0.5, metric = 'euclidean', n_ticks = 100):
        self.data = data
        self.alpha = alpha

        self.n_points = self.data.shape[0]
        self.indice = None
        self.dist_matrix = pairwise_distances(X = self.data, metric=metric)
        self.sorted_neighbors = np.argsort(self.dist_matrix, axis=1)
        self.sorted_dist = np.sort(self.dist_matrix, axis=1)
        self.outer_ptr = np.zeros(self.n_points).astype(int)
        self.inner_ptr = np.zeros(self.n_points).astype(int)
        self.n_ticks = n_ticks

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
        r_max = self.sorted_dist[:, sqrt_n].max() / self.alpha
        r_min = self.sorted_dist[:, 10].min()
        self.rs = []
        self.scores = [[] for _ in range(self.n_points)]
        print('r range: %.3f - %.3f, alpha: %.3f' % (r_min, r_max, self.alpha))

        for i in range(self.n_ticks):
            r = i / self.n_ticks * (r_max - r_min) + r_min
            self.rs.append(r)
            self.update_outer_pointer(r)
            self.update_inner_pointer(self.alpha * r)
            for p_ix in range(self.n_points):
                neighbors = self._get_sampling_N(p_ix)
                n_values = self._get_alpha_n(neighbors)
                cur_alpha_n = self._get_alpha_n(p_ix)

                n_hat = np.mean(n_values)
                mdef = 1 - (cur_alpha_n / n_hat)
                sigma_mdef = np.std(n_values) / n_hat
                self.result[p_ix]['records'].append((r, mdef, sigma_mdef))

                score = 0
                if len(neighbors) >= 20:
                    if sigma_mdef > 0:
                        score = mdef / sigma_mdef
                self.scores[p_ix].append(score)

        step = (r_max - r_min) / self.n_ticks
        r = self.sorted_dist[:, int(math.sqrt(self.n_points))].mean()
        self.indice = np.array([int((self.sorted_dist[i, sqrt_n] - r_min) / step) for i in range(self.n_points)])
        self.min_indice = np.array([np.argmin(self.scores[i]) for i in range(self.n_points)])
        self.max_indice = np.array([np.argmax(self.scores[i]) for i in range(self.n_points)])
        self.scores = np.array(self.scores)
        self.outlier_score = np.array([self.scores[i, self.indice[i]] for i in range(self.n_points)])
        return True

    def select_indice(self, target):
        target /= self.alpha
        for i, r in enumerate(self.rs):
            if r > target:
                self.indice = np.array([i for _ in range(self.n_points)])
                self.outlier_score = np.array([self.scores[i, self.indice[i]] for i in range(self.n_points)])
                break

    def label_propagation(self, x, label, thres = 0.10):
        conf = {}
        visit = {}
        conf[x] = 1
        Q = [x]
        head = 0
        while head < len(Q):
            x = Q[head]
            head += 1
            if x in visit:
                continue
            visit[x] = 1
            
            if label == 0:
                new_indice = self.min_indice[x]
            else:
                new_indice = self.max_indice[x]
            self.indice[x] = self.indice[x] * conf[x] + new_indice * (1 - conf[x])
            for i, j in enumerate(self.sorted_neighbors[x]):
                sim = (1 - self.sorted_dist[x, i]) * conf[x]
                if sim < thres:
                    break
                conf[j] = conf.get(j, 0) + sim
                Q.append(j)

    def _get_sampling_N(self, p_ix: int):
        return self.sorted_neighbors[p_ix][:self.outer_ptr[p_ix]]

    def _get_alpha_n(self, indices: Union[int, np.ndarray]):
        return self.inner_ptr[indices]



class DetectorEnsemble:
    def __init__(self, mode = 'iforest', adjust_sample_weight = 0.01):
        self.detectors = []

        if mode == 'fast':
            self.detectors.append(('iforest1', IsolationForest(random_state = 0, max_samples = 128, n_estimators = 100)))
        elif mode == 'iforest':
            self.detectors.append(('iforest1', IsolationForest(random_state = 0, max_samples = 128, n_estimators = 100)))
            self.detectors.append(('iforest2', IsolationForest(random_state = 0, max_samples = 128, n_estimators = 200)))
            self.detectors.append(('iforest3', IsolationForest(random_state = 0, max_samples = 256, n_estimators = 100)))
            self.detectors.append(('iforest4', IsolationForest(random_state = 0, max_samples = 256, n_estimators = 200)))
            self.detectors.append(('iforest5', IsolationForest(random_state = 0, max_samples = 512, n_estimators = 100)))
            self.detectors.append(('iforest6', IsolationForest(random_state = 0, max_samples = 512, n_estimators = 200)))
        else:
            self.detectors.append(('knn', NearestNeighbors(algorithm='ball_tree')))
            self.detectors.append(('lof', LocalOutlierFactor(metric="precomputed")))
            self.detectors.append(('robustcov', MinCovDet()))
            self.detectors.append(('iforest', IsolationForest()))
            self.detectors.append(('ocsvm', OneClassSVM()))
            self.detectors.append(('dbscan',  DBSCAN()))
        self.adjust_sample_weight_ratio = adjust_sample_weight
    
    def fit_detector(self, X, y):
        self.clf = LinearRegression(fit_intercept=True, normalize=False, copy_X=True).fit(X, y)

    def fit(self, mat):
        dist = pairwise_distances(X = mat, metric='euclidean')
        self.scores = []
        for (name, detector) in self.detectors:
            if name[:3] == 'lof':
                detector.fit_predict(dist)
                self.scores.append(-detector.negative_outlier_factor_)
            elif name == 'robustcov':
                detector.fit(mat)
                self.scores.append(detector.mahalanobis(mat))
            elif name == 'knn':
                detector.fit(mat)
                self.scores.append(-detector.kneighbors(mat)[0][:, -1])
            elif name == 'dbscan':
                detector.fit(mat)
                score = np.array([1 if x == -1 else 0 for x in detector.labels_])
                self.scores.append(score)
            else:
                detector.fit_predict(mat)
                self.scores.append(-detector.score_samples(mat))
            print(name, min(self.scores[-1]), max(self.scores[-1]), self.scores[-1].shape)
        tmp = []
        for score in self.scores:
            min_s = np.min(score)
            max_s = np.max(score)
            range_s = max(1, max_s - min_s)
            score = (score - min_s) / range_s
            tmp.append(score)
        self.n = mat.shape[0]
        self.scores = np.array(tmp)
        self.ground_truth = {}
        self.adjust_sample_weight = self.n * self.adjust_sample_weight_ratio
        self.weights = np.ones(len(self.detectors))
        weights = self.weights / np.sum(self.weights)

        self.scores = self.scores.transpose()
        y = (self.scores * weights).sum(axis = 1)
        self.fit_detector(self.scores, y)
    
    def weighted_score(self):
        y = self.clf.predict(self.scores)
        for i in self.ground_truth:
            y[i] = self.ground_truth[i]
        return y

    def adjust_weight(self, idx, score):
        self.ground_truth[idx] = score
        sample_weight = np.ones(self.n)
        for i in self.ground_truth:
            sample_weight[i] = self.adjust_sample_weight
        y = self.weighted_score()
        self.fit_detector(self.scores, y)
