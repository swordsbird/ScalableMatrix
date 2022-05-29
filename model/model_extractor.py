
import pulp
import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances
from sklearn.covariance import MinCovDet
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LinearRegression
class DetectorEnsemble:
    def __init__(self):
        self.detectors = []
        self.detectors.append(('1', IsolationForest(random_state = 0, n_estimators = 100)))
        self.detectors.append(('2', IsolationForest(random_state = 0, n_estimators = 200)))
        self.detectors.append(('3',IsolationForest(random_state = 10, n_estimators = 100)))
        self.detectors.append(('4', IsolationForest(random_state = 10, n_estimators = 200)))
        self.detectors.append(('5',  IsolationForest(random_state = 20, n_estimators = 100, contamination = 0.2)))
        self.detectors.append(('6',  IsolationForest(random_state = 20, n_estimators = 200, contamination = 0.2)))
        '''
        self.detectors.append(('knn', NearestNeighbors(algorithm='ball_tree')))
        self.detectors.append(('lof2', LocalOutlierFactor(metric="precomputed")))
        self.detectors.append(('iforest1', IsolationForest(random_state = 0)))
        self.detectors.append(('ocsvm1', OneClassSVM(gamma='auto', kernel='rbf')))
        self.detectors.append(('dbscan',  DBSCAN()))
        '''
    
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
        self.adjust_sample_weight = self.n // 100
        self.weights = np.ones(len(self.detectors))
        weights = self.weights / np.sum(self.weights)

        self.scores = self.scores.transpose()
        y = (self.scores * weights).sum(axis = 1)
        print('before fit', self.scores.shape, y.shape)
        self.fit_detector(self.scores, y)
        print('after fit')
    
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

class Extractor:
    def __init__(self, paths, X_train, y_train):
        n_samples = 2000
        if len(y_train) > n_samples:
            idxes = random.sample(range(len(y_train)), n_samples)
            X_train = X_train[idxes]
            y_train = y_train[idxes]
        self.X_raw = X_train
        self.y_raw = y_train
        self.paths = paths
        self.mat = self.getMat(self.X_raw, self.y_raw, self.paths)
        self.weight = self.getWeight(self.mat)

    def compute_accuracy_on_train(self, paths):
        y_pred = self.predict(self.X_raw, paths)
        y_pred = np.where(y_pred == 1, 1, 0)
        return np.sum(np.where(y_pred == self.y_raw, 1, 0)) / len(self.X_raw)

    def evaluate(self, weights, X, y):
        paths = deepcopy(self.paths)
        for i in range(len(paths)):
            paths[i]['weight'] =  weights[i]
        y_pred = self.predict(X, paths)
        y_pred = np.where(y_pred == 1, 1, 0)
        return np.sum(np.where(y_pred == y, 1, 0)) / len(X)

    def extract(self, n_rules, tau, lambda_):
        mat = self.mat
        w = self.weight
        paths_weight, obj = self.LP_extraction(w, mat, n_rules, tau, lambda_)
        accuracy_origin1 = self.compute_accuracy_on_train(self.paths)
        path_copy = deepcopy(self.paths)
        for i in range(len(path_copy)):
            path_copy[i]['weight'] = 1 if paths_weight[i] > 0 else 0
        accuracy_new1 = self.compute_accuracy_on_train(path_copy)
        return paths_weight, accuracy_origin1, accuracy_new1, obj

    def predict_raw(self, X, paths):
        Y = np.zeros(X.shape[0])
        for p in paths:
            ans = np.ones(X.shape[0])
            m = p.get('range')
            for key in m:
                ans = ans * (X[:,int(key)] >= m[key][0]) * (X[:,int(key)] < m[key][1])
            Y += ans * (p.get('weight') * p.get('value'))
        return Y

    def coverage(self, weights, X):
        paths = deepcopy(self.paths)
        for i in range(len(paths)):
            paths[i]['weight'] =  weights[i]# 1 if weights[i] > 0 else 0
        return self.coverage_raw(X, paths)

    def coverage_raw(self, X, paths):
        Y = np.zeros(X.shape[0])
        for p in paths:
            ans = np.ones(X.shape[0])
            m = p.get('range')
            for key in m:
                ans = ans * (X[:,int(key)] >= m[key][0]) * (X[:,int(key)] < m[key][1])
            Y += ans * (p.get('weight') > 0)
        return Y

    def predict(self, X, paths):
        Y = np.zeros(X.shape[0])
        for p in paths:
            ans = np.ones(X.shape[0])
            m = p.get('range')
            for key in m:
                ans = ans * (X[:,int(key)] >= m[key][0]) * (X[:,int(key)] < m[key][1])
            Y += ans * (p.get('weight') * p.get('value'))
        Y = np.where(Y > 0, 1, 0)
        return Y

    def getMat(self, X_raw, y_raw, paths):
        mat = np.array([self.path_score(p, X_raw, y_raw) for p in paths]).astype('float')
        return mat

    def path_score(self, path, X, y):
        value = float(path.get('value'))
        y = y * 2 - 1
        ans = value * y
        m = path.get('range')
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        return ans

    def getWeight(self, mat):
        path_mat = np.abs(mat)
        ensemble = DetectorEnsemble()
        ensemble.fit(path_mat)
        YW = np.array([x for x in ensemble.weighted_score()])
        self.YW = YW
        return self.YW

    def LP_extraction(self, score, y, n_rules, tau, lambda_):
        m = pulp.LpProblem(sense=pulp.LpMaximize)
        var = []
        N = y.shape[1]
        M = y.shape[0]
        zero = 1000
        for i in range(M):
            var.append(pulp.LpVariable(f'z{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
        for i in range(N):
            var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
        first_term = pulp.LpVariable('first', cat=pulp.LpContinuous, lowBound=0)
        second_term = pulp.LpVariable('second', cat=pulp.LpContinuous, lowBound=0)
        m.setObjective(first_term + second_term)
        m += (pulp.lpSum([var[j + M] for j in range(N)]) >= first_term)
        m += (pulp.lpSum([var[j] * score[j] * lambda_ for j in range(M)]) >= second_term)
        m += (pulp.lpSum([var[j] for j in range(M)]) <= n_rules)
        for j in range(N):
            m += (var[j + M] <= zero + pulp.lpSum([var[k] * y[k][j] for k in range(M)]))
            m += (var[j + M] <= zero + tau)

        m.solve(pulp.PULP_CBC_CMD())  # solver = pulp.solver.CPLEX())#
        z = [var[i].value() for i in range(M)]
        for k in np.argsort(z)[:-n_rules]:
            z[k] = 0
        z = z / np.sum(z)
        return z, (pulp.value(m.objective) - zero * N, first_term.value() - zero * N, second_term.value())
