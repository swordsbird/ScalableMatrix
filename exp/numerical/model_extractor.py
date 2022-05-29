
from statistics import quantiles
import pulp
import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances

def anomaly_detection(mat, n_neighbors = 20):
    r_dist = pairwise_distances(X = mat, metric='euclidean')
    clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
    lof_outlier = clf.fit_predict(r_dist)
    lof_outlier = np.where(lof_outlier == 1, 0, 1)
    lof_score = -clf.negative_outlier_factor_

    clf = IsolationForest(random_state = 0)
    forest_outlier = clf.fit_predict(mat)
    forest_outlier = np.where(forest_outlier == 1, 0, 1)
    forest_score = clf.score_samples(mat)

    clf = OneClassSVM(gamma='auto')
    svm_outlier = clf.fit_predict(mat)
    svm_outlier = np.where(svm_outlier == 1, 0, 1)
    svm_score = clf.score_samples(mat)

    return {
        'One Class SVM': [svm_outlier, svm_score],
        'Local Outlier Factor': [lof_outlier, lof_score],
        'Isolation forest': [forest_outlier, forest_score],
    }
    #return -clf.negative_outlier_factor_

class Extractor:
    def __init__(self, paths, X_train, y_train):

        '''
        n_samples = 2000
        if len(y_train) > n_samples:
            idxes = random.sample(range(len(y_train)), n_samples)
            X_train = X_train[idxes]
            y_train = y_train[idxes]
        '''
        self.X_raw = X_train
        self.y_raw = y_train
        self.paths = [p for p in paths if len(np.flatnonzero(self.path_score(p, X_train, y_train))) > 0]

    def init_weight(self):
        self.max_weight = 5
        self.mat = self.getMat(self.X_raw, self.y_raw, self.paths)
        self.weights = self.getWeight(self.mat)
        self.weight = self.weights['Local Outlier Factor'][1]

    def compute_accuracy_on_train(self, paths):
        y_pred = self.predict(self.X_raw, paths)
        y_pred = np.where(y_pred == 1, 1, 0)
        return np.sum(np.where(y_pred == self.y_raw, 1, 0)) / len(self.X_raw)

    def anomaly_score(self, weights):
        scores = []
        for i in range(len(self.paths)):
            if weights[i] > 0:
                scores.append(self.weight[i])
        return np.mean(scores)
    
    def anomaly_info(self, percent = 0.02):
        anomaly_idxes = np.argsort(self.weight)[-int(len(self.weight) * percent):]
        self.anomaly_idxes = set(anomaly_idxes)
        quantiles = np.quantile(self.weight, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        return quantiles

    def anomaly_percent(self, weights):
        cnt = 0
        tot = 0
        for i in range(len(self.paths)):
            if weights[i] > 0:
                if i in self.anomaly_idxes:
                    cnt += 1
                tot += 1
        return cnt / tot

    def evaluate(self, weights, X, y):
        paths = deepcopy(self.paths)
        for i in range(len(paths)):
            paths[i]['weight'] = weights[i]
        y_pred = self.predict(X, paths)
        y_pred = np.where(y_pred == 1, 1, 0)
        return np.sum(np.where(y_pred == y, 1, 0)) / len(X)

    def extract(self, n_rules, n_trees, tau, lambda_):
        cost = [1.0 / x for x in self.weight]
        paths_weight = self.LP_extraction(cost, self.mat, n_rules, n_trees, tau, lambda_)
        accuracy_origin1 = self.compute_accuracy_on_train(self.paths)
        path_copy = deepcopy(self.paths)
        for i in range(len(path_copy)):
            path_copy[i]['weight'] = 1 if paths_weight[i] > 0 else 0
        accuracy_new1 = self.compute_accuracy_on_train(path_copy)
        return paths_weight, accuracy_origin1, accuracy_new1

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
        mat = np.where(np.abs(mat) > 0, 1.0, 0.0)
        #for i in range(mat.shape[0]):
        #    mat[i, :] /= mat[i].sum()
        weight = anomaly_detection(mat)
        return weight

    def LP_extraction(self, cost, mat, n_rules, n_trees, tau, lambda_):
        m = pulp.LpProblem(sense=pulp.LpMinimize)
        var = []
        N = mat.shape[1]
        M = mat.shape[0]
        zero = 1000
        for i in range(M):
            var.append(pulp.LpVariable(f'z{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
        for i in range(N):
            var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
        m += pulp.lpSum([var[j + M] for j in range(N)] + [var[j] * cost[j] * lambda_ * n_trees / M * N for j in range(M)])
        m += (pulp.lpSum([var[j] for j in range(M)]) <= n_rules)
        for j in range(N):
            m += (var[j + M] >= zero + tau - pulp.lpSum([var[k] * mat[k][j] for k in range(M)]))
            m += (var[j + M] >= zero)

        m.solve(pulp.PULP_CBC_CMD())
        z = [var[i].value() for i in range(M)]
        for k in np.argsort(z)[:-n_rules]:
            z[k] = 0
        z = z / np.sum(z)
        return z
