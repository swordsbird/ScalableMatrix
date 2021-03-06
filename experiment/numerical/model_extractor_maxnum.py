
import pulp
import numpy as np
from copy import deepcopy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances

class Extractor:
    def __init__(self, paths, X_train, y_train):
        self.X_raw = X_train
        self.y_raw = y_train
        self.paths = paths
        self.max_weight = 5

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

    def extract(self, lamb, tau):
        Mat = self.getMat(self.X_raw, self.y_raw, self.paths)
        w = self.getWeight(self.getMat(self.X_raw, self.y_raw, self.paths))
        paths_weight = self.LP_extraction(w, Mat, lamb, tau)
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
        Mat = np.array([self.path_score(p, X_raw, y_raw) for p in paths]).astype('float')
        self.ma = Mat
        return Mat

    def path_score(self, path, X, y):
        value = float(path.get('value'))
        y = y * 2 - 1
        ans = value * y
        m = path.get('range')
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        return ans

    def getWeight(self, mat):
        return np.ones(len(self.y_raw))
        path_mat = np.abs(mat).transpose()
        path_dist = pairwise_distances(X = path_mat, metric='jaccard')
        K = min(10, int(np.ceil(np.sqrt(len(self.paths)))))
        clf = LocalOutlierFactor(n_neighbors=K, metric="precomputed")
        clf.fit(path_dist)
        XW = -clf.negative_outlier_factor_
        MXW, mXW = np.max(XW), np.min(XW)
        XW = 1 + (self.max_weight - 1) * (XW - mXW) / (MXW - mXW)
        self.XW = XW / np.sum(XW)
        return self.XW

    def LP_extraction(self, cost, y, lamb, tau):
        m = pulp.LpProblem(sense=pulp.LpMinimize)
        var = []
        N = len(cost)
        M = len(self.paths)
        zero = 1000
        for i in range(M):
            var.append(pulp.LpVariable(f'z{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
        for i in range(N):
            var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
        m += pulp.lpSum([cost[j] * var[j + M] for j in range(N)])
        m += (pulp.lpSum([var[j] for j in range(M)]) <= lamb)
        for j in range(N):
            m += (var[j + M] >= zero + tau - pulp.lpSum([var[k] * y[k][j] for k in range(M)]))
            m += (var[j + M] >= zero)

        m.solve(pulp.PULP_CBC_CMD())  # solver = pulp.solver.CPLEX())#
        z = [var[i].value() for i in range(M)]
        for k in np.argsort(z)[:-lamb]:
            z[k] = 0
        z = z / np.sum(z)
        return z
