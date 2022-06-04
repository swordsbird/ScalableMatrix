
import pulp
import numpy as np
from copy import deepcopy
from .anomaly_detection import DetectorEnsemble

class Extractor:
    def __init__(self, paths, X_train, y_train):
        # n_samples = 2000
        # if len(y_train) > n_samples:
        #    idxes = random.sample(range(len(y_train)), n_samples)
        #    X_train = X_train[idxes]
        #    y_train = y_train[idxes]
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

    def extract(self, n_rules, tau, lambda_, method = 'maximize'):
        mat = self.mat
        w = self.weight
        if method == 'maximize':
            paths_weight, obj = self.LP_extraction_maximize(w, mat, n_rules, tau, lambda_)
        else:
            paths_weight, obj = self.LP_extraction_minimize(w, mat, n_rules, tau, lambda_)
        accuracy_origin1 = self.compute_accuracy_on_train(self.paths)
        path_copy = deepcopy(self.paths)
        for i in range(len(path_copy)):
            path_copy[i]['weight'] = 1 if paths_weight[i] > 0 else 0
        accuracy_new1 = self.compute_accuracy_on_train(path_copy)
        return paths_weight, accuracy_origin1, accuracy_new1, obj

    def LP_extraction_minimize(self, score, y, n_rules, tau, lambda_):
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
        m += (pulp.lpSum([var[j + M] for j in range(N)]) <= first_term)
        m += (pulp.lpSum([var[j] * score[j] * lambda_ for j in range(M)]) <= second_term)
        m += (pulp.lpSum([var[j] for j in range(M)]) <= n_rules)
        m += (pulp.lpSum([var[j] for j in range(M)]) >= n_rules)
        for j in range(N):
            m += (var[j + M] >= zero + tau - pulp.lpSum([var[k] * y[k][j] for k in range(M)]))
            m += (var[j + M] >= zero)

        m.solve(pulp.PULP_CBC_CMD())  # solver = pulp.solver.CPLEX())#
        z = [var[i].value() for i in range(M)]
        for k in np.argsort(z)[:-n_rules]:
            z[k] = 0
        z = z / np.sum(z)
        return z, (pulp.value(m.objective) - zero * N, first_term.value() - zero * N, second_term.value())

    def LP_extraction_maximize(self, score, y, n_rules, tau, lambda_):
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
