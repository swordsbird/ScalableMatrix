
import pulp
import numpy as np
from copy import deepcopy
from sklearn.neighbors import LocalOutlierFactor

class Extractor:
    # 可以调用的接口：compute_accuracy和extract
    def __init__(self, paths, X_train, y_train):
        # X_raw、y_raw：训练数据集
        self.X_raw = X_train
        self.y_raw = y_train
        self.paths = paths

    def compute_accuracy_on_train(self, paths):
        # 计算训练集在给定规则集下的accuracy
        # paths：规则集
        y_pred = self.predict(self.X_raw, paths)
        y_pred = np.where(y_pred == 1, 1, 0)
        return np.sum(np.where(y_pred == self.y_raw, 1, 0)) / len(self.X_raw)

    def evaluate(self, weights, X, y):
        paths = deepcopy(self.paths)
        for i in range(len(paths)):
            paths[i]['weight'] =  weights[i]# 1 if weights[i] > 0 else 0
        y_pred = self.predict(X, paths)
        y_pred = np.where(y_pred == 1, 1, 0)
        return np.sum(np.where(y_pred == y, 1, 0)) / len(X)

    def extract(self, lamb, tau):
        # 根据给定的max_num和tau，使用rf的全部规则和数据集抽取出相应的规则
        # max_num：抽取出规则的最大数量
        # tau：每个样本允许的最大惩罚
        # 返回抽取出规则的列表、数据集使用全部规则的accuracy、数据集使用抽取规则的accuracy

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
        # 根据给定规则集对数据进行预测
        Y = np.zeros(X.shape[0])
        for p in paths:
            ans = np.ones(X.shape[0])
            m = p.get('range')
            for key in m:
                ans = ans * (X[:,int(key)] >= m[key][0]) * (X[:,int(key)] < m[key][1])
            Y += ans * (p.get('weight') > 0)
        #Y = np.where(Y > 0, 1, 0)
        return Y

    def predict(self, X, paths):
        # 根据给定规则集对数据进行预测
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

    def getWeight(self, Mat):
        # 权重向量w
        RXMat = np.abs(Mat)
        XRMat = RXMat.transpose()
        XXAnd = np.dot(XRMat, RXMat)
        XROne = np.ones(XRMat.shape)
        XXOr = 2 * np.dot(XROne, RXMat) - XXAnd
        XXOr = (XXOr + XXOr.transpose()) / 2
        XXDis = 1 - XXAnd / XXOr
        K = int(np.ceil(np.sqrt(len(self.X_raw))))
        clf = LocalOutlierFactor(n_neighbors=K, metric="precomputed")
        clf.fit(XXDis)
        XW = -clf.negative_outlier_factor_
        MXW, mXW = np.max(XW), np.min(XW)
        XW = 1 + (3 - 1) * (XW - mXW) / (MXW - mXW)
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
        # 添加变量x_0至x_{M-1}, k_0至k_{N-1}

        m += pulp.lpSum([cost[j] * var[j + M] for j in range(N)] + [var[i] * lamb for i in range(M)])
        # 添加目标函数
        # m += (pulp.lpSum([var[j] for j in range(M)]) <= max_num)

        for j in range(N):
            m += (var[j + M] >= zero + tau - pulp.lpSum([var[k] * y[k][j] for k in range(M)]))
            m += (var[j + M] >= zero)
            # max约束

        m.solve(pulp.PULP_CBC_CMD())  # solver = pulp.solver.CPLEX())#
        z = [var[i].value() for i in range(M)]
        z = np.array([value if value > 0.5 else 0 for value in z])
        z = z / np.sum(z)
        print('total path weight: ', sum(z))
        return z
