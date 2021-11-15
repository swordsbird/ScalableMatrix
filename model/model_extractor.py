
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

    def extract(self, max_num, tau):
        # 根据给定的max_num和tau，使用rf的全部规则和数据集抽取出相应的规则
        # max_num：抽取出规则的最大数量
        # tau：每个样本允许的最大惩罚
        # 返回抽取出规则的列表、数据集使用全部规则的accuracy、数据集使用抽取规则的accuracy
        Mat = self.getMat(self.X_raw, self.y_raw, self.paths)
        #print('getWeight')
        w = self.getWeight(self.getMat(self.X_raw, self.y_raw, self.paths))
        #print('LP_extraction')
        paths_weight = self.LP_extraction(w, Mat, max_num, tau)
        #print('compute_accuracy_on_test')
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

    def LP_extraction(self, w, Mat, max_num, tau):
        m = pulp.LpProblem(sense=pulp.LpMinimize)
        # 创建最小化问题
        var = []
        for i in range(len(self.paths)):
            var.append(pulp.LpVariable(f'x{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
        for i in range(len(w)):
            var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
        # 添加变量x_0至x_{M-1}, k_0至k_{N-1}

        m += pulp.lpSum([w[j] * (var[j + len(self.paths)])
                         for j in range(len(w))])
        # 添加目标函数

        m += (pulp.lpSum([var[j] for j in range(len(self.paths))]) <= max_num)
        # 筛选出不超过max_num条规则

        for j in range(len(w)):
            m += (var[j + len(self.paths)] >= 1000 + tau - pulp.lpSum(
                [var[k] * Mat[k][j] for k in range(len(self.paths))]))
            m += (var[j + len(self.paths)] >= 1000)
            # max约束

        m.solve(pulp.PULP_CBC_CMD())  # solver = pulp.solver.CPLEX())#
        paths_weight = [var[i].value() for i in range(len(self.paths))]
        paths_weight = np.array(paths_weight)
        self.ow = paths_weight
        paths_weight = paths_weight / np.sum(paths_weight)
        for k in np.argsort(paths_weight)[:-max_num]:
            paths_weight[k] = 0
        #print('paths_weight', sum(paths_weight))
        return paths_weight
