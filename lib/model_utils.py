import sys
import numpy as np
sys.path.append('../..')
from lib.tree_extractor import path_extractor, assign_samples
from lib.data_encoding import german_credit_encoding


class ModelUtil():
    def __init__(self, data_name, model_name):
        if data_name == 'german' and model_name == 'random forest':
            from lib.tree_ensemble_training.german_rf import get_model
            clf, (X_train, y_train, X_test, y_test, data_table), dataset, model, parameters = get_model()
            paths = path_extractor(clf, 'random forest', (X_train, y_train))
            target = 'credit_risk'
            X = data_table.drop(target, axis=1).values
            y = data_table[target].values
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
            self.paths = paths
            self.X = X
            self.y = y
            self.X_test = X_test
            self.y_test = y_test
            self.X_train = X_train
            self.y_train = y_train
            self.data_table = data_table
            self.model = model
            self.dataset = dataset
            self.parameters = parameters
            self.feature_range = feature_range
            self.feature_pos = feature_pos
            self.current_encoding = german_credit_encoding
            self.output_labels = ['reject', 'accept']
            self.suffix_sum = None
            self.categorical_data = ['foreign_worker', 'savings', 'personal_status_sex', 'credit_history', 'purpose', 'other_debtors', 'property', 'other_installment_plans', 'housing', 'job', 'people_liable', 'telephone', 'foreign_worker', 'number_credits']
    
    def init_suffix_sum(self, X):
        self.suffix_sum = []
        for i in range(X.shape[1]):
            sum = np.zeros(X[:, i].max() + 2)
            for val in X[:, i]:
                sum[val] += 1
            for i in range(1, len(sum)):
                sum[i] += sum[i - 1]
            self.suffix_sum.append(sum)

    def get_sum(self, key, left, right):
        return self.suffix_sum[int(key)][right] - self.suffix_sum[int(key)][left - 1]

    def interpret_path(self, path):
        conds = {}
        for k in path['range']:
            name = self.feature_pos[k][0]
            val = path['range'][k]
            if name in self.current_encoding:
                if name not in conds:
                    conds[name] = [1] * len(self.current_encoding[name])
                if name in self.data_table.columns:
                    for i in range(self.feature_range[name][0], self.feature_range[name][1]):
                        if i < val[0] or i > val[1]:
                            conds[name][i - self.feature_range[name][0]] = 0
                else:
                    if val[0] > 0:
                        conds[name] = [0] * len(self.current_encoding[name])
                        conds[name][self.feature_pos[k][1]] = 1
                    else:
                        conds[name][self.feature_pos[k][1]] = 0
            else:
                cond = [max(self.feature_range[name][0], val[0]), min(self.feature_range[name][1], val[1])]
                conds[name] = cond

        output_conds = []
        for name in conds:
            val = conds[name]
            op = 'is'
            value = ''
            if name in self.current_encoding:
                is_negation = np.sum(val) * 2 >= len(val) and len(val) > 2
                if is_negation:
                    op = 'is not'
                    for i, d in enumerate(val):
                        if d == 0:
                            value = value + ' and ' + self.current_encoding[name][i]
                    value = value[5:]
                else:
                    for i, d in enumerate(val):
                        if d == 1:
                            value = value + ' or ' + self.current_encoding[name][i]
                    value = value[4:]
            else:
                if val[0] == self.feature_range[name][0]:
                    op = '<='
                    value = int(val[1])
                elif val[1] == self.feature_range[name][1]:
                    op = '>='
                    value = int(val[0])
                else:
                    op = 'in'
                    value = '%d to %d' % (int(val[0]), int(val[1]))
            output_conds.append((name, op, value))
        output_label = self.output_labels[path['output']]
        return output_conds, output_label

    def check_path(self, path, X):
        n_samples = len(X)
        if self.suffix_sum is None:
            self.init_suffix_sum(X)

        m = path['range']
        cover = np.ones(n_samples)
        for key in m:
            name = self.feature_pos[key][0]
            if name in self.categorical_data:
                cover = cover * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
            else:
                cond = [int(max(self.feature_range[name][0], m[key][0])), int(min(self.feature_range[name][1], m[key][1]))]
                center_count = self.get_sum(int(key), cond[1], cond[0])
                remain_count = n_samples - center_count
                for i in range(len(cover)):
                    dist = 0
                    val = X[i, int(key)]
                    if val < cond[0]:
                        dist = self.get_sum(int(key), cond[0] - 1, val) / remain_count
                    elif val > cond[1]:
                        dist = self.get_sum(int(key), val, cond[1] + 1) / remain_count
                    cover[i] *= (1 - dist)
        return cover

    def get_cover_matrix(self, X, normalize = False, fuzzy = False):
        if not fuzzy:
            mat = np.array([p['sample'] for p in self.paths]).astype('float')
        else:
            mat = np.array([self.check_path(p, X) for p in self.paths]).astype('float')

        if normalize:
            for i, path in enumerate(self.paths):
                sum = np.sqrt(np.sum(mat[i]))
                if sum > 0:
                    mat[i] /= sum
        return mat
