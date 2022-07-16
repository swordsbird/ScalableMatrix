import sys
import math
import numpy as np
sys.path.append('../..')
from lib.tree_extractor import path_extractor, assign_samples
from lib.data_encoding import german_credit_encoding
import copy

def rule_to_text(rule):
    conds = rule[0]
    ret = rule[1]
    return 'IF ' + ' AND '.join(['%s %s %s' % (str(a), str(b), str(c)) for (a, b, c) in conds]) + ' THEN ' + ret

class ModelUtil():
    def __init__(self, data_name, model_name, parameters = None):
        if data_name == 'german_credit' and model_name == 'RF' or data_name == 'german' and model_name == 'random forest':
            from lib.tree_ensemble_training.german_rf import get_model
            clf, (X_train, y_train, X_test, y_test, data_table), dataset, model, parameters = get_model(parameters)
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
            self.clf = clf
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
            self.target = target
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

    def interpret_path(self, path, to_text = False):
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
        if to_text:
            return rule_to_text((output_conds, output_label))
        else:
            return output_conds, output_label

    def check_path(self, path, X, byclass = False):
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
                center_count = self.get_sum(int(key), cond[0], cond[1])
                remain_count = n_samples - center_count
                for i in range(len(cover)):
                    dist = 0
                    val = X[i, int(key)]
                    if val < cond[0]:
                        dist = self.get_sum(int(key), val, cond[0] - 1) / remain_count
                    elif val > cond[1]:
                        dist = self.get_sum(int(key), cond[1] + 1, val) / remain_count
                    cover[i] *= (1 - dist)
        if byclass:
            conf = np.array(path['distribution']).astype('float') ** 2
            if conf.sum() >= 1:
                conf /= conf.sum()
            cover = (cover.repeat(2).reshape((n_samples, 2)) * conf).reshape(-1)
        return cover

    def get_rule_matrix(self):
        if self.suffix_sum is None:
            self.init_suffix_sum(self.X)

        feature_pos = copy.deepcopy(self.feature_pos)

        is_feature_categorical = {}
        feature_val_idxs = {}
        for i in self.feature_pos:
            if i == -1:
                continue
            feature, index = self.feature_pos[i]
            is_feature_categorical[feature] = index > 0

        for i in self.feature_pos:
            if i == -1:
                continue
            feature, index = self.feature_pos[i]
            if is_feature_categorical[feature]:
                if feature not in feature_val_idxs:
                    feature_val_idxs[feature] = []
                if i >= 0:
                    feature_val_idxs[feature].append(i)

        is_categorical = []
        feature_range = []
        for i in self.feature_pos:
            if i == -1:
                continue
            feature, index = self.feature_pos[i]
            is_categorical.append(is_feature_categorical[feature])
            feature_range.append(self.feature_range[feature])

        paths = self.paths
        n_features = len(is_categorical)
        to_category_idx = [3]
        for i in to_category_idx:
            name = self.feature_pos[i][0]
            is_categorical[i] = True
            feature_len =  self.feature_range[name][1] - self.feature_range[name][0]
            idx = [i] + [j for j in range(n_features, n_features + feature_len - 1)]
            for it, j in enumerate(idx):
                feature_pos[j] = (name, it)
                is_categorical.append(True)
            feature_val_idxs[name] = idx
            n_features += feature_len - 1

        X = np.ones((len(paths), n_features)).astype('float') * 0.5
        y = np.ones((len(paths), ))

        mid = np.zeros((n_features)).astype('float')
        columns = [feature for feature in self.data_table.columns if feature != self.target]
        n_samples = len(self.data_table)
        for i, feature in enumerate(columns):
            if is_categorical[i]:
                if i in to_category_idx:
                    vmin = self.data_table[feature].min()
                    vmax = self.data_table[feature].max()
                    for k, j in enumerate(feature_val_idxs[feature]):
                        val = (self.data_table[feature] == k + vmin).sum() / n_samples
                        mid[j] = val
                else:
                    _, k = feature_pos[i]
                    vmin = self.data_table[feature].min()
                    mid[i] = (self.data_table[feature] == 1).sum() / n_samples

        for row_i, p in enumerate(paths):
            m = p['range']
            row = mid.copy()
            for i in m:
                if is_categorical[i]:
                    feature, _ = self.feature_pos[i]
                    idx = feature_val_idxs[feature]
                    if i in to_category_idx:
                        ll = feature_range[i][0]
                        rr = feature_range[i][1]
                        left = int(max(ll, math.floor(m[i][0])))
                        right = int(min(rr, math.ceil(m[i][1])))
                        if ll > 0:
                            left -= ll
                            right -= ll
                            rr -= ll
                            ll = 0
                        rr = min(rr, len(idx))
                        right = min(right, rr)
                        for j in range(rr):
                            row[idx[j]] = 0
                        for j in range(left, right):
                            row[idx[j]] = mid[idx[j]]
                    else:
                        if m[i][1] > 1:
                            if row[idx].sum() == 1:
                                row[idx] = 0
                            row[i] = mid[i]
                        else:
                            row[i] = 0
                else:
                    left = int(max(feature_range[i][0], m[i][0]))
                    right = int(min(feature_range[i][1] - 1, m[i][1]))
                    left = self.get_sum(i, 1, left) / len(self.X)
                    right = self.get_sum(i, 1, right) / len(self.X)
                    row[i] = (left + right) / 2 - 0.5
            for feature in feature_val_idxs:
                idx = feature_val_idxs[feature]
                tot = row[idx].sum()
                if tot > 0:
                    row[idx] /= tot

            X[row_i] = row
            y[row_i] = p['output']

        self.feature_name = []
        for i in range(n_features):
            name, k = feature_pos[i]
            if name in self.current_encoding and is_categorical[i]:
                name = name + ' ' + self.current_encoding[name][k]
            self.feature_name.append(name)
        return X, y

    def get_cover_matrix(self, X, normalize = False, fuzzy = False, byclass = False):
        if not fuzzy:
            mat = np.array([p['sample'] for p in self.paths]).astype('float')
        else:
            if byclass:
                mat = np.array([self.check_path(p, X, byclass=True) for p in self.paths]).astype('float')
            else:
                mat = np.array([self.check_path(p, X) for p in self.paths]).astype('float')

        if normalize:
            for i, path in enumerate(self.paths):
                sum = np.sqrt(np.sum(mat[i]))
                if sum > 0:
                    mat[i] /= sum
        return mat

def export_rules_to_csv(filename, model, idxes):
    rules = []
    max_n_conds = 0
    for it, i in enumerate(idxes):
        conds, output = model.interpret_path(model.paths[i])
        rules.append({'cond': conds, 'predict': output, 'index': i, 'order': it, 'attr': 0 })
        max_n_conds = max(len(conds), max_n_conds)
    conds_per_line = 4
    max_n_conds = math.ceil(max_n_conds / conds_per_line) * conds_per_line

    f = open(filename + '.csv', 'w')

    for it, rule in enumerate(rules):
        s = '' + str(rule['order'])
        line = 0
        n_conds = len(rule['cond'])
        n_lines = math.ceil(n_conds / conds_per_line)
        index = rule['index']

        for line in range(n_lines):
            if line == 0:
                s += ',#%d,IF,' % (index)
            else:
                s += ',,,'
            for pos in range(conds_per_line):
                i = pos + line * conds_per_line
                if i < n_conds:
                    item = rule['cond'][i]
                    s += item[0] + ',' + item[1] + ',' + str(item[2]) + ','
                    s += 'AND,' if i < n_conds - 1 else '...,'
                else:
                    s += '...,...,...,...,'
            if line == n_lines - 1:
                s = s[:-4]
                s += 'THEN,%s,%d,%3f' % (rule['predict'], np.sum(model.paths[index]['distribution']), model.paths[index]['confidence'])
            s += '\n'
        f.write(s + '\n')
    f.close()
