from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sample import create_sampler
from ..lib.tree_extractor import path_extractor
from lightgbm import LGBMClassifier

random_state = 10

class ExpModel:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.n_splits = 4
        self._accuracy = []
        self._precision = []
        self._recall = []
        self._f1_score = []

    def init(self):
        if self.model == 'lightgbm':
            if self.dataset == 'bankruptcy':
                self.target = 'Bankrupt?'
                parameters = {
                    'n_estimators': 500,
                    'learning_rate': 0.15,
                    'num_leaves': 150,
                    'min_data_in_leaf': 200,
                    'max_depth': 6,
                    'max_bin': 239,
                    'min_data_in_leaf': 320,
                    'lambda_l1': 0.00000121865,
                    'lambda_l2': 0.03078951866,
                    'bagging_fraction': 0.908,
                    'feature_fraction': 0.943,
                    'bagging_freq': 4,
                    'min_child_samples': 10,
                }

                data_table = pd.read_csv('data/bank.csv')
                X = data_table.drop(self.target, axis=1).values
                y = data_table[self.target].values
            self.data_table = data_table
            self.X = X
            self.y = 1 - y
            self.parameters = parameters
            
            kf = KFold(n_splits = self.n_splits, random_state=random_state, shuffle=True)
            self.splits = []
            for train_index, test_index in kf.split(X):
                self.splits.append((train_index, test_index))
            self.fold = 0

    def has_next_fold(self):
        return self.fold < len(self.splits)
    
    def next_fold(self):
        self.fold += 1

    def train(self):
        sm = SMOTE(random_state=random_state)
        data_table = self.data_table
        X = self.X
        y = self.y
        parameters = self.parameters
        train_index, test_index = self.splits[self.fold]
        X_train = self.X[train_index]
        y_train = self.y[train_index]
        X_test = self.X[test_index]
        y_test = self.y[test_index]
        X_train, y_train = sm.fit_resample(X_train, y_train)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.data_table = data_table
        if self.model == 'RF':
            clf = RandomForestClassifier(**parameters)
        else:
            clf = LGBMClassifier(**parameters)
        clf.fit(X_train, y_train)
        self.clf = clf
        self.y_pred = clf.predict(X_test)
        self.features = data_table.drop(self.target, axis=1).columns.to_list()

    def evaluate(self):
        _accuracy = accuracy_score(self.y_test, self.y_pred)
        _precision = precision_score(self.y_test, self.y_pred)
        _recall = recall_score(self.y_test, self.y_pred)
        _f1_score = f1_score(self.y_test, self.y_pred)
        print('Accuracy Score is', _accuracy)
        print('Precision is', _precision)
        print('Recall is', _recall)
        print('F1 Score is', _f1_score)
        self._accuracy.append(_accuracy)
        self._precision.append(_precision)
        self._recall.append(_recall)
        self._f1_score.append(_f1_score)
    
    def summary(self):
        return float(np.mean(self._accuracy)), float(np.mean(self._precision)), float(np.mean(self._recall)), float(np.mean(self._f1_score))

    def oversampling(self, rate = 2):
        is_continuous = []
        is_categoryal = []
        is_integer = []

        for feature in self.data_table.columns:
            if feature == self.target:
                continue
            if self.data_table[feature].dtype == 'O':
                is_continuous.append(False)
                is_categoryal.append(True)
            else:
                is_continuous.append(True)
                is_categoryal.append(False)
            is_integer.append(False)
        sampler = create_sampler(self.X_train, is_continuous, is_categoryal, is_integer)
        return sampler(len(self.X_train) * rate)

    def generate_paths(self):
        if self.model == 'RF':
            paths = path_extractor(self.clf, 'random forest', (self.X_train, self.y_train))
        else:
            paths = path_extractor(self.clf, 'lightgbm')
        print('num of paths', len(paths))
        return paths

dataset = 'bankruptcy'
model = 'lightgbm'
exp = ExpModel(dataset, model)
exp.init()
exp.train()
paths = exp.generate_paths()
exp.evaluate()

from tree_extractor import assign_samples_lgbm as assign_samples
assign_samples(paths, (exp.X, exp.y))
paths = [p for p in paths if p['coverage'] > 0]

last_paths = paths
name2path = {}
for index, path in enumerate(paths):
    name2path[path['name']] = path
    path['level'] = 0

params = [600, 80]
level_info = {}

from model_extractor_maxnum import Extractor
for level, n in enumerate(params):
    tau = (n / 80) ** 0.5
    ex = Extractor(last_paths, exp.X_train, exp.clf.predict(exp.X_train))
    w, _, fidelity_train = ex.extract(n, tau)
    [idx] = np.nonzero(w)

    accuracy_train = ex.evaluate(w, exp.X_train, exp.y_train)
    accuracy_test = ex.evaluate(w, exp.X_test, exp.y_test)
    fidelity_test = ex.evaluate(w, exp.X_test, exp.clf.predict(exp.X_test))
    print(level, n, 'accuracy_train', accuracy_train, 'accuracy_test', accuracy_test, 'fidelity_test', fidelity_test)
    
    level_info[level + 1] = {
        'fidelity_test': fidelity_test,
        'accuracy_test': accuracy_test,
    }
    for i in idx:
        name2path[last_paths[i]['name']]['level'] = level + 1
    curr_paths = [last_paths[i] for i in idx]
    last_paths = curr_paths
    print(len(last_paths))

    import shap

explainer = shap.Explainer(exp.clf)
shap_values = explainer(exp.X)
shap_values = shap_values[:,:,0]
new_shaps = []

new_feature = {}
features = [feature for feature in exp.data_table.columns if feature != exp.target]
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

features = []
feature_index = {}
feature_type = {}
for key in new_feature:
    if len(new_feature[key]) == 1:
        i = new_feature[key][0]
        values = exp.data_table[key].values
        values.sort()
        n = len(values)
        qmin, qmax = values[0], values[-1]
        q5, q25, q50, q75, q95 = values[n * 5 // 100], values[n * 25 // 100], values[n * 50 // 100], values[n * 75 // 100], values[n * 95 // 100]
        features.append({
            "name": key,
            "quantile": { "5": q5, "25": q25, "50": q50, "75": q75, "95": q95 },
            "range": [qmin, qmax],
            "importance": exp.clf.feature_importances_[i],
            "dtype": "number",
        })
        shaps = shap_values[:, i]
        feature_index[i] = len(features) - 1
        feature_type[i] = "number"
    else:
        features.append({
            "name": key,
            "range": [0, len(new_feature[key])],
            "importance": sum([exp.clf.feature_importances_[i] for i in new_feature[key] if i != -1]),
            "dtype": "category",
        })
        feature_idxs = [i for i in  new_feature[key] if i != -1]
        shaps = shap_values[:, feature_idxs[0]]
        for i in feature_idxs[1:]:
            shaps = shaps + shap_values[:, i]

        for index, i in enumerate(new_feature[key]):
            if i != -1:
                feature_index[i] = [len(features) - 1, index]
                feature_type[i] = "category"
    new_shaps.append(shaps)

for path in paths:
    new_range = {}
    for index in path['range']:
        if feature_type[int(index)] == 'number':
            i = feature_index[int(index)]
            new_range[i] = path['range'][index]
        else:
            i, j = feature_index[int(index)]
            if i not in new_range:
                new_range[i] = [0] * features[i]['range'][1]
                if path['range'][index][0] <= 1 and 1 <= path['range'][index][1]:
                    new_range[i][j] = 1
                else:
                    for k in range(len(new_range[i])):
                        if k != j:
                            new_range[i][k] = 1
                        new_range[i][j] = 0
    path['range'] = new_range
    path['represent'] = False

for i in idx:
    paths[i]['represent'] = True

output_data = {
    'paths': paths,
    'features': features,
    'selected': [paths[i]['name'] for i in idx],
    'shap_values': new_shaps,
    'model_info': {
        'accuracy': exp._accuracy[-1],
        'info': level_info,
        'num_of_rules': len(paths),
        'dataset': 'Taiwan Company Bankruptcy',
        'model': 'LightGBM',
    }
}

import pickle
pickle.dump(output_data, open('output/bankruptcy0129.pkl', 'wb'))