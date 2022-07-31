import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from lib.model_reduction_variant import Extractor
from lib.anomaly_detection import LOCIMatrixNew
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

from lib.model_utils import ModelUtil
model = ModelUtil(data_name = 'german_credit', model_name = 'RF')
paths = model.paths
cond_mat, output = model.get_rule_matrix()

high_conf_idxes = [i for i, p in enumerate(paths) if paths[i]['confidence'] > 0.8]

import copy
feature_pos = copy.deepcopy(model.feature_pos)

is_feature_categorical = {}
feature_val_idxs = {}
for i in model.feature_pos:
    if i == -1:
        continue
    feature, index = model.feature_pos[i]
    is_feature_categorical[feature] = index > 0

for i in model.feature_pos:
    if i == -1:
        continue
    feature, index = model.feature_pos[i]
    if is_feature_categorical[feature]:
        if feature not in feature_val_idxs:
            feature_val_idxs[feature] = []
        if i >= 0:
            feature_val_idxs[feature].append(i)

is_categorical = []
feature_range = []
for i in model.feature_pos:
    if i == -1:
        continue
    feature, index = model.feature_pos[i]
    is_categorical.append(is_feature_categorical[feature])
    feature_range.append(model.feature_range[feature])

n_features = len(is_categorical)
to_category_idx = [3]
for i in to_category_idx:
    name = model.feature_pos[i][0]
    is_categorical[i] = True
    feature_len =  model.feature_range[name][1] - model.feature_range[name][0]
    idx = [i] + [j for j in range(n_features, n_features + feature_len - 1)]
    for it, j in enumerate(idx):
        feature_pos[j] = (name, it)
        is_categorical.append(True)
    feature_val_idxs[name] = idx
    n_features += feature_len - 1

X = np.ones((len(paths), n_features)).astype('float') * 0.5
y = np.ones((len(paths), ))

mid = np.zeros((n_features)).astype('float')
columns = [feature for feature in model.data_table.columns if feature != model.target]
n_samples = len(model.data_table)
for i, feature in enumerate(columns):
    if is_categorical[i]:
        if i in to_category_idx:
            vmin = model.data_table[feature].min()
            vmax = model.data_table[feature].max()
            for k, j in enumerate(feature_val_idxs[feature]):
                val = (model.data_table[feature] == k + vmin).sum() / n_samples
                mid[j] = val
        else:
            _, k = feature_pos[i]
            vmin = model.data_table[feature].min()
            mid[i] = (model.data_table[feature] == 1).sum() / n_samples

for row_i, p in enumerate(paths):
    m = p['range']
    row = mid.copy()
    for i in m:
        if is_categorical[i]:
            feature, _ = model.feature_pos[i]
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
            left = model.get_sum(i, 1, left) / len(model.X)
            right = model.get_sum(i, 1, right) / len(model.X)
            row[i] = (left + right) / 2 - 0.5
    for feature in feature_val_idxs:
        idx = feature_val_idxs[feature]
        tot = row[idx].sum()
        if tot > 0:
            row[idx] /= tot

    X[row_i] = row
    y[row_i] = p['output']
lr_mat = X

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X[high_conf_idxes], y[high_conf_idxes])
y_pred = lr.predict(X[high_conf_idxes])
from sklearn.metrics import accuracy_score, precision_score, f1_score
print('Accuracy Score is', accuracy_score(y[high_conf_idxes], y_pred))
print('Precision Score is', precision_score(y[high_conf_idxes], y_pred))
print('F1 Score is', f1_score(y[high_conf_idxes], y_pred))

feature_importance = np.abs(lr.coef_[0])

mat = cond_mat * feature_importance
res_with_output = LOCIMatrixNew(mat, r = 0.28, metric = 'euclidean', n_ticks=100, output = output, output_alpha=0.6)
res_with_output.run()

predict_proba = lr.predict_proba(X)
lr_score = np.array([predict_proba[i, 1 - int(y[i])] for i in range(predict_proba.shape[0])])
loci_score = res_with_output.scores[:, 39]
mixed_score = loci_score / loci_score.max() + lr_score / lr_score.max()


for i, val in enumerate(mixed_score):
    paths[i]['score'] = val
    paths[i]['cost'] = 1.0 / np.exp(-val)

curves = []
n = 80
tau = model.parameters['n_estimators'] * n / len(paths) * 0.5
lambda_ = 0
ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))
while lambda_ < 2:
    w, _, fidelity_train, obj = ex.extract(n, tau, lambda_)
    [idx] = np.nonzero(w)
    scores = [paths[i]['score'] for i in idx]
    avg_score = np.mean(scores)
    second_term = np.sum([paths[i]['cost'] for i in idx]) * lambda_
    first_term = obj - second_term

    selected_paths = [paths[i] for i in idx]
    pred_train = ex.predict(model.X_train, selected_paths)
    pred_test = ex.predict(model.X_test, selected_paths)
    accuracy_train = accuracy_score(pred_train, model.y_train)
    accuracy_test = accuracy_score(pred_test, model.y_test)
    fidelity_train = accuracy_score(pred_train, model.clf.predict(model.X_train))
    fidelity_test = accuracy_score(pred_test, model.clf.predict(model.X_test))
    print(lambda_, len(idx), first_term, second_term, avg_score)
    curves.append((lambda_, first_term, 'fidelity'))
    curves.append((lambda_, second_term, 'score'))
    curves.append((lambda_, obj, 'obj'))
    f = open('../output/data/record_0722.txt', 'a')
    f.write('%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (lambda_, first_term, second_term, obj, avg_score, fidelity_train, fidelity_test))
    f.close()
    lambda_ += 0.1

df = pd.DataFrame({
    'x': [t[0] for t in curves],
    'y': [t[1] for t in curves],
    'label': [t[2] for t in curves],
})
plt.figure(figsize=(15, 10))
sns.lineplot(data=df, x='x', y='y', hue='label', markers=True)
plt.savefig('curve2.png')
