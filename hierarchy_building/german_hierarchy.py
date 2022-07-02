from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import sys
sys.path.append('..')
from lib.tree_extractor import path_extractor
from lib.model_reduction_variant import Extractor
from lib.exp_model import ExpModel
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

random_state = 190

dataset = 'german_credit'
model = 'RF'
exp = ExpModel(dataset, model)
exp.init()
exp.train()
paths = exp.generate_paths()
exp.evaluate()

from lib.tree_extractor import assign_samples
assign_samples(paths, (exp.X, exp.y))
paths = [p for p in paths if p['coverage'] > 0]

last_paths = paths
name2path = {}
for index, path in enumerate(paths):
    name2path[path['name']] = path
    path['level'] = 0

params = [80]
level_info = {}

curves = []

for level, n in enumerate(params):
    tau = exp.parameters['n_estimators'] * n / len(paths) * 0.5
    lambda_ = 0.01
    ex = Extractor(last_paths, exp.X_train, exp.clf.predict(exp.X_train))
    while lambda_ < 40:
        w, _, fidelity_train, result = ex.extract(n, tau, lambda_)
        [idx] = np.nonzero(w)

        accuracy_train = ex.evaluate(w, exp.X_train, exp.y_train)
        accuracy_test = ex.evaluate(w, exp.X_test, exp.y_test)
        fidelity_train = ex.evaluate(w, exp.X_train, exp.clf.predict(exp.X_train))
        fidelity_test = ex.evaluate(w, exp.X_test, exp.clf.predict(exp.X_test))
        obj, first_term, second_term = result
        second_term /= lambda_
        curves.append((lambda_, first_term, 'fidelity'))
        curves.append((lambda_, second_term, 'score'))
        curves.append((lambda_, obj, 'obj'))
        f = open('../output/data/record_0608.txt', 'a')
        f.write('%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (lambda_, first_term, second_term, obj, fidelity_train, fidelity_test))
        f.close()
        lambda_ += 1

    level_info[level + 1] = {
        'fidelity_test': fidelity_test,
        'accuracy_test': accuracy_test,
    }
    for i in idx:
        name2path[last_paths[i]['name']]['level'] = level + 1
    curr_paths = [last_paths[i] for i in idx]
    last_paths = curr_paths


df = pd.DataFrame({
    'x': [t[0] for t in curves],
    'y': [t[1] for t in curves],
    'label': [t[2] for t in curves],
})
plt.figure(figsize=(15, 10))
sns.lineplot(data=df, x='x', y='y', hue='label', markers=True)
plt.savefig('curve2.png')

'''

for level, n in enumerate(params):
    tau = exp.parameters['n_estimators'] * n / len(paths) * 0.5
    lambda_ = 6
    ex = Extractor(last_paths, exp.X_train, exp.clf.predict(exp.X_train))
    w, _, fidelity_train, result = ex.extract(n, tau, lambda_)
    [idx] = np.nonzero(w)

    accuracy_train = ex.evaluate(w, exp.X_train, exp.y_train)
    accuracy_test = ex.evaluate(w, exp.X_test, exp.y_test)
    fidelity_train = ex.evaluate(w, exp.X_train, exp.clf.predict(exp.X_train))
    fidelity_test = ex.evaluate(w, exp.X_test, exp.clf.predict(exp.X_test))
    obj, first_term, second_term = result
    second_term /= lambda_
    print('lambda: %.6f\nfirst term: %.6f\nsecond term: %.6f\nobj value: %.6f\nfidelity train: %.6f\nfidelity test: %.6f\n' % (lambda_, first_term, second_term, obj, fidelity_train, fidelity_test))

    level_info[level + 1] = {
        'fidelity_test': fidelity_test,
        'accuracy_test': accuracy_test,
    }
    for i in idx:
        name2path[last_paths[i]['name']]['level'] = level + 1
    curr_paths = [last_paths[i] for i in idx]
    last_paths = curr_paths

'''

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

value_encoding = {
    'credit_risk' : ['No', 'Yes'], 
    'credit_history' : [
        "delay in paying off in the past",
        "critical account/other credits elsewhere",
        "no credits taken/all credits paid back duly",
        "existing credits paid back duly till now",
        "all credits at this bank paid back duly",
    ],
    'purpose' : [
        "others",
        "car (new)",
        "car (used)",
        "furniture/equipment",
        "radio/television",
        "domestic appliances",
        "repairs",
        "vacation",
        "retraining",
        "business"
    ],
    'installment_rate': ["< 20", "20 <= ... < 25",  "25 <= ... < 35", ">= 35"],
    'present_residence': [
        "< 1 yr", 
        "1 <= ... < 4 yrs",
        "4 <= ... < 7 yrs", 
        ">= 7 yrs"
    ],
    'number_credits': ["1", "2-3", "4-5", ">= 6"],
    'people_liable': ["0 to 2", "3 or more"],
    'savings': [
        "unknown/no savings account",
        "... <  100 DM", 
        "100 <= ... <  500 DM",
        "500 <= ... < 1000 DM", 
        "... >= 1000 DM",
    ],
    'employment_duration': [
        "unemployed", 
        "< 1 yr", 
        "1 <= ... < 4 yrs",
        "4 <= ... < 7 yrs", 
        ">= 7 yrs"
    ],
    'personal_status_sex': [
        "not married male",
        "married male",
    ],
    'other_debtors': [
        'none',
        'co-applicant',
        'guarantor'
    ],
    'property': [
        "real estate",
        "building soc. savings agr./life insurance", 
        "car or other",
        "unknown / no property",
    ],
    'other_installment_plans': ['bank', 'stores', 'none'],
    'housing': ["rent", "own", "for free"],
    'job': [
        'unemployed/ unskilled - non-resident',
        'unskilled - resident',
        'skilled employee / official',
        'management/ self-employed/ highly qualified employee/ officer'
    ],
    'status': [
        "no checking account",
        "... < 0 DM",
        "0<= ... < 200 DM",
        "... >= 200 DM / salary for at least 1 year",
    ],
    'telephone': ['No', 'Yes'],
    'foreign_worker': ['No', 'Yes'],
}

features = []
feature_index = {}
feature_type = {}
for key in new_feature:
    if len(new_feature[key]) == 1:
        i = new_feature[key][0]
        if key in ['status', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex']:
            min_value = min(exp.data_table[key].values)
            max_value = max(exp.data_table[key].values)
            unique_values = np.unique(exp.data_table[key].values) - min_value
            sorted(unique_values)
            features.append({
                "name": key,
                "range": [0, len(unique_values)],
                "values": unique_values.tolist(),
                "min": min_value,
                "importance": exp.clf.feature_importances_[i],
                "dtype": "category",
            })
            feature_type[i] = "category"
        else:
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
        feature_type[i] = "number"
        feature_index[i] = [len(features) - 1, 0]
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

for path in paths:
    if not path.get('represent', True):
        continue
    new_range = {}
    for index in path['range']:
        i, j = feature_index[index]
        if feature_type[index] == 'number':
            r = path['range'][index]
            key = features[i]['name']
            if exp.data_table[key].dtype == np.int64:
                if r[0] < 0:
                    r[0] = 0
                if r[1] > features[i]['range'][1]:
                    r[1] = features[i]['range'][1]
                if features[index]['range'][0] > 0:
                    if r[0] < int(r[0]) + 1e-7:
                        r[0] = int(r[0]) - 1
                    else:
                        r[0] = int(r[0])
                    if r[1] > int(r[1]) + 1e-7:
                        r[1] = int(r[1])
                else:
                    if r[0] > int(r[0]) + 1e-7:
                        r[0] = int(r[0]) + 0.5
                    if r[1] > int(r[1]) + 1e-7:
                        r[1] = int(r[1]) + 0.5
            new_range[i] = r
        else:
            key = features[i]['name']
            if 'min' in features[i] and key in ['status', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex']:
                new_range[i] = [0] * features[i]['range'][1]
                min_value = features[i]['min']
                r = path['range'][index]
                for j in range(features[i]['range'][1]):
                    if j + min_value >= r[0] and j + min_value <= r[1]:
                        new_range[i][j] = 1
            else:
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
    'model_info': {
        'accuracy': exp._accuracy[-1],
        'info': level_info,
        'num_of_rules': len(paths),
        'dataset': 'German Credit',
        'model': 'Random Forest',
    }
}

import pickle
pickle.dump(output_data, open('output/german0315v2.pkl', 'wb'))