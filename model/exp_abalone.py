from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sample import create_sampler
from tree_extractor import path_extractor
from model_extractor import Extractor
import pickle

random_state = 114
parameters = {
    'n_estimators': 80,
    # 'max_depth': 30,
    'random_state': 10,
    'max_features': 'auto',
    'oob_score': True,
    'min_samples_split': 9,
    'min_samples_leaf': 5,
}

target = 'Rings'
dataset = 'abalone'
model = 'RF'

data_table = pd.read_csv('data/abalone.csv')
X = data_table.drop(target, axis=1).values
y = data_table[target].values
y = np.array([0 if v <= 7 else 1 for v in y])

sm = SMOTE(random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)

clf = RandomForestClassifier(**parameters)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
features = data_table.drop(target, axis=1).columns.to_list()

output_data = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
}

print('Accuracy Score is', output_data['accuracy'])
print('Precision is', output_data['precision'])
print('Recall is', output_data['recall'])
print('F1 Score is', output_data['f1_score'])

is_continuous = []
is_categorical = []
is_integer = []

for feature in data_table.columns:
    values = data_table[feature].values
    if feature == target:
        continue
    if data_table[feature].dtype == 'O':
        is_continuous.append(False)
        is_categorical.append(True)
    else:
        is_continuous.append(True)
        is_categorical.append(False)
    is_integer.append(False)
sampler = create_sampler(X_train, is_continuous, is_categorical, is_integer)
X2 = sampler(len(X) * 2)

if model == 'RF':
    paths = path_extractor(clf, 'random forest', (X_train, y_train))
else:
    paths = path_extractor(clf, 'lightgbm')
print('num of paths', len(paths))

num_of_rules = [50, 100, 200, 400, 800, 1600]
tau_of_rules = [1, 1, 1.5, 1.75, 2.0, 2.5]

has_file = os.path.exists('summary.csv')
f = open('summary.csv', 'a')
if not has_file:
    f.write('dataset,oversampling,num_of_rules,tau,accuracy_train,fidelity_train,accuracy_test,fidelity_test\n')
f.close()

output_data['paths'] = paths
output_data['records'] = []

ex = Extractor(paths, X_train, clf.predict(X_train))
for it in range(len(num_of_rules)):
    n = num_of_rules[it]
    tau = tau_of_rules[it]
    w, _, fidelity_train = ex.extract(n, tau)
    [idx] = np.nonzero(w)
    vec = []
    for i in idx:
        vec.append((i, float(w[i])))
    accuracy_train = ex.evaluate(w, X_train, y_train)
    accuracy_test = ex.evaluate(w, X_test, y_test)
    fidelity_test = ex.evaluate(w, X_test, clf.predict(X_test))
    f = open('summary.csv', 'a')
    f.write('%s,%s,%s,%s,%s,%s,%s,%s\n'%(dataset,0,n,tau,accuracy_train,fidelity_train,accuracy_test,fidelity_test))
    f.close()
    output_data['records'].append({
        'n': n,
        'tau': tau,
        'oversampling': 0,
        'weights': vec,
    })

ex = Extractor(paths, X2, clf.predict(X2))
for it in range(len(num_of_rules)):
    n = num_of_rules[it]
    tau = tau_of_rules[it]
    w, _, fidelity_train = ex.extract(n, tau)
    [idx] = np.nonzero(w)
    vec = []
    for i in idx:
        vec.append((i, float(w[i])))
    accuracy_train = ex.evaluate(w, X_train, y_train)
    accuracy_test = ex.evaluate(w, X_test, y_test)
    fidelity_test = ex.evaluate(w, X_test, clf.predict(X_test))
    f = open('summary.csv', 'a')
    f.write('%s,%s,%s,%s,%s,%s,%s,%s\n'%(dataset,1,n,tau,accuracy_train,fidelity_train,accuracy_test,fidelity_test))
    f.close()
    output_data['records'].append({
        'n': n,
        'tau': tau,
        'oversampling': 1,
        'weights': vec,
    })

pickle.dump(output_data, open('output/%s.pkl'%(dataset), 'wb'))