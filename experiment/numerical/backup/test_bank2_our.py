from rulematrix.surrogate import rule_surrogate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris

import numpy as np

random_state = 10

# adult RF model
# Test 0.8425938204482164
# Train 0.9906159910572454
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

target = 'Bankrupt?'
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
X = data_table.drop(target, axis=1).values
y = data_table[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)

sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)


clf = LGBMClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))

from tree_extractor import path_extractor
paths = path_extractor(clf, 'lightgbm')
#paths = path_extractor(clf, 'random forest', (X_train, y_train))
print('number of rules', len(paths))

from model_extractor_maxnum import Extractor
ex = Extractor(paths, X_train, clf.predict(X_train))
for n in [40, 80, 160, 320, 640]:
    tau = (n / 80) ** 0.5
    w, _, fidelity_train = ex.extract(n, tau)
    accuracy_test = ex.evaluate(w, X_test, y_test)
    fidelity_test = ex.evaluate(w, X_test, clf.predict(X_test))
    print('number of rules', n)
    print('fidelity', round(fidelity_test, 4))
    print('accuracy', round(accuracy_test, 4))
