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

def train_surrogate(model, sampling_rate=2.0, n_rules=20, **kwargs):
    is_categorical = []
    is_continuous = []
    for i in range(X_train.shape[1]):
        values = np.unique(X_train[:, i])
        if len(values) < 10:
            is_categorical.append(True)
            is_continuous.append(False)
        else:
            is_categorical.append(False)
            is_continuous.append(True)
    is_categorical = None#np.array(is_categorical)
    is_continuous = np.array(is_continuous)
    surrogate = rule_surrogate(model.predict,
                               X_train,
                               sampling_rate=sampling_rate,
                               is_continuous=None,
                               is_categorical=None,
                               is_integer=None,
                               number_of_rules=n_rules,
                               **kwargs)

    #train_fidelity = surrogate.score(X_train)
    test_fidelity = surrogate.score(X_test)
    test_pred = surrogate.student.predict(X_test)
    test_accuracy = np.sum(test_pred == y_test) / len(y_test)
    print('fidelity:', round(test_fidelity, 4))
    print('accuracy:', round(test_accuracy, 4))
    return surrogate

for n_rules in [40, 80, 160, 320, 640]:
    print('number of rules', n_rules)
    surrogate = train_surrogate(clf, -1, n_rules, seed=random_state)