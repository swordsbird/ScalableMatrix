from rulematrix.surrogate import rule_surrogate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris

import numpy as np

random_state = 190

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

target = 'credit_risk'
#random_state = 24
parameters = {
    'random_state': random_state,
    'max_depth': 12,
    'n_estimators': 150,
    'max_leaf_nodes': 100,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'bootstrap': True,
}

data_table = pd.read_csv('data/german.csv')
qualitative_features = [
    'credit_history', 'purpose', 'other_debtors', 
    'property', 'other_installment_plans', 
    'housing', 'job', 'people_liable', 'telephone',
    'foreign_worker', 'number_credits',
]
for feature in qualitative_features:
    unique_values = np.unique(data_table[feature].values)
    sorted(unique_values)
    if int(unique_values[0]) == 0:
        for i in unique_values:
            data_table[feature + ' - '+ str(i)] = data_table[feature].values == i
    else:
        for i in unique_values:
            data_table[feature + ' - '+ str(int(i) - 1)] = data_table[feature].values == i
data_table['personal_status_sex'] = 1 * (data_table['personal_status_sex'].values == 3)
for feature in qualitative_features:
    data_table = data_table.drop(feature, axis = 1)
#data_table = data_table.drop('Other installment plans', axis = 1)
X = data_table.drop(target, axis=1).values
y = data_table[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)

sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)

parameters = {
    'n_estimators': 200,
    'max_depth': 20,
    'random_state': random_state,
    'max_features': 'auto',
    'oob_score': True,
    'min_samples_split': 20,
    'min_samples_leaf': 8,
}
# Test 0.8425938204482164
# Train 0.9906159910572454
clf = RandomForestClassifier(**parameters)
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