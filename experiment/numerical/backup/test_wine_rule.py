
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from rulematrix.surrogate import rule_surrogate

random_state = 10

data_table = pd.read_csv('./data/winequality-red.csv')

labels = data_table['quality']
labels = labels.apply(lambda x: 1 if x > 6 else 0)
y = labels.values
data_table = data_table.drop(['quality'], axis=1)
X = data_table.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state)

sm = SMOTE(random_state=random_state)
X_train, y_train = sm.fit_resample(X_train, y_train)

parameters = {
    'n_estimators': 200, 
    'max_depth': 15, 
    'min_samples_split': 5, 
    'min_samples_leaf': 2,
    'bootstrap': True,
}
clf = RandomForestClassifier(**parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))
print('Precision Score is', precision_score(y_test, y_pred))
print('F1 Score is', f1_score(y_test, y_pred))

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
    surrogate = train_surrogate(clf, 4, n_rules, seed=random_state)