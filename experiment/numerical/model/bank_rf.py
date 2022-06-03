from rulematrix.surrogate import rule_surrogate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris

import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def get_model():
    project_path = '/home/lizhen/projects/extree/exp'
    data_table = pd.read_csv(os.path.join(project_path, 'data/bank.csv'))

    random_state = 10
    corr = data_table.corr()
    n_cols = len(data_table.columns)
    dropped = []
    for i in range(n_cols):
        k1 = data_table.columns[i]
        for k2 in data_table.columns[i + 1:]:
            if k1 != k2 and abs(corr[k1][k2]) > 0.9:
                dropped.append(k2)
    dropped = set(dropped)
    dropped = [k for k in dropped]
    dropped += ['Net Income Flag', 'Liability-Assets Flag']
    data_table = data_table.drop(dropped, axis = 1)
    print(len(data_table.columns))

    X = data_table.drop('Bankrupt?', axis=1).values
    y = data_table['Bankrupt?'].values
    features = data_table.drop('Bankrupt?', axis=1).columns.to_list()

    parameters = {
            'n_estimators': 150,
            'max_depth': 30,
            'random_state': random_state,
    }

    clf = RandomForestClassifier(**parameters)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state)

    sm = SMOTE(random_state=random_state)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('Test')
    print('Accuracy Score is', accuracy_score(y_test, y_pred))
    print('Precision Score is', precision_score(y_test, y_pred))
    print('F1 Score is', f1_score(y_test, y_pred))

    y_pred = clf.predict(X_train)

    print('Train')
    print('Accuracy Score is', accuracy_score(y_train, y_pred))

    return clf, (X_train,y_train, X_test, y_test), 'bankrupt', 'rf', parameters

if __name__ == '__main__':
    get_model()