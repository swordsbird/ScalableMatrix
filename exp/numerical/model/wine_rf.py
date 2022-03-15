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

    random_state = 10
    data_table = pd.read_csv(os.path.join(project_path, 'data/winequality-red.csv'))

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

    return clf, (X_train,y_train, X_test, y_test), 'wine', 'rf', parameters

if __name__ == '__main__':
    get_model()