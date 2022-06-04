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

random_state = 10

def get_model():
    project_path = '/home/lizhen/projects/extree/exp'
    data_table = pd.read_csv(os.path.join(project_path, 'data/cancer.csv'))

    labels = data_table['diagnosis']
    labels = labels.apply(lambda x: 1 if x == 'M' else 0)
    data_table = data_table.drop(['diagnosis', 'id'], axis=1)
    X = data_table.values
    y = labels.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=labels)

    parameters = {'n_estimators': 400, 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 1}

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

    return clf, (X_train,y_train, X_test, y_test), 'cancer', 'rf', parameters

if __name__ == '__main__':
    get_model()