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
from imblearn.over_sampling import SMOTE

def get_model():
    project_path = '/home/lizhen/projects/extree/exp'

    random_state = 190
    target = 'credit_risk'
    data_table = pd.read_csv(os.path.join(project_path, 'data/german.csv'))

    parameters = {'n_estimators': 100, 'learning_rate': 0.02, 'max_depth': 10}

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


    clf = LGBMClassifier(**parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Test')
    print('Accuracy Score is', accuracy_score(y_test, y_pred))
    print('Precision Score is', precision_score(y_test, y_pred))
    print('F1 Score is', f1_score(y_test, y_pred))

    y_pred = clf.predict(X_train)

    print('Train')
    print('Accuracy Score is', accuracy_score(y_train, y_pred))

    return clf, (X_train,y_train, X_test, y_test), 'german', 'lgbm', parameters

if __name__ == '__main__':
    get_model()