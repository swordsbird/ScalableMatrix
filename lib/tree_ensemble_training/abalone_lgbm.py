from rulematrix.surrogate import rule_surrogate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd

random_state = 10

def get_model():
    project_path = '/home/lizhen/projects/extree/exp'
    data = pd.read_csv(os.path.join(project_path, 'data/abalone.csv'))

    category = np.repeat(0, data.shape[0])
    for i in range(0, data["Rings"].size):
        if data["Rings"][i] <= 7:
            category[i] = 0
        elif data["Rings"][i] > 7:
            category[i] = 1

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data['Sex'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    data = data.drop(['Sex'], axis=1)
    data['category_size'] = category
    data = data.drop(['Rings'], axis=1)

    features = data.iloc[:, np.r_[0:7]]
    labels = data.iloc[:, 7]

    X_train, X_test, y_train, y_test, X_gender, X_gender_test = \
        train_test_split(features, labels, onehot_encoded, random_state=10, test_size=0.2)

    temp = X_train.values
    X_train_gender = np.concatenate((temp, X_gender), axis=1)
    X_train = X_train_gender

    temp = X_test.values
    X_test_gender = np.concatenate((temp, X_gender_test), axis=1)
    X_test = X_test_gender

    parameters = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'num_leaves': 25,
        'max_depth': 5,
        'min_data_in_leaf': 200,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
    }

    clf = LGBMClassifier(**parameters)
    clf.fit(X_train_gender, y_train)

    y_pred = clf.predict(X_test)

    print('Test')
    print('Accuracy Score is', accuracy_score(y_test, y_pred))
    print('Precision Score is', precision_score(y_test, y_pred))
    print('F1 Score is', f1_score(y_test, y_pred))

    y_pred = clf.predict(X_train)

    print('Train')
    print('Accuracy Score is', accuracy_score(y_train, y_pred))

    return clf, (X_train,y_train, X_test, y_test), 'abalone', 'lgbm', parameters

if __name__ == '__main__':
    get_model()