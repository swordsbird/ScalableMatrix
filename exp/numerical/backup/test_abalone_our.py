from rulematrix.surrogate import rule_surrogate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris

import numpy as np
import random

random_state = 10

# abalone RF model
# Test 0.9055023923444976
# Train 0.9428314875785693
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

data = pd.read_csv("data/abalone.csv", sep=",", header='infer')

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
    # 'lambda_l1': 0.65,
    # 'lambda_l2': 2,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'feature_fraction': 0.9,
}

clf = LGBMClassifier(**parameters)
clf.fit(X_train_gender, y_train)

y_pred = clf.predict(X_test_gender)

print('Test')
print('Accuracy Score is', accuracy_score(y_test, y_pred))

y_pred = clf.predict(X_train_gender)

print('Train')
print('Accuracy Score is', accuracy_score(y_train, y_pred))

from tree_extractor import path_extractor
paths = path_extractor(clf, 'lightgbm')
#paths = path_extractor(clf, 'random forest', (X_train, y_train))
print('number of rules', len(paths))
if len(paths) > 10000:
    paths = [x for x in paths if x['confidence'] > 0.8]
if len(paths) > 10000:
    paths = random.sample(paths, 10000)
#paths = [x for x in paths if x['confidence'] > 0.8]


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
