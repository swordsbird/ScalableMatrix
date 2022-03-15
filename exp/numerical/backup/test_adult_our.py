from rulematrix.surrogate import rule_surrogate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris

import numpy as np
import random

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

adults = pd.read_csv('data/adult.csv')

adults = adults.replace(to_replace="?", value=np.nan)
adults = adults.dropna()
adults = adults.reset_index(drop=True)

adults.rename(columns={"education.num": "education_num", "marital.status": "marital_status",
                       "capital.gain": "capital_gain", "capital.loss": "capital_loss",
                       "hours.per.week": "hours_per_week", "native.country":"native_country"},
              inplace=True)

adults.drop(columns=["education"], inplace=True)

numerical_columns = adults.select_dtypes(include=[np.number]).columns
categorical_columns = adults.select_dtypes(exclude=[np.number]).columns
categorical_columns = categorical_columns.difference(["income"])

scaler = StandardScaler()
encoder = OneHotEncoder(sparse=False)
std_num_adults = pd.DataFrame(scaler.fit_transform(adults[numerical_columns]), columns=numerical_columns)
ohe_cat_adults = pd.DataFrame(encoder.fit_transform(adults[categorical_columns]))

scaled_encoded_adults = pd.concat([std_num_adults, ohe_cat_adults], axis=1)

X = scaled_encoded_adults.values
y = adults["income"].apply(lambda income: 1 if income == ">50K" else 0).values

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


from tree_extractor import path_extractor
#paths = path_extractor(clf, 'lightgbm')
paths = path_extractor(clf, 'random forest', (X_train, y_train))
paths = [x for x in paths if x['confidence'] > 0.9]
if len(paths) > 10000:
    paths = random.sample(paths, 10000)
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
