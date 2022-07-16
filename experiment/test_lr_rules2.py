import sys
import math
import numpy as np
import sys
sys.path.append('..')
from lib.model_utils import export_rules_to_csv
import pandas as pd

from lib.model_utils import ModelUtil
model = ModelUtil(
    data_name = 'german_credit', 
    model_name = 'RF',
    parameters = {
        'n_estimators': 150,
        'max_depth': 10,
        'random_state': 190,
        'max_features': 'auto',
        'min_samples_split': 12,
        'min_samples_leaf': 4,
    })
paths = model.paths

X, y = model.get_rule_matrix()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X[:], y[:])
y_pred = lr.predict(X[:])
from sklearn.metrics import accuracy_score, precision_score, f1_score
print('Fitting on all rules')
print('Accuracy Score is', accuracy_score(y[:], y_pred))
print('Precision Score is', precision_score(y[:], y_pred))
print('F1 Score is', f1_score(y[:], y_pred))

for i, coef in enumerate(lr.coef_[0]):
    print("%s %.4f" % (model.feature_name[i], coef))

predict_proba = lr.predict_proba(X)
score = np.array([predict_proba[i, int(1 - y[i])] for i in range(predict_proba.shape[0])])
export_rules_to_csv('csv/0716_LR_setting_3_top200', model, score.argsort()[-200:][::-1])
