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
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 190,
        'max_features': 'auto',
        'min_samples_split': 10,
        'min_samples_leaf': 5,
    })
paths = model.paths
high_conf_idxes = [i for i, p in enumerate(paths) if paths[i]['confidence'] > 0.8]

X, y = model.get_rule_matrix()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X[high_conf_idxes], y[high_conf_idxes])
y_pred = lr.predict(X[high_conf_idxes])
from sklearn.metrics import accuracy_score, precision_score, f1_score
print('Fitting on rules with confidence > 0.8')
print('Accuracy Score is', accuracy_score(y[high_conf_idxes], y_pred))
print('Precision Score is', precision_score(y[high_conf_idxes], y_pred))
print('F1 Score is', f1_score(y[high_conf_idxes], y_pred))

for i, coef in enumerate(lr.coef_[0]):
    print("%s %.4f" % (model.feature_name[i], coef))

predict_proba = lr.predict_proba(X)
score = np.array([predict_proba[i, int(1 - y[i])] for i in range(predict_proba.shape[0])])
export_rules_to_csv('csv/0716_LR_setting_2_top200', model, score.argsort()[-200:][::-1])
