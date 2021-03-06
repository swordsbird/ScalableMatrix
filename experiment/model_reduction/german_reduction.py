from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import os
import sys
sys.path.append('../..')
from lib.tree_extractor import path_extractor
from lib.model_reduction import Extractor
from lib.exp_model import ExpModel
import pickle

from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

random_state = 190

dataset = 'german_credit'
model = 'RF'
exp = ExpModel(dataset, model)
exp.init()
exp.train()
paths = exp.generate_paths()
exp.evaluate()

from lib.tree_extractor import assign_samples
assign_samples(paths, (exp.X, exp.y))
paths = [p for p in paths if p['coverage'] > 0]

n = 80
curves = []

tau = exp.parameters['n_estimators'] * n / len(paths) * 0.5
lambda_ = 0.01
ex = Extractor(paths, exp.X_train, exp.clf.predict(exp.X_train))
while lambda_ < 40:
    w, _, fidelity_train, result = ex.extract(n, tau, lambda_)
    [idx] = np.nonzero(w)

    accuracy_train = ex.evaluate(w, exp.X_train, exp.y_train)
    accuracy_test = ex.evaluate(w, exp.X_test, exp.y_test)
    fidelity_train = ex.evaluate(w, exp.X_train, exp.clf.predict(exp.X_train))
    fidelity_test = ex.evaluate(w, exp.X_test, exp.clf.predict(exp.X_test))
    obj, first_term, second_term = result
    second_term /= lambda_
    curves.append((lambda_, first_term, 'fidelity'))
    curves.append((lambda_, second_term, 'score'))
    curves.append((lambda_, obj, 'obj'))
    f = open('../../output/data/reduction_minimize_0531.txt', 'a')
    f.write('%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (lambda_, first_term, second_term, obj, fidelity_train, fidelity_test))
    f.close()
    lambda_ += 1

df = pd.DataFrame({
    'x': [t[0] for t in curves],
    'y': [t[1] for t in curves],
    'label': [t[2] for t in curves],
})
plt.figure(figsize=(15, 10))
sns.lineplot(data=df, x='x', y='y', hue='label', markers=True)
plt.savefig('../../output/figures/reduction_minimize_0531.png')
