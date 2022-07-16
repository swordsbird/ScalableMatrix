import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from lib.model_reduction_variant import Extractor
from lib.anomaly_detection import LOCIMatrix
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

from lib.model_utils import ModelUtil
model = ModelUtil(data_name = 'german', model_name = 'random forest')
paths = model.paths
mat = model.get_cover_matrix(model.X, fuzzy = True)
res = LOCIMatrix(mat, alpha = 0.8, metric = 'euclidean')
res.run()
res.select_indice(11.5)

score = np.max(res.outlier_score, 0)
for i, val in enumerate(score):
    paths[i]['score'] = 1.0 / np.exp(val)

curves = []

n = 80
tau = model.parameters['n_estimators'] * n / len(paths) * 0.5
lambda_ = 0.01
ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))
while lambda_ < 40:
    w, _, fidelity_train, result = ex.extract(n, tau, lambda_)
    [idx] = np.nonzero(w)

    accuracy_train = ex.evaluate(w, model.X_train, model.y_train)
    accuracy_test = ex.evaluate(w, model.X_test, model.y_test)
    fidelity_train = ex.evaluate(w, model.X_train, model.clf.predict(model.X_train))
    fidelity_test = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))
    obj, first_term, second_term = result
    second_term /= lambda_
    curves.append((lambda_, first_term, 'fidelity'))
    curves.append((lambda_, second_term, 'score'))
    curves.append((lambda_, obj, 'obj'))
    f = open('../output/data/record_0707.txt', 'a')
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
plt.savefig('curve2.png')
