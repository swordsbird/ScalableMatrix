get_model_funcs = []

from model.cancer_rf import get_model
get_model_funcs.append(get_model)
from model.german_rf import get_model
get_model_funcs.append(get_model)
from model.wine_rf import get_model
get_model_funcs.append(get_model)
from model.bank_rf import get_model
get_model_funcs.append(get_model)
#from model.abalone_rf import get_model
#get_model_funcs.append(get_model)

from model.cancer_lgbm import get_model
get_model_funcs.append(get_model)
from model.wine_lgbm import get_model
get_model_funcs.append(get_model)
from model.bank_lgbm import get_model
get_model_funcs.append(get_model)
from model.german_lgbm import get_model
get_model_funcs.append(get_model) 
#from model.abalone_lgbm import get_model
#get_model_funcs.append(get_model)


from tree_extractor import path_extractor
from model_extractor import Extractor
import numpy as np

for get_model in get_model_funcs:
    clf, (X_train, y_train, X_test, y_test), dataset, model, parameters = get_model()

    print('dataset', dataset, 'model', model)
    if model == 'rf':
        paths = path_extractor(clf, 'random forest', (X_train, y_train))
    else:
        paths = path_extractor(clf, 'lightgbm')
    print('number of rules', len(paths))
    ex = Extractor(paths, X_train, clf.predict(X_train))
    f = open('anomaly.csv', 'a')
    f.write('\n%s,%s\n'%(dataset, model))

    for key in ex.weights:
        outlier, score = ex.weights[key]
        outlier_num = np.sum(outlier)
        s = ''
        scores = np.quantile(score, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        for x in scores:
            s += '%.4f,' % (x)
        s = s[:-1]
        f.write('\n%s, %d / %d, %s\n'%(key, outlier_num, len(outlier), s))
    f.close()
    print('-' * 20)
