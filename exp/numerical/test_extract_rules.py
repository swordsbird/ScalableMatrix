get_model_funcs = []

'''
from model.cancer_rf import get_model
get_model_funcs.append(get_model)
from model.german_rf import get_model
get_model_funcs.append(get_model)
from model.bank_rf import get_model
get_model_funcs.append(get_model)

from model.cancer_lgbm import get_model
get_model_funcs.append(get_model)
from model.wine_lgbm import get_model
get_model_funcs.append(get_model)
from model.german_lgbm import get_model

get_model_funcs.append(get_model) 
from model.wine_rf import get_model
get_model_funcs.append(get_model)
from model.bank_lgbm import get_model
get_model_funcs.append(get_model)
'''
from model.abalone_rf import get_model
get_model_funcs.append(get_model)
from model.abalone_lgbm import get_model
get_model_funcs.append(get_model)


from tree_extractor import path_extractor
from model_extractor import Extractor
import numpy as np
import pickle

for get_model in get_model_funcs:
    clf, (X_train, y_train, X_test, y_test), dataset, model, parameters = get_model()

    print('dataset', dataset, 'model', model)
    if model == 'rf':
        paths = path_extractor(clf, 'random forest', (X_train, y_train))
    else:
        paths = path_extractor(clf, 'lightgbm')
    print('number of rules', len(paths))
    ex = Extractor(paths, X_train, clf.predict(X_train))
    data = {
        'paths': ex.paths,
        'X': X_train,
        'y': y_train,
    }
    pickle.dump(data, open('%s_%s.pkl' % (dataset, model), 'wb'))
