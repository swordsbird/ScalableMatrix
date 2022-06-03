get_model_funcs = []

from model.cancer_rf import get_model
get_model_funcs.append(get_model)
from model.german_rf import get_model
get_model_funcs.append(get_model)
from model.wine_rf import get_model
get_model_funcs.append(get_model)
from model.bank_rf import get_model
get_model_funcs.append(get_model)
from model.abalone_rf import get_model
get_model_funcs.append(get_model)

from model.cancer_lgbm import get_model
get_model_funcs.append(get_model)
from model.wine_lgbm import get_model
get_model_funcs.append(get_model)
from model.bank_lgbm import get_model
get_model_funcs.append(get_model)
from model.german_lgbm import get_model
get_model_funcs.append(get_model)
from model.abalone_lgbm import get_model
get_model_funcs.append(get_model)


from tree_extractor import path_extractor
from model_extractor import Extractor

for get_model in get_model_funcs:
    clf, (X_train, y_train, X_test, y_test), dataset, model, parameters = get_model()

    print('dataset', dataset, 'model', model)
    if model == 'rf':
        paths = path_extractor(clf, 'random forest', (X_train, y_train))
    else:
        paths = path_extractor(clf, 'lightgbm')
    print('number of rules', len(paths))
    ex = Extractor(paths, X_train, clf.predict(X_train))
    for n in [40, 80, 160, 320, 640]:
        tau = parameters['n_estimators'] * n / len(paths) 
        lambda_ = .2
        w, _, fidelity_train = ex.extract(n, tau, lambda_)
        accuracy_test = ex.evaluate(w, X_test, y_test)
        fidelity_test = ex.evaluate(w, X_test, clf.predict(X_test))
        print('n', n, 'fidelity', round(fidelity_test, 4))
        f = open('our.csv', 'a')
        f.write('%s,%s,%s,%s,%s\n'%(dataset, model, n, fidelity_test, accuracy_test))
        f.close()
    print('-' * 20)