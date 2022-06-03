get_model_funcs = []

from model.cancer_lgbm import get_model
get_model_funcs.append(get_model)
from model.cancer_rf import get_model
get_model_funcs.append(get_model)

from model.wine_lgbm import get_model
get_model_funcs.append(get_model)
from model.wine_rf import get_model
get_model_funcs.append(get_model)

from model.bank_lgbm import get_model
get_model_funcs.append(get_model)
from model.bank_rf import get_model
get_model_funcs.append(get_model)

from model.german_lgbm import get_model
get_model_funcs.append(get_model)
from model.german_rf import get_model
get_model_funcs.append(get_model)


from model.abalone_lgbm import get_model
get_model_funcs.append(get_model)
from model.abalone_rf import get_model
get_model_funcs.append(get_model)


from tree_extractor import path_extractor
from sbrl_extractor import make_sbrl, test_sbrl
from rulematrix.surrogate import upsampling
import random

random_state = 10

def data_upsampling(model, sampling_rate=2.0, n_rules=20, **kwargs):
    X = upsampling(model.predict,
                            X_train,
                            sampling_rate=sampling_rate,
                            is_continuous=None,
                            is_categorical=None,
                            is_integer=None,
                            number_of_rules=n_rules,
                            **kwargs)
    return X

for get_model in get_model_funcs:
    clf, (X_train, y_train, X_test, y_test), dataset, model = get_model()

    print('dataset', dataset, 'model', model)
    if model == 'rf':
        paths = path_extractor(clf, 'random forest', (X_train, y_train))
    else:
        paths = path_extractor(clf, 'lightgbm')
    if len(paths) > 30000:
        paths = sorted(paths, key = lambda x: -x['confidence'])
        paths = paths[:30000]
    if len(paths) > 15000:
        paths = random.sample(paths, 15000)
    print('number of rules', len(paths))
    sampling_rate = 4
    if dataset == 'bankrupt' or dataset == 'abalone':
        sampling_rate = -1
    X = data_upsampling(clf, sampling_rate, 100, seed=random_state)
    y = clf.predict(X)
    for n in [40, 80, 160, 320, 640]:
        sbrl = make_sbrl(paths, X, y, n)
        accuracy_test = test_sbrl(sbrl, X_test, y_test, True)
        fidelity_test = test_sbrl(sbrl, X_test, clf.predict(X_test), True)
        print('n', n, 'fidelity', round(fidelity_test, 4))
        f = open('sbrl.csv', 'a')
        f.write('%s,%s,%s,%s,%s\n'%(dataset, model, n, fidelity_test, accuracy_test))
        f.close()
    print('-' * 20)