get_model_funcs = []

from model.german_rf import get_model
get_model_funcs.append(get_model)
from model.german_lgbm import get_model
get_model_funcs.append(get_model)

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

from model.abalone_lgbm import get_model
get_model_funcs.append(get_model)
from model.abalone_rf import get_model
get_model_funcs.append(get_model)


from tree_extractor import path_extractor
from model_extractor_maxnum import Extractor
import numpy as np
from rulematrix.surrogate import rule_surrogate
import random

random_state = 10

def train_surrogate(model, sampling_rate=2.0, n_rules=20, **kwargs):
    surrogate = rule_surrogate(model.predict,
                               X_train,
                               sampling_rate=sampling_rate,
                               is_continuous=None,
                               is_categorical=None,
                               is_integer=None,
                               number_of_rules=n_rules,
                               **kwargs)

    test_fidelity = surrogate.score(X_test)
    test_pred = surrogate.student.predict(X_test)
    test_accuracy = np.sum(test_pred == y_test) / len(y_test)
    return surrogate, test_accuracy, test_fidelity

for get_model in get_model_funcs:
    clf, (X_train, y_train, X_test, y_test), dataset, model, _ = get_model()

    print('dataset', dataset, 'model', model)
    if model == 'rf':
        paths = path_extractor(clf, 'random forest', (X_train, y_train))
    else:
        paths = path_extractor(clf, 'lightgbm')
    sampling_rate = 4
    if dataset == 'bankrupt' or dataset == 'abalone' or dataset == 'german':
        sampling_rate = -1
    for n in [40, 80, 160, 320, 640]:
        surrogate, accuracy_test, fidelity_test = train_surrogate(clf, sampling_rate, n, seed=random_state)
        print('n', n, 'fidelity', round(fidelity_test, 4))
        f = open('rulematrix.csv', 'a')
        f.write('%s,%s,%s,%s,%s\n'%(dataset, model, n, fidelity_test, accuracy_test))
        f.close()
    print('-' * 20)
