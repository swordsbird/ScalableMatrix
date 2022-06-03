get_model_funcs = []

from model.cancer_rf import get_model
get_model_funcs.append(get_model)
from model.german_rf import get_model
get_model_funcs.append(get_model)
from model.wine_rf import get_model
get_model_funcs.append(get_model)
#from model.bank_rf import get_model
#get_model_funcs.append(get_model)
#from model.abalone_rf import get_model
#get_model_funcs.append(get_model)

from model.cancer_lgbm import get_model
get_model_funcs.append(get_model)
#from model.german_lgbm import get_model
#get_model_funcs.append(get_model) 
#from model.wine_lgbm import get_model
#get_model_funcs.append(get_model)
#from model.bank_lgbm import get_model
#get_model_funcs.append(get_model)
#from model.abalone_lgbm import get_model
#get_model_funcs.append(get_model)


from tree_extractor import path_extractor
from model_extractor import Extractor


for lambda_ in [k * 0.05 for k in range(20)]:
    f = open('our4.csv', 'a')
    f.write("\nlambda %s" % (lambda_))
    f.close()
    for get_model in get_model_funcs:
        clf, (X_train, y_train, X_test, y_test), dataset, model, parameters = get_model()

        print('dataset', dataset, 'model', model)
        if model == 'rf':
            paths = path_extractor(clf, 'random forest', (X_train, y_train))
        else:
            paths = path_extractor(clf, 'lightgbm')
        print('number of rules', len(paths))
        ex = Extractor(paths, X_train, clf.predict(X_train))
        for iter, n in enumerate([80]):
            tau = parameters['n_estimators'] * n / len(paths) 
            w, _, fidelity_train = ex.extract(n, parameters['n_estimators'], tau, lambda_)
            accuracy_test = ex.evaluate(w, X_test, y_test)
            fidelity_test = ex.evaluate(w, X_test, clf.predict(X_test))
            if iter == 0:    
                anomaly_min, anomaly_thres, anomaly_max, _ = ex.anomaly_info(0.02)
            avg_score = ex.anomaly_score(w)
            percent = ex.anomaly_percent(w)
            print('n', n, 'fidelity', round(fidelity_test, 4))
            f = open('our.csv', 'a')
            if iter == 0:
                f.write('\n%s,%s\n'%(dataset, model))
            f.write('%s,%.4f,%.4f,%.4f,%.4f\n'%(n, fidelity_test, accuracy_test, avg_score, percent))
            f.close()
        print('-' * 20)