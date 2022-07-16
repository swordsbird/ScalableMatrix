import numpy as np
import sys
sys.path.append('..')
from lib.model_utils import ModelUtil
from lib.model_reduction_variant import Extractor
from lib.anomaly_detection import LOCIMatrix
import pickle

def generate_model_paths(dataset, model_name, k_fold = 0):
    model = ModelUtil(data_name = dataset, model_name = model_name)
    paths = model.paths
    mat = model.get_cover_matrix(model.X, fuzzy = True)
    res = LOCIMatrix(mat, alpha = 0.8, metric = 'euclidean')
    res.run()
    res.select_indice(11.5)
    score = (res.outlier_score > 0) * res.outlier_score
    for i, val in enumerate(score):
        model.paths[i]['score'] = val
        model.paths[i]['cost'] = val
    return model

def param_xi_search(model, paths, min_value, max_value, step, n = 80):
    alpha = model.parameters['n_estimators'] * n / len(paths)
    best_fidelity_test = 0
    best_xi = 0
    xi = 0
    ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))
    while xi <= max_value:
        w, _, fidelity_train, result = ex.extract(n, xi * alpha, 0)
        [idx] = np.nonzero(w)
        accuracy_test = ex.evaluate(w, model.X_test, model.y_test)
        fidelity_test = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))

        if fidelity_test > best_fidelity_test:
            best_fidelity_test = fidelity_test
            best_xi = xi
        else:
            break
        xi += step
    return best_xi

def param_lambda_search(model, paths, min_value, max_value, step, xi = 0.5, n = 80):
    curves = []
    alpha = model.parameters['n_estimators'] * n / len(paths)
    lambda_ = min_value
    ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))

    w, _, fidelity_train, result = ex.extract(n, xi * alpha, lambda_)
    [idx] = np.nonzero(w)
    best_fidelity_test = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))
    best_lambda = 0
    while lambda_ <= max_value:
        w, _, fidelity_train, result = ex.extract(n, xi * alpha, lambda_)
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

        if fidelity_test > best_fidelity_test * 0.99:
            best_lambda = lambda_
        else:
            break
        lambda_ += step
    return best_lambda

def generate_hierarchy(dataset, model_name, n = 80, n_fold = 4):

    '''
    xis = []
    for k_fold in range(n_fold):
        model, paths = generate_model_paths(dataset, model_name, k_fold)
        xi = param_xi_search(model, paths, 0.1, 1.5, 0.1)
        xis.append(xi)
    xi = np.mean(xis)

    model, paths = generate_model_paths(dataset, model_name)
    lambda_ = param_lambda_search(model, paths, 0.1, 50, 1, xi, n)
    '''
    model = generate_model_paths(dataset, model_name)
    paths = model.paths
    xi = 0.5
    lambda_ = .6
    alpha = model.parameters['n_estimators'] * n / len(paths)
    print('xi', xi)
    print('lambda', lambda_)
    ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))
    w, _, fidelity_train, obj = ex.extract(n, xi * alpha, lambda_)
    [idx] = np.nonzero(w)

    accuracy_test = ex.evaluate(w, model.X_test, model.y_test)
    fidelity_test = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))

    model.accuracy = accuracy_test
    model.fidelity = fidelity_test
    level_info = {
        'fidelity_test': fidelity_test,
        'accuracy_test': accuracy_test,
        'xi': xi,
        'lambda_': lambda_,
    }
    print('fidelity_test', fidelity_test)
    print('accuracy_test', accuracy_test)
    #curr_paths = [paths[i] for i in idx]
    return model, paths, level_info, idx

def post_process(dataset, model, paths, level_info, selected_idx):
    idx = selected_idx
    new_feature = {}
    features = [feature for feature in model.data_table.columns if feature != model.target]
    for index, feature in enumerate(features):
        if ' - ' in feature:
            name, p = feature.split(' - ')
            p = int(p)
            if name not in new_feature:
                new_feature[name] = []
            while p >= len(new_feature[name]):
                new_feature[name].append(-1)
            new_feature[name][p] = index
        else:
            new_feature[feature] = [index]

    if dataset == 'german_credit':
        features = []
        feature_index = {}
        feature_type = {}
        for key in new_feature:
            if len(new_feature[key]) == 1:
                i = new_feature[key][0]
                if key in ['status', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex']:
                    min_value = min(model.data_table[key].values)
                    max_value = max(model.data_table[key].values)
                    unique_values = np.unique(model.data_table[key].values) - min_value
                    sorted(unique_values)
                    features.append({
                        "name": key,
                        "range": [0, len(unique_values)],
                        "values": unique_values.tolist(),
                        "min": min_value,
                        "importance": model.clf.feature_importances_[i],
                        "dtype": "category",
                    })
                    feature_type[i] = "category"
                else:
                    values = model.data_table[key].values
                    values.sort()
                    n = len(values)
                    qmin, qmax = values[0], values[-1]
                    q5, q25, q50, q75, q95 = values[n * 5 // 100], values[n * 25 // 100], values[n * 50 // 100], values[n * 75 // 100], values[n * 95 // 100]
                    features.append({
                        "name": key,
                        "quantile": { "5": q5, "25": q25, "50": q50, "75": q75, "95": q95 },
                        "range": [qmin, qmax],
                        "importance": model.clf.feature_importances_[i],
                        "dtype": "number",
                    })
                feature_type[i] = "number"
                feature_index[i] = [len(features) - 1, 0]
            else:
                features.append({
                    "name": key,
                    "range": [0, len(new_feature[key])],
                    "importance": sum([model.clf.feature_importances_[i] for i in new_feature[key] if i != -1]),
                    "dtype": "category",
                })

                for index, i in enumerate(new_feature[key]):
                    if i != -1:
                        feature_index[i] = [len(features) - 1, index]
                        feature_type[i] = "category"

        for path in paths:
            if not path.get('represent', True):
                continue
            new_range = {}
            for index in path['range']:
                i, j = feature_index[index]
                if feature_type[index] == 'number':
                    r = path['range'][index]
                    key = features[i]['name']
                    if model.data_table[key].dtype == np.int64:
                        if r[0] < 0:
                            r[0] = 0
                        if r[1] > features[i]['range'][1]:
                            r[1] = features[i]['range'][1]
                        if features[index]['range'][0] > 0:
                            if r[0] < int(r[0]) + 1e-7:
                                r[0] = int(r[0]) - 1
                            else:
                                r[0] = int(r[0])
                            if r[1] > int(r[1]) + 1e-7:
                                r[1] = int(r[1])
                        else:
                            if r[0] > int(r[0]) + 1e-7:
                                r[0] = int(r[0]) + 0.5
                            if r[1] > int(r[1]) + 1e-7:
                                r[1] = int(r[1]) + 0.5
                    new_range[i] = r
                else:
                    key = features[i]['name']
                    if 'min' in features[i] and key in ['status', 'savings', 'employment_duration', 'installment_rate', 'personal_status_sex']:
                        new_range[i] = [0] * features[i]['range'][1]
                        min_value = features[i]['min']
                        r = path['range'][index]
                        for j in range(features[i]['range'][1]):
                            if j + min_value >= r[0] and j + min_value <= r[1]:
                                new_range[i][j] = 1
                    else:
                        if i not in new_range:
                            new_range[i] = [0] * features[i]['range'][1]
                            if path['range'][index][0] <= 1 and 1 <= path['range'][index][1]:
                                new_range[i][j] = 1
                            else:
                                for k in range(len(new_range[i])):
                                    if k != j:
                                        new_range[i][k] = 1
                                    new_range[i][j] = 0
            path['range'] = new_range
            path['represent'] = False

        for i in idx:
            paths[i]['represent'] = True

        output_data = {
            'paths': paths,
            'features': features,
            'selected': [paths[i]['name'] for i in idx],
            'model_info': {
                'accuracy': model.accuracy,
                'info': level_info,
                'num_of_rules': len(paths),
                'dataset': 'German Credit',
                'model': 'Random Forest',
            }
        }
    return output_data

dataset = 'german_credit'
model_name = 'RF'
model, paths, level_info, idx = generate_hierarchy(dataset, model_name)
data = post_process(dataset, model, paths, level_info, idx)

import pickle
pickle.dump(data, open('../output/german_0707.pkl', 'wb'))
