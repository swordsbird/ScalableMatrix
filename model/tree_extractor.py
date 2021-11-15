
from os import path
from copy import deepcopy

def visit_boosting_tree(tree, path = {}):
    if 'decision_type' not in tree:
        path['value'] = tree['leaf_value']
        path['weight'] = tree['leaf_weight']
        return [{
            'range': path,
            'value': tree['leaf_value'],
            'weight': tree['leaf_weight'],
        }]
    
    key = tree['split_feature']
    thres = tree['threshold']
    ret = []
    leftpath = deepcopy(path)
    if key in leftpath:
        r = leftpath[key]
        leftpath[key] = [r[0], min(r[1], thres)]
    else:
        leftpath[key] = [-1e9, thres]
    ret += visit_boosting_tree(tree['left_child'], leftpath)

    rightpath = deepcopy(path)
    if key in rightpath:
        r = rightpath[key]
        rightpath[key] = [max(r[0], thres), r[1]]
    else:
        rightpath[key] = [thres, 1e9]
    ret += visit_boosting_tree(tree['right_child'], rightpath)

    return ret

def visit_decision_tree(tree, index = 0, path = {}):
    if tree.children_left[index] == -1 and tree.children_right[index] == -1:
        return [{
            'range': path,
            'value': 0,
            'weight': 1,
        }]
    key = tree.feature[index]
    thres = tree.threshold[index]
    ret = []
    leftpath = deepcopy(path)
    if key in leftpath:
        r = leftpath[key]
        leftpath[key] = [r[0], min(r[1], thres)]
    else:
        leftpath[key] = [-1e9, thres]
    ret += visit_decision_tree(tree, tree.children_left[index], leftpath)
    
    rightpath = deepcopy(path)
    if key in rightpath:
        r = rightpath[key]
        rightpath[key] = [max(r[0], thres), r[1]]
    else:
        rightpath[key] = [thres, 1e9]
    ret += visit_decision_tree(tree, tree.children_right[index], rightpath)

    return ret

def assign_value_for_random_forest(paths, data):
    X, y = data
    for path in paths:
        ans = 2 * y - 1
        m = path['range']
        for key in m:
            ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
        pos = (ans == 1).sum()
        neg = (ans == -1).sum()
        if pos + neg > 0:
            v = pos / (pos + neg) - 0.5
            path['value'] = (v ** 2) * (-1 if v < 0 else 1)
            #1 if pos > neg else -1
            if pos + neg > 0:
                path['confidence'] = max(pos, neg) / (pos + neg)
            else:
                path['confidence'] = 0
        else:
            path['value'] = 0
            path['confidence'] = 0

def path_extractor(model, model_type, data = None):
    if model_type == 'random forest' :
        ret = []
        for estimator in model.estimators_:
            ret += path_extractor(estimator, 'decision tree')
        assign_value_for_random_forest(ret, data)
        return ret
    elif model_type == 'lightgbm':
        ret = []
        info = model._Booster.dump_model()
        for tree in info['tree_info']:
            ret += visit_boosting_tree(tree['tree_structure'])
        return ret
    elif model_type == 'decision tree':
        return visit_decision_tree(model.tree_)
    return []
    