import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances
import pandas as pd
import seaborn as sns
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import math

class DetectorEnsemble:
    def __init__(self):
        self.detectors = []
        '''
        self.detectors.append(('iforest1', IsolationForest(random_state = 0, max_samples = 128, n_estimators = 100)))
        self.detectors.append(('iforest2', IsolationForest(random_state = 0, max_samples = 128, n_estimators = 200)))
        self.detectors.append(('iforest3', IsolationForest(random_state = 0, max_samples = 256, n_estimators = 100)))
        self.detectors.append(('iforest4', IsolationForest(random_state = 0, max_samples = 256, n_estimators = 200)))
        self.detectors.append(('iforest5', IsolationForest(random_state = 0, max_samples = 512, n_estimators = 100)))
        self.detectors.append(('iforest6', IsolationForest(random_state = 0, max_samples = 512, n_estimators = 200)))
        '''
        self.detectors.append(('knn', NearestNeighbors(algorithm='ball_tree')))
        self.detectors.append(('lof', LocalOutlierFactor(metric="precomputed")))
        #self.detectors.append(('robustcov', MinCovDet()))
        self.detectors.append(('iforest', IsolationForest()))
        self.detectors.append(('ocsvm', OneClassSVM()))
        self.detectors.append(('dbscan',  DBSCAN()))
    
    def fit_detector(self, X, y):
        self.clf = LinearRegression(fit_intercept=True, normalize=False, copy_X=True).fit(X, y)

    def fit(self, mat):
        dist = pairwise_distances(X = mat, metric='euclidean')
        self.scores = []
        for (name, detector) in self.detectors:
            if name[:3] == 'lof':
                detector.fit_predict(dist)
                self.scores.append(-detector.negative_outlier_factor_)
            elif name == 'robustcov':
                detector.fit(mat)
                self.scores.append(detector.mahalanobis(mat))
            elif name == 'knn':
                detector.fit(mat)
                self.scores.append(-detector.kneighbors(mat)[0][:, -1])
            elif name == 'dbscan':
                detector.fit(mat)
                score = np.array([1 if x == -1 else 0 for x in detector.labels_])
                self.scores.append(score)
            else:
                detector.fit_predict(mat)
                self.scores.append(-detector.score_samples(mat))
            print(name, min(self.scores[-1]), max(self.scores[-1]), self.scores[-1].shape)
        tmp = []
        for score in self.scores:
            min_s = np.min(score)
            max_s = np.max(score)
            range_s = max(1, max_s - min_s)
            score = (score - min_s) / range_s
            tmp.append(score)
        self.n = mat.shape[0]
        self.scores = np.array(tmp)
        self.ground_truth = {}
        self.adjust_sample_weight = self.n // 100
        self.weights = np.ones(len(self.detectors))
        weights = self.weights / np.sum(self.weights)

        self.scores = self.scores.transpose()
        y = (self.scores * weights).sum(axis = 1)
        print('before fit', self.scores.shape, y.shape)
        self.fit_detector(self.scores, y)
        print('after fit')
    
    def weighted_score(self):
        y = self.clf.predict(self.scores)
        for i in self.ground_truth:
            y[i] = self.ground_truth[i]
        return y

    def adjust_weight(self, idx, score):
        self.ground_truth[idx] = score
        sample_weight = np.ones(self.n)
        for i in self.ground_truth:
            sample_weight[i] = self.adjust_sample_weight
        y = self.weighted_score()
        self.fit_detector(self.scores, y)

current_encoding = {
    'credit_risk' : ['No', 'Yes'], 
    'credit_history' : [
        "delay in paying off in the past",
        "critical account/other credits elsewhere",
        "no credits taken/all credits paid back duly",
        "existing credits paid back duly till now",
        "all credits at this bank paid back duly",
    ],
    'purpose' : [
        "others",
        "car (new)",
        "car (used)",
        "furniture/equipment",
        "radio/television",
        "domestic appliances",
        "repairs",
        "education",
        "vacation",
        "retraining",
        "business"
    ],
    'installment_rate': ["< 20", "20 <= ... < 25",  "25 <= ... < 35", ">= 35"],
    'present_residence': [
        "< 1 yr", 
        "1 <= ... < 4 yrs",
        "4 <= ... < 7 yrs", 
        ">= 7 yrs"
    ],
    'number_credits': ["1", "2-3", "4-5", ">= 6"],
    'people_liable': ["0 to 2", "3 or more"],
    'savings': [
        "unknown/no savings account",
        "... <  100 DM", 
        "100 <= ... <  500 DM",
        "500 <= ... < 1000 DM", 
        "... >= 1000 DM",
    ],
    'employment_duration': [
        "unemployed", 
        "< 1 yr", 
        "1 <= ... < 4 yrs",
        "4 <= ... < 7 yrs", 
        ">= 7 yrs"
    ],
    'personal_status_sex': [
        "not married male",
        "married male",
    ],
    'other_debtors': [
        'none',
        'co-applicant',
        'guarantor'
    ],
    'property': [
        "real estate",
        "building soc. savings agr./life insurance", 
        "car or other",
        "unknown / no property",
    ],
    'other_installment_plans': ['bank', 'stores', 'none'],
    'housing': ["rent", "own", "for free"],
    'job': [
        'unemployed/ unskilled - non-resident',
        'unskilled - resident',
        'skilled employee / official',
        'management/ self-employed/ highly qualified employee/ officer'
    ],
    'status': [
        "no checking account",
        "... < 0 DM",
        "0<= ... < 200 DM",
        "... >= 200 DM / salary for at least 1 year",
    ],
    'telephone': ['No', 'Yes'],
    'foreign_worker': ['No', 'Yes'],
}

model = pickle.load(open('../../output/dump/german0315v2.pkl', 'rb'))
paths = model['paths']
features = model['features']
mat = np.array([p['sample'] for p in paths]).astype('float')
for i in range(mat.shape[0]):
    mat[i] = mat[i] > 0
    mat[i] /= mat[i].sum()
all_dist = pairwise_distances(X = mat, metric='euclidean')

expected_count = 50
expected_one_class_count = 40

ensemble = DetectorEnsemble()
ensemble.fit(mat)
selected_path_idxes = ensemble.weighted_score().argsort()[::-1]

output_labels = ['reject', 'accept']

def interpret_path(path, features):
    conds = []
    for key in path['range']:
        feature = features[key]
        values = path['range'][key]
        name = feature['name']
        op = 'is'
        value = ''
        if feature['dtype'] == 'category':
            if len(values) < len(feature['values']):
                t_values = [1 if (i >= values[0] and i <= values[1]) else 0 for i in range(1, len(feature['values']) + 1)]
                values = t_values
            is_negation = np.sum(values) + 1 == len(values)
            if is_negation:
                op = 'is not'
                for i, d in enumerate(values):
                    if d == 0:
                        value = feature['values'][i]
                        break
            else:
                for i, d in enumerate(values):
                    if d == 1:
                        value = value + ' or ' + feature['values'][i]
                value = value[4:]
        else:
            op = 'in'
            value = '%d ~ %d' % (values[0], values[1])
        conds.append((name, op, value))
    output_label = output_labels[path['output']]
    # print(output_labels, path['output'])
    return conds, output_label

for index, feature in enumerate(features):
    if feature['name'] in current_encoding:
        feature['values'] = current_encoding[feature['name']]
    else:
        feature['values'] = feature['range']

rules = []
class_count = {}
max_n_conds = 0
for i in selected_path_idxes:
    conds, output = interpret_path(paths[i], features)
    if class_count.get(output, 0) >= expected_one_class_count:
        continue
    class_count[output] = class_count.get(output, 0) + 1
    rules.append({'cond': conds, 'predict': output, 'index': i})
    max_n_conds = max(len(conds), max_n_conds)
    if len(rules) >= expected_count:
        break
conds_per_line = 4
max_n_conds = math.ceil(max_n_conds / conds_per_line) * conds_per_line


rule_idxes = [rule['index'] for rule in rules]
'''
rule_type = [
    1,1,1,1,1, 0,0,0,0,1, 
    0,0,0,1,1, 0,0,0,1,0,
    0,1,0,1,1, 0,0,0,0,1,
    1,1,0,0,0, 0,0,1,0,1,
    0,0,1,0,1, 0,0,1,0,1
]
'''
rule_type = [
    1,1,0,1,1,0,0,0,0,1,
    0,0,0,1,1,0,0,0,1,0,
    0,0,0,1,1,0,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,1,0,1,
]

f = open('rule.csv', 'w')
for it, rule in enumerate(rules):
    nearest = all_dist[rule_idxes[it], rule_idxes].argsort()[1:6]
    print('rule', it, rule_type[it], ''.join([str(rule_type[i]) for i in nearest]))
    print('nearest', nearest)
    print('dist', all_dist[rule_idxes[it], nearest])
    s = '' + str(it)
    line = 0
    n_conds = len(rule['cond'])
    n_lines = math.ceil(n_conds / conds_per_line)

    for line in range(n_lines):
        if line == 0:
            s += ',IF,'
        else:
            s += ',,'
        for pos in range(conds_per_line):
            i = pos + line * conds_per_line
            if i < n_conds:
                item = rule['cond'][i]
                s += item[0] + ',' + item[1] + ',' + item[2] + ','
                s += 'AND,' if i < n_conds - 1 else ','
            else:
                s += '...,...,...,...,'
        if line == n_lines - 1:
            s += 'THEN,' + rule['predict']
        s += '\n'
    f.write(s + '\n')
f.close()
