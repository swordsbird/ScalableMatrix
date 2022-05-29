
from random import *
from turtle import position
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from annoy import AnnoyIndex
import os
import bisect

cache_dir_path = './cache'

data_encoding = {}
data_encoding['german'] = {
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

class DetectorEnsemble:
    def __init__(self):
        self.detectors = []
        self.detectors.append(('lof1', LocalOutlierFactor(metric="precomputed", n_neighbors=10)))
        self.detectors.append(('lof2', LocalOutlierFactor(metric="precomputed", n_neighbors=20)))
        self.detectors.append(('lof3', LocalOutlierFactor(metric="precomputed", n_neighbors=30)))
        self.detectors.append(('iforest1', IsolationForest(random_state = 0, n_estimators = 100)))
        self.detectors.append(('iforest2', IsolationForest(random_state = 0, n_estimators = 200)))
        self.detectors.append(('ocsvm1', OneClassSVM(gamma='auto', kernel='rbf')))
    
    def fit_detector(self, X, y, sample_weight = None):
        self.clf = LinearRegression(fit_intercept=True, normalize=False, copy_X=True).fit(X, y, sample_weight = sample_weight)

    def load(self, scores):
        self.scores = scores
        self.n = scores.shape[0]
        self.ground_truth = {}
        self.adjust_sample_weight = self.n // 50
        weights = np.ones(len(self.detectors))
        weights = weights / np.sum(weights)
        y = (self.scores * weights).sum(axis = 1)
        self.fit_detector(self.scores, y)

    def init(self, mat):
        dist = pairwise_distances(X = mat, metric='euclidean')
        self.scores = []
        for (name, detector) in self.detectors:
            if name[:3] == 'lof':
                detector.fit_predict(dist)
                self.scores.append(-detector.negative_outlier_factor_)
            else:
                detector.fit_predict(mat)
                self.scores.append(-detector.score_samples(mat))
        tmp = []
        for score in self.scores:
            min_s = np.min(score)
            max_s = np.max(score)
            score = (score - min_s) / (max_s - min_s)
            tmp.append(score)
        self.n = mat.shape[0]
        self.scores = np.array(tmp)
        self.ground_truth = {}
        self.adjust_sample_weight = self.n // 50
        weights = np.ones(len(self.detectors))
        weights = weights / np.sum(weights)

        self.scores = self.scores.transpose()
        y = (self.scores * weights).sum(axis = 1)
        self.fit_detector(self.scores, y)
    
    def predict_score(self):
        y = self.clf.predict(self.scores)
        for i in self.ground_truth:
            y[i] = self.ground_truth[i]
        return y

    def adjust_weight(self, idx, score):
        self.ground_truth[idx] = score
        sample_weight = np.ones(self.n)
        for i in self.ground_truth:
            sample_weight[i] = self.adjust_sample_weight
        y = self.predict_score()
        self.fit_detector(self.scores, y, sample_weight)

class DataLoader():
    def __init__(self, data, model, name, target, targets, target_value = 1):
        self.data_table = data
        self.model = model
        self.name = name
        self.target_value = target_value
        self.paths = self.model['paths']
        self.shap_values = self.model['shap_values']
        self.path_index = {}

        for index, path in enumerate(self.paths):
            self.path_index[path['name']] = index
        max_level = max([path['level'] for path in self.paths])
        self.selected_indexes = [path['name'] for path in self.paths if path['level'] == max_level]#self.model['selected']
        self.features = self.model['features']
        current_encoding = data_encoding.get(name, {})
        for index, feature in enumerate(self.features):
            if feature['name'] in current_encoding:
                feature['values'] = current_encoding[feature['name']]
            else:
                feature['values'] = feature['range']
            q = data[feature['name']].quantile([0.25, 0.5, 0.75]).tolist()
            q = [feature['range'][0]] + q + [feature['range'][1]]
            feature['avg'] = data[feature['name']].mean()
            feature['q'] = q
        self.X = self.data_table.drop(target, axis=1).values
        self.y = self.data_table[target].values
        self.model['model_info']['target'] = target
        self.model['model_info']['targets'] = targets
        self.target = target
        if self.model['model_info']['model'] == 'LightGBM':
            self.model['model_info']['weighted'] = True
        else:
            self.model['model_info']['weighted'] = False

        if not os.path.exists(cache_dir_path):
            os.mkdir(cache_dir_path)
        
        cache_path = os.path.join(cache_dir_path, name + '.pkl')
        if os.path.exists(cache_path):
            data = pickle.load(open(cache_path, 'rb'))
            self.paths = data['paths']
            self.detector = DetectorEnsemble()
            self.detector.load(data['scores'])
        else:
            path_mat = np.array([path['sample'] for path in self.paths])
            np.seterr(divide='ignore',invalid='ignore')
            path_mat = path_mat.astype(np.float32)

            path_dist = pairwise_distances(X = path_mat, metric='jaccard')
            tree = AnnoyIndex(len(path_mat[0]), 'euclidean')
            for i in range(len(path_mat)):
                tree.add_item(i, path_mat[i])
            tree.build(10)
            self.tree = tree
            self.detector = DetectorEnsemble()
            self.detector.init(path_mat)
            path_lof = self.detector.predict_score()

            for i in range(len(self.paths)):
                self.paths[i]['lof'] = float(path_lof[i])
                self.paths[i]['represent'] = False
                self.paths[i]['children'] = []

            for level in range(max_level, 0, -1):
                ids = []
                for i in range(len(self.paths)):
                    if self.paths[i]['level'] == level:
                        ids.append(i)
                for i in range(len(self.paths)):
                    if self.paths[i]['level'] == level - 1:
                        self.paths[i]['children'] = []
                        nearest = -1
                        nearest_dist = 1e10
                        for j in ids:
                            if path_dist[i][j] < nearest_dist and self.paths[i]['output'] == self.paths[j]['output']:
                                nearest = j
                                nearest_dist = path_dist[i][j]
                        j = nearest
                        self.paths[i]['father'] = j
                        self.paths[j]['children'].append(i)
            for i in range(len(self.paths)):
                self.paths[i]['father'] = i
            for i in range(len(self.paths)):
                for j in self.paths[i]['children']:
                    self.paths[j]['father'] = i
            pickle.dump({ 'paths': self.paths, 'scores': self.detector.scores }, open(cache_path, 'wb'))
        self.path_dict = {}
        for path in self.paths:
            self.path_dict[path['name']] = path
            path['sample'] = np.array(path['sample'])
        
        self.init_corr()
    

    def init_corr(self):
        corr = self.data_table.corr()
        self.corr = corr
        n_cols = len(self.data_table.columns)
        corr_values = []
        for i in range(n_cols):
            k1 = self.data_table.columns[i]
            for k2 in self.data_table.columns[i + 1:]:
                if k1 != k2 and abs(corr[k1][k2]) > 0:
                    corr_values.append(abs(corr[k1][k2]))
        self.has_high_corr_thres = np.quantile(corr_values, 0.975)
        self.has_low_corr_thres = np.quantile(corr_values, 0.85)
        features = [[i, x['importance'], x['name']] for i, x in enumerate(self.features)]
        features = sorted(features, key = lambda x: -x[1])
        for i, x in enumerate(features):
            k1 = x[2]
            for j in range(i + 1, len(features)):
                k2 = features[j][2]
                if abs(corr[k1][k2]) > self.has_high_corr_thres:
                    features[i][1] += features[j][1]
                    features[j][1] = 0
        features = [x for x in features if x[1] > 0]
        features = sorted(features, key = lambda x: -x[1])
        self.independent_features = [x[2] for x in features]

    def discretize(self):
        for feature in self.features:
            values = np.unique(self.data_table[feature['name']])
            #print(values)
            if len(values) > 15:
                feature['discretize'] = True
            else:
                feature['discretize'] = False
                continue
            #values = self.data_table[feature['name']].values.copy()
            values.sort()
            feature['values'] = values
            n = len(values)
            frac_n = 1.0 / n
            feature['uniques'] = n
            new_values = self.data_table[feature['name']]
            new_values = [bisect.bisect_left(values, x) * frac_n for x in new_values]
            self.data_table[feature['name']] = new_values
            new_values = self.original_data[feature['name']]
            #if feature['name'] == 'Total debt/Total net worth':
            #    print(new_values[:10])
            new_values = [bisect.bisect_left(values, x) * frac_n for x in new_values]
            #if feature['name'] == 'Total debt/Total net worth':
            #    print(new_values)
            self.original_data[feature['name']] = new_values
            q = self.data_table[feature['name']].quantile([0.25, 0.5, 0.75]).tolist()
            q = [feature['range'][0]] + q + [feature['range'][1]]
            feature['avg'] = self.data_table[feature['name']].mean()
            feature['q'] = q
            #if feature['name'] == 'Total debt/Total net worth':
            #    print(q)
            #    print(values)
        for path in self.paths:
            for i in path['range']:
                if self.features[i]['discretize']:
                    [left, right] = path['range'][i]
                    path['range'][i] = [
                        bisect.bisect_left(self.features[i]['values'], left) / self.features[i]['uniques'],
                        bisect.bisect_right(self.features[i]['values'], right) / self.features[i]['uniques'],
                    ]
    
    def get_relevant_features(self, idxes, k = 6):
        feature_count = {}
        for i in idxes:
            for j in self.paths[i]['range']:
                if j not in feature_count:
                    feature_count[j] = 0
                feature_count[j] += 1
        path_relevant_features = [(j, feature_count[j]) for j in feature_count]
        path_relevant_features = sorted(path_relevant_features, key = lambda x: -x[1])
        #path_top_relevant_features = path_relevant_features[:k]
        path_top_relevant_features = [x for x in path_relevant_features if x[1] >= len(idxes) // 2]
        if len(path_top_relevant_features) < k:
            path_top_relevant_features = path_relevant_features[:k]
        path_top_relevant_features = [self.features[x[0]]['name'] for x in path_top_relevant_features]
        return path_top_relevant_features

    def get_feature_hint(self, path_idxes, sample_idxes, target, n = 8):
        top_relevant_features = self.get_relevant_features(path_idxes)
        feature_candidates = []
        for x in self.independent_features:
            flag = False
            for y in top_relevant_features:
                if abs(self.corr[x][y]) > self.has_low_corr_thres:
                    flag = True
                    break
            if flag:
                continue
            feature_candidates.append(x)
        pattern_candidates = []
        prob = (self.data_table[self.target][sample_idxes] == target).sum() / len(sample_idxes)
        prob_general = (self.data_table[self.target] == target).sum() / len(self.data_table[self.target])
        deltas = []
        for x in feature_candidates:
            sorted_idxes = sorted(sample_idxes, key = lambda i: self.data_table[x][i])
            feature_median = self.data_table[x][sorted_idxes].median()
            if feature_median == 0:
                continue
            first_half = sorted_idxes[:len(sorted_idxes) // 2]
            second_half = sorted_idxes[len(sorted_idxes) // 2:]
            prob_greater = (self.data_table[self.target][second_half] == target).sum() / len(second_half)
            prob_smaller = (self.data_table[self.target][first_half] == target).sum() / len(first_half)
            second_half = np.flatnonzero(self.data_table[x] >= feature_median).tolist()
            first_half = np.flatnonzero(self.data_table[x] < feature_median).tolist()
            # unbalanced feature
            ratio = len(first_half) / len(self.data_table[x])
            if ratio < 0.25 or ratio > 0.75:
                continue
            prob_greater_general = (self.data_table[self.target][second_half] == target).sum() / len(second_half)
            prob_smaller_general = (self.data_table[self.target][first_half] == target).sum() / len(first_half)
            delta = abs(prob_greater_general - prob_smaller_general)
            deltas.append(delta)
            if prob_greater > prob:
                if prob_greater_general - prob_general < prob_greater - prob:
                    pattern_candidates.append((prob_greater, delta, (x, '>=', feature_median), (len(first_half), len(second_half))))
            elif prob_smaller > prob:
                if prob_smaller_general - prob_general < prob_smaller - prob:
                    pattern_candidates.append((prob_smaller, delta, (x, '<', feature_median), (len(first_half), len(second_half))))
        if len(pattern_candidates) == 0:
            return []
        pattern_candidates = sorted(pattern_candidates, key = lambda x: -x[0])
        prob_max = pattern_candidates[0][0]
        for k in pattern_candidates:
            print(k)
        mid_delta = np.quantile(deltas, 0.75)
        pattern_candidates = [x for x in pattern_candidates if x[0] > (prob + prob_max) / 2 and (x[1] < mid_delta or x[0] == prob_max)]
        return [(x[0], x[2]) for x in pattern_candidates[:n]]

    def get_general_info(self, idxes = None):
        if idxes is None:
            positives = (self.data_table[self.target] == self.target_value).sum()
            total = len(self.data_table[self.target])
        else:
            positives = (self.data_table[self.target][idxes] == self.target_value).sum()
            total = len(idxes)
        return (positives, total, positives / total)
            
    def get_relevant_samples(self, idxes):
        sample_array = self.paths[idxes[0]]['sample'].copy()
        for i in idxes[1:]:
            sample_array = sample_array + self.paths[i]['sample']
        thres = 1
        if (sample_array >= thres).sum() * 2 > len(sample_array):
            thres += 1
        return np.flatnonzero(sample_array >= thres).tolist()

    def get_encoded_path(self, idx):
        path = self.paths[idx]
        return {
            'name': path['name'],
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'father': path['father'],
            'range': path['range'],
            'level': path['level'],
            'weight': path['weight'],
            'LOF': path['lof'],
            'num_children': len(path['children']),
            'distribution': path['distribution'],
            'coverage': path['coverage'] / len(self.X),
            'output': path['output'],
            'samples': np.flatnonzero(path['sample']).tolist(),
        }

    def model_info(self):
        return self.model['model_info']

    def set_original_data(self, original_data):
        self.original_data = original_data

class DatasetLoader():
    def __init__(self):
        data_loader = {}

        original_data = pd.read_csv('../model/data/german_detailed.csv')
        data = pd.read_csv('../model/data/german.csv')
        target = 'credit_risk'
        targets = ['Rejected', 'Approved']
        
        model = pickle.load(open('../model/output/german0315v2.pkl', 'rb'))
        #        model = pickle.load(open('../model/output/german0120v2.pkl', 'rb'))
        loader = DataLoader(data, model, 'german', target, targets)
        loader.set_original_data(original_data)
        data_loader['german'] = loader

        original_data = pd.read_csv('../model/data/bank.csv')
        data = pd.read_csv('../model/data/bank.csv')
        target = 'Bankrupt?'
        model = pickle.load(open('../model/output/bankruptcy0127v3.pkl', 'rb'))
        targets = ['Bankrupt', 'Non-bankrupt']
        loader = DataLoader(data, model, 'bankruptcy', target, targets)
        loader.set_original_data(original_data)
        data_loader['bankruptcy'] = loader
        #loader.discretize()

        self.data_loader = data_loader
        
    def get(self, name):
        return self.data_loader.get(name, None)
        