from flask import Flask, render_template, jsonify, request
from random import *
import random
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances
from annoy import AnnoyIndex

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return 0 if obj else 1
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

#dataset = 'german'
dataset = 'bankruptcy'

if dataset == 'german':
    reassign = {
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
else:
    reassign = {}

class DataLoader():
    def __init__(self, data, model, target):
        self.data_table = data
        self.model = model
        self.paths = self.model['paths']
        self.shap_values = self.model['shap_values']
        self.path_index = {}
        for index, path in enumerate(self.paths):
            self.path_index[path['name']] = index
        max_level = max([path['level'] for path in self.paths])
        self.selected_indexes = [path['name'] for path in self.paths if path['level'] == max_level]#self.model['selected']
        self.features = self.model['features']
        for index, feature in enumerate(self.features):
            if feature['name'] in reassign:
                feature['values'] = reassign[feature['name']]
            else:
                feature['values'] = feature['range']
        self.X = self.data_table.drop(target, axis=1).values
        self.y = self.data_table[target].values

        path_mat = np.array([path['sample'] for path in self.paths])
        np.seterr(divide='ignore',invalid='ignore')
        path_mat = path_mat.astype(np.float32)

        path_dist = pairwise_distances(X = path_mat, metric='jaccard')
        tree = AnnoyIndex(len(path_mat[0]), 'euclidean')
        for i in range(len(path_mat)):
            s = np.sum(path_mat[i])
            path_mat[i] /= s
            tree.add_item(i, path_mat[i])
        tree.build(10)
        self.tree = tree
        K = int(np.ceil(np.sqrt(len(self.X))))
        K = 10
        clf = LocalOutlierFactor(n_neighbors=K, metric="precomputed")
        clf.fit(path_dist)
        path_lof = -clf.negative_outlier_factor_
        print('rule LOF', min(path_lof), max(path_lof))

        #clf2 = LocalOutlierFactor(n_neighbors=K, metric="precomputed")
        #path_dist2 = pairwise_distances(X = path_mat.transpose(), metric='jaccard')
        #clf2.fit(path_dist2)
        #path_lof2 = -clf2.negative_outlier_factor_
        #print('sample LOF', min(path_lof2), max(path_lof2))

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
    def model_info(self):
        return self.model['model_info']

if dataset == 'german':
    original_data = pd.read_csv('../model/data/german_detailed.csv')
    data = pd.read_csv('../model/data/german.csv')
    model = pickle.load(open('../model/output/german0120v2.pkl', 'rb'))
    loader = DataLoader(data, model, 'credit_risk')
else:
    original_data = pd.read_csv('../model/data/bank.csv')
    data = pd.read_csv('../model/data/bank.csv')
    model = pickle.load(open('../model/output/bankruptcy0117.pkl', 'rb'))
    loader = DataLoader(data, model, 'Bankrupt?')


@app.route('/api/data_table', methods=["POST"])
def get_data():
    features = [feature for feature in original_data.columns]
    n = 1000
    values = [original_data[feature].values[:n] for feature in original_data.columns]
    shap = [loader.shap_values[i].values[:n] for i in range(len(loader.shap_values))]
    response = {
        'features' : features,
        'values': values,
        'shap': shap,
    }
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/samples', methods=["POST"])
def get_samples():
    data = json.loads(request.get_data(as_text=True))
    ids = data['ids']
    response = []
    for i in ids:
        response.append({
            'x': loader.X[i].tolist(),
            'y': str(loader.y[i]),
        })
    return jsonify(response)

@app.route('/api/features')
def get_features():
    return json.dumps(loader.features, cls=NpEncoder)

@app.route('/api/explore_rules', methods=["POST"])
def get_explore_rules():
    data = json.loads(request.get_data(as_text=True))
    idxs = data['idxs']
    N = data['N']
    K = int(N / len(idxs)) + 3
    response = []
    nns = []
    for name in idxs:
        j = loader.path_index[name]
        neighbors = loader.paths[j]['children']
        #neighbors = [i for i in neighbors if not loader.paths[i]['represent']]
        nns += [j] + neighbors
    nns_set = set()
    last_rep = -1
    for i in nns:
        if i in nns_set:
            continue
        nns_set.add(i)
        path = loader.paths[i]
        response.append({
            'name': path['name'],
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'father': path['father'],
            'range': path['range'],
            'level': path['level'],
            'LOF': path['lof'],
            'num_children': len(path['children']),
            'distribution': path['distribution'],
            'coverage': path['coverage'] / len(loader.X),
            'output': path['output'],
            'samples': np.flatnonzero(path['sample']).tolist(),
        })
    print('response', len(response))
    #response = response[:N]
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/rule_samples', methods=["POST"])
def get_relevant_samples():
    data = json.loads(request.get_data(as_text=True))
    names = data['names']
    N = data['N']
    vec = np.zeros(loader.X.shape[0])
    for name in names:
        vec += loader.paths[loader.path_index[name]]['sample']
    ids = np.flatnonzero(vec).tolist()
    ids = random.sample(ids, N)
    response = []
    for i in ids:
        response.append({
            'id': i,
            'x': loader.X[i].tolist(),
            'y': str(loader.y[i]),
            'shap_values': loader.shap_values[i].values,
        })
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/selected_rules')
def get_selected_rules():
    response = []
    for i in loader.selected_indexes:
        path = loader.paths[loader.path_index[i]]
        response.append({
            'name': path['name'],
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'range': path['range'],
            'level': path['level'],
            'num_children': len(path['children']),
            'LOF': path['lof'],
            'distribution': path['distribution'],
            'coverage': path['coverage'] / len(loader.X),
            'output': path['output'],
            'samples': np.flatnonzero(path['sample']).tolist(),
        })
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/model_info')
def get_model_info():
    return json.dumps(loader.model_info())

'''
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get('http://localhost:8080/{}'.format(path)).text
    return render_template("index.html")
'''