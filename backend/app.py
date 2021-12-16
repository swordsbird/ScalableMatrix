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
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

reassign = {
    'Creditability' : ['No', 'Yes' ], 
    'Payment Status of Previous Credit' : [
        'no credits taken/ all credits paid back duly',
        'all credits at this bank paid back duly',
        'existing credits paid back duly till now',
        'delay in paying off in the past',
        'critical account/ other credits existing (not at this bank)',
    ],
    'Purpose' : [
        'car (new)',
        'car (used)',
        'furniture/equipment',
        'radio/television',
        'domestic appliances',
        'repairs',
        'education',
        '(vacation - does not exist?)',
        'retraining',
        'business',
        'others'
    ],
    'Value Savings/Stocks': [
        '< 100 DM',
        '100 <= ... < 500 DM',
        '500 <= ... < 1000 DM',
        '>= 1000 DM',
        'unknown/ no savings account',
    ],
    'Length of current employment': [
        'unemployed',
        '< 1 year',
        '1 <= ... < 4 years',
        '4 <= ... < 7 years',
        '>= 7 years'
    ],
    'Sex & Marital Status': [
        'male : divorced/separated',
        'female : divorced/separated/married',
        'male : single',
        'male : married/widowed',
        'female : single'
    ],
    'Guarantors': [
        'none',
        'co-applicant',
        'guarantor'
    ],
    'Most valuable available asset': [
        'real estate',
        'building society savings agreement/ life insurance',
        'car or other',
        'unknown / no property'
    ],
    'Other installment plans': ['bank', 'stores', 'none'],
    'Type of apartment': ['rent', 'own', 'for free'],
    'Occupation': [
        'unemployed/ unskilled - non-resident',
        'unskilled - resident',
        'skilled employee / official',
        'management/ self-employed/ highly qualified employee/ officer'
    ],
    'Account Balance': [
        '... < 0 DM',
        '0 <= ... < 200 DM',
        '... >= 200 DM / salary assignments for at least 1 year',
        'no checking account',
    ],
    'Telephone': ['No', 'Yes'],
    'Foreign Worker': ['Yes', 'No'],
}

class DataLoader():
    def __init__(self, data, model, target):
        self.data_table = data
        self.model = model
        self.paths = self.model['paths']
        self.shap_values = self.model['shap_values']
        self.path_index = {}
        for index, path in enumerate(self.paths):
            self.path_index[path['name']] = index
        self.selected_indexes = self.model['selected']
        self.features = self.model['features']
        for index, feature in enumerate(self.features):
            if feature['name'] in reassign:
                feature['values'] = reassign[feature['name']]
            else:
                feature['values'] = feature['range']
        self.X = self.data_table.drop(target, axis=1).values
        self.y = self.data_table[target].values

        path_mat = np.array([path['sample'] + [path['output']] for path in self.paths])
        np.seterr(divide='ignore',invalid='ignore')
        path_mat = path_mat.astype(np.float32)
        tree = AnnoyIndex(len(path_mat[0]), 'euclidean')
        for i in range(len(path_mat)):
            s = np.sum(path_mat[i, :-1])
            path_mat[i, :-1] /= s
            tree.add_item(i, path_mat[i])
        tree.build(10)
        self.tree = tree

        path_dist = pairwise_distances(X = path_mat)
        K = int(np.ceil(np.sqrt(len(self.X))))
        clf = LocalOutlierFactor(n_neighbors=K, metric="precomputed")
        clf.fit(path_dist)
        path_lof = -clf.negative_outlier_factor_

        for i in range(len(self.paths)):
            self.paths[i]['lof'] = float(path_lof[i])
            self.paths[i]['represent'] = False

        rtree = AnnoyIndex(len(path_mat[0]), 'euclidean')
        for i, name in enumerate(self.selected_indexes):
            path = self.paths[self.path_index[name]]
            path['represent'] = True
            path['children'] = []
            rtree.add_item(i, path_mat[self.path_index[name]])
        rtree.build(3)
        for i in range(len(self.paths)):
            if not self.paths[i]['represent']:
                self.paths[i]['children'] = []
                nearest = rtree.get_nns_by_vector(path_mat[i], 1)[0]
                name = self.selected_indexes[nearest]
                path = self.paths[self.path_index[name]]
                path['children'].append(i)
        for i, name in enumerate(self.selected_indexes):
            path = self.paths[self.path_index[name]]
    def model_info(self):
        return self.model['model_info']

original_data = pd.read_csv('../model/data/german_detailed.csv')
data = pd.read_csv('../model/data/german.csv')
model = pickle.load(open('../model/output/german1210.pkl', 'rb'))
loader = DataLoader(data, model, 'Creditability')

@app.route('/api/data_table', methods=["POST"])
def get_data():
    features = [feature for feature in original_data.columns]
    values = [original_data[feature].values for feature in original_data.columns]
    shap = [loader.shap_values[i].values for i in range(len(loader.shap_values))]
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
        if path['represent']:
            last_rep = i
        response.append({
            'name': path['name'],
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'father': last_rep,
            'range': path['range'],
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