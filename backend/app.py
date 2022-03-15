from turtle import position
from urllib import response
from flask import Flask, jsonify, request
from random import *
import random
from flask_cors import CORS
import json
import numpy as np

from dataset import DatasetLoader
data_loader = DatasetLoader()

app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return 0 if obj else 1
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@app.route('/api/distribution', methods=["POST"])
def get_distribution():
    data = json.loads(request.get_data(as_text=True))
    id = data['id']
    feature = data['feature']
    dataname = data['dataname']
    loader = data_loader.get(dataname)
    path = loader.path_dict[id]
    idxes = np.flatnonzero(path['sample']).tolist()
    if len(idxes) > 200:
        idxes = random.sample(idxes, 200)
    values = loader.original_data[feature][idxes].tolist()
    return json.dumps(values, cls=NpEncoder)

@app.route('/api/data_table', methods=["POST"])
def get_data():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    loader = data_loader.get(dataname)
    features = [feature for feature in loader.original_data.columns]
    n = 1000
    values = [loader.original_data[feature].values[:n] for feature in loader.original_data.columns]
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
    dataname = data['dataname']
    loader = data_loader.get(dataname)
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
    dataname = request.args.get('dataname')
    loader = data_loader.get(dataname)
    return json.dumps(loader.features, cls=NpEncoder)


@app.route('/api/suggestions', methods=["POST"])
def get_suggestions():
    data = json.loads(request.get_data(as_text=True))
    print(data)
    ids = data['ids']
    dataname = data['dataname']
    target = data['target']
    loader = data_loader.get(dataname)
    idxes = [loader.path_index[name] for name in ids]
    samples = loader.get_relevant_samples(idxes)
    suggestions = loader.get_feature_hint(idxes, samples, target, 5)
    return json.dumps(suggestions, cls=NpEncoder)


@app.route('/api/explore_rules', methods=["POST"])
def get_explore_rules():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    loader = data_loader.get(dataname)
    fathers = data['idxs']
    idxes = []
    for name in fathers:
        j = loader.path_index[name]
        neighbors = loader.paths[j]['children']
        idxes += [j] + neighbors
    idxset = set()
    new_idxes = []
    for i in idxes:
        if i not in idxset:
            new_idxes.append(i)
            idxset.add(i)
    idxes = new_idxes
    paths = []
    for i in idxes:
        paths.append(loader.get_encoded_path(i))
    samples = loader.get_relevant_samples(idxes)
    positives, total, prob = loader.get_general_info(samples)
    response = {
        'paths': paths,
        'samples': samples,
        'info': {
            'prob': prob,
            'positives': positives,
            'total': total,
        },
    }
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/selected_rules')
def get_selected_rules():
    dataname = request.args.get('dataname')
    loader = data_loader.get(dataname)
    response = []
    idxes = [loader.path_index[i] for i in loader.selected_indexes]
    paths = []
    for i in idxes:
        paths.append(loader.get_encoded_path(i))
    samples = loader.get_relevant_samples(idxes)
    positives, total, prob = loader.get_general_info()
    response = {
        'paths': paths,
        'samples': samples,
        'info': {
            'prob': prob,
            'positives': positives,
            'total': total,
        },
    }
    return json.dumps(response, cls=NpEncoder)

@app.route('/api/rule_samples', methods=["POST"])
def get_relevant_samples():
    data = json.loads(request.get_data(as_text=True))
    dataname = data['dataname']
    loader = data_loader.get(dataname)
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

@app.route('/api/model_info')
def get_model_info():
    dataname = request.args.get('dataname')
    loader = data_loader.get(dataname)
    return json.dumps(loader.model_info())

'''
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get('http://localhost:8080/{}'.format(path)).text
    return render_template("index.html")
'''