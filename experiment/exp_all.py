from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sample import create_sampler
from tree_extractor import path_extractor
from model_extractor import Extractor
import pickle

random_state = 114

class ExpModel:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.n_splits = 4
        self._accuracy = []
        self._precision = []
        self._recall = []
        self._f1_score = []

    def init(self):
        if self.model == 'RF':
            if self.dataset == 'breast_cancer':
                self.target = 'diagnosis'
                parameters = {
                        'n_estimators': 200,
                        'max_depth': 10,
                        'random_state': random_state,
                        'max_features': None,
                }
                data_table = pd.read_csv('data/cancer.csv')
                data_table = data_table.drop(['id'], axis=1)
                X = data_table.drop(self.target, axis=1).values
                y = data_table[self.target].values
            elif self.dataset == 'abalone':
                self.target = 'Rings'
                parameters = {
                    'n_estimators': 80,
                    # 'max_depth': 30,
                    'random_state': 10,
                    'max_features': 'auto',
                    'oob_score': True,
                    'min_samples_split': 9,
                    'min_samples_leaf': 5,
                }

                data_table = pd.read_csv('data/abalone.csv')
                X = data_table.drop(self.target, axis=1).values
                y = data_table[self.target].values
                y = np.array([0 if v <= 7 else 1 for v in y])
            elif self.dataset == 'bankruptcy':
                self.target = 'Bankrupt?'
                parameters = {
                        'n_estimators': 150,
                        'max_depth': 15,
                        'random_state': random_state,
                }

                data_table = pd.read_csv('data/bank.csv')
                X = data_table.drop(self.target, axis=1).values
                y = data_table[self.target].values
            elif self.dataset == 'diabetes':
                self.target = 'class'
                parameters = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': random_state,
                    'max_features': None,
                }

                data_table = pd.read_csv('data/diabetes.csv')
                X = data_table.drop(self.target, axis=1).values
                y = data_table[self.target].values
            elif self.dataset == 'german_credit':
                self.target = 'Creditability'
                parameters = {
                    'random_state': random_state,
                    'max_depth': 12,
                    'n_estimators': 150,
                    'max_leaf_nodes': 100,
                    'min_samples_split': 6,
                    'min_samples_leaf': 3,
                    'bootstrap': True,
                }

                data_table = pd.read_csv('data/german.csv')          
                purposes = [
                    "car (new)",
                    "car (used)",
                    "furniture/equipment",
                    "radio/television",
                    "domestic appliances",
                    "repairs",
                    "education",
                    "vacation",
                    "retraining",
                    "business",
                    "others"
                ]
                concurrent_credits = ['', 'bank', 'stores', 'none']
                housings = ['', 'rent', 'own', 'for free']
                for i in np.unique(data_table['Purpose'].values):
                    data_table['Purpose - '+ purposes[i]] = data_table['Purpose'].values == i
                for i in np.unique(data_table['Type of apartment'].values):
                    data_table['Housing - '+ housings[i]] = data_table['Type of apartment'].values == i
                #for i in np.unique(data_table['Other installment plans'].values):
                #    data_table['installment - '+ concurrent_credits[i]] = data_table['Other installment plans'].values == i
                data_table['no savings account'] = data_table['Value Savings/Stocks'] == 5
                data_table['Value Savings/Stocks'] = np.array([v if v < 5 else 0 for v in data_table['Value Savings/Stocks'].values])
                data_table['no checking account'] = data_table['Account Balance'].values == 4
                data_table['Account Balance'] = np.array([v if v < 4 else 0 for v in data_table['Account Balance'].values])
                data_table = data_table.drop('Purpose', axis = 1)
                data_table = data_table.drop('Type of apartment', axis = 1)
                #data_table = data_table.drop('Other installment plans', axis = 1)
                X = data_table.drop(self.target, axis=1).values
                y = data_table[self.target].values
            elif self.dataset == 'wine':
                self.target = 'quality'
                parameters = {
                    'n_estimators': 150,
                    'max_depth': 13,
                    'random_state': random_state,
                    'max_features': 'auto',
                    'oob_score': True,
                }

                data_table = pd.read_csv('data/wine.csv')
                X = data_table.drop(self.target, axis=1).values
                y = data_table[self.target].values
                y = np.array([0 if v < 6 else 1 for v in y])
            self.data_table = data_table
            self.X = X
            self.y = y
            self.parameters = parameters
            
            kf = KFold(n_splits = self.n_splits, random_state=random_state, shuffle=True)
            self.splits = []
            for train_index, test_index in kf.split(X):
                self.splits.append((train_index, test_index))
            self.fold = 0

    def has_next_fold(self):
        return self.fold < len(self.splits)
    
    def next_fold(self):
        self.fold += 1

    def train(self):
        sm = SMOTE(random_state=random_state)
        data_table = self.data_table
        X = self.X
        y = self.y
        parameters = self.parameters
        train_index, test_index = self.splits[self.fold]
        X_train = self.X[train_index]
        y_train = self.y[train_index]
        X_test = self.X[test_index]
        y_test = self.y[test_index]
        X_train, y_train = sm.fit_resample(X_train, y_train)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.data_table = data_table
        clf = RandomForestClassifier(**parameters)
        clf.fit(X_train, y_train)
        self.clf = clf
        self.y_pred = clf.predict(X_test)
        self.features = data_table.drop(self.target, axis=1).columns.to_list()

    def evaluate(self):
        _accuracy = accuracy_score(self.y_test, self.y_pred)
        _precision = precision_score(self.y_test, self.y_pred)
        _recall = recall_score(self.y_test, self.y_pred)
        _f1_score = f1_score(self.y_test, self.y_pred)
        print('Accuracy Score is', _accuracy)
        print('Precision is', _precision)
        print('Recall is', _recall)
        print('F1 Score is', _f1_score)
        self._accuracy.append(_accuracy)
        self._precision.append(_precision)
        self._recall.append(_recall)
        self._f1_score.append(_f1_score)
    
    def summary(self):
        return float(np.mean(self._accuracy)), float(np.mean(self._precision)), float(np.mean(self._recall)), float(np.mean(self._f1_score))

    def oversampling(self, rate = 2):
        is_continuous = []
        is_categorical = []
        is_integer = []

        for feature in self.data_table.columns:
            if feature == self.target:
                continue
            if self.data_table[feature].dtype == 'O':
                is_continuous.append(False)
                is_categorical.append(True)
            else:
                is_continuous.append(True)
                is_categorical.append(False)
            is_integer.append(False)
        sampler = create_sampler(self.X_train, is_continuous, is_categorical, is_integer)
        return sampler(len(self.X_train) * rate)

    def generate_paths(self):
        if self.model == 'RF':
            paths = path_extractor(self.clf, 'random forest', (self.X_train, self.y_train))
        else:
            paths = path_extractor(self.clf, 'lightgbm')
        print('num of paths', len(paths))
        return paths


num_of_rules = [50, 100, 200, 400, 800, 1600]
tau_of_rules = [1, 1, 1.5, 1.75, 2.0, 2.5]
exp_models = ['RF']
#exp_datasets = ['german_credit', 'abalone', 'bankruptcy']
#exp_datasets = ['wine']
exp_datasets = ['abalone', 'diabetes', 'german_credit', 'wine', 'breast_cancer', 'bankruptcy']
exp_enable_oversampling = {
    'breast_cancer': True,
    'diabetes': True,
    'german_credit': True,
    'wine': True,
    'abalone': True,
    'bankruptcy': False,
}

has_file = os.path.exists('summary2.csv')
f = open('summary2.csv', 'a')
if not has_file:
    f.write('dataset,oversampling,fold,num_of_rules,tau,accuracy_train,fidelity_train,accuracy_test,fidelity_test\n')
f.close()

for oversampling in [0, 1]:
    for dataset in exp_datasets:
        for model in exp_models:
            exp = ExpModel(dataset, model)
            exp.init()
            output_data = {}
            output_data['records'] = []
            while exp.has_next_fold():
                exp.train()
                paths = exp.generate_paths()
                output_data['paths'] = paths
                exp.evaluate()
                if oversampling == 0:
                    ex = Extractor(paths, exp.X_train, exp.clf.predict(exp.X_train))
                elif not exp_enable_oversampling[dataset]:
                    break
                else:
                    X2 = exp.oversampling(2)
                    ex = Extractor(paths, X2, exp.clf.predict(X2))
                for it in range(len(num_of_rules)):
                    n = num_of_rules[it]
                    tau = tau_of_rules[it]
                    w, _, fidelity_train = ex.extract(n, tau)
                    [idx] = np.nonzero(w)
                    vec = []
                    for i in idx:
                        vec.append((i, float(w[i])))
                    accuracy_train = ex.evaluate(w, exp.X_train, exp.y_train)
                    accuracy_test = ex.evaluate(w, exp.X_test, exp.y_test)
                    fidelity_test = ex.evaluate(w, exp.X_test, exp.clf.predict(exp.X_test))
                    f = open('summary2.csv', 'a')
                    #print((dataset,oversampling,exp.fold,n,tau,accuracy_train,fidelity_train,accuracy_test,fidelity_test))
                    f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(dataset,oversampling,exp.fold,n,tau,accuracy_train,fidelity_train,accuracy_test,fidelity_test))
                    f.close()
                    output_data['records'].append({
                        'n': n,
                        'tau': tau,
                        'fold': exp.fold,
                        'oversampling': oversampling,
                        'weights': vec,
                    })
                exp.next_fold()
            output_data['performance'] = exp.summary()
            pickle.dump(output_data, open('output/%s.pkl'%(dataset), 'wb'))
