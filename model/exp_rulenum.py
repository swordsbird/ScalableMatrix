from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sample import create_sampler
from tree_extractor import path_extractor
from model_extractor_maxnum import Extractor

random_state = 126

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
                qualitative_features = ['Account Balance' , 'Payment Status of Previous Credit' , 'Purpose' , 'Value Savings/Stocks' , 'Length of current employment' , 'Sex & Marital Status' , 'Guarantors' , 'Most valuable available asset' , 'Other installment plans' , 'Type of apartment' ,  'Occupation' , 'Telephone' , 'Foreign Worker']
                for feature in qualitative_features:
                    unique_values = np.unique(data_table[feature].values)
                    sorted(unique_values)
                    if int(unique_values[0]) == 0:
                        for i in unique_values:
                            data_table[feature + ' - '+ str(i)] = data_table[feature].values == i
                    else:
                        for i in unique_values:
                            data_table[feature + ' - '+ str(int(i) - 1)] = data_table[feature].values == i

                #    data_table['installment - '+ concurrent_credits[i]] = data_table['Other installment plans'].values == ix
                #data_table['Account Balance'] = np.array([v if v < 4 else 0 for v in data_table['Account Balance'].values])
                for feature in qualitative_features:
                    data_table = data_table.drop(feature, axis = 1)
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
        
dataset = 'german_credit'
model = 'RF'
exp = ExpModel(dataset, model)
exp.init()
exp.train()
paths = exp.generate_paths()
exp.evaluate()

from tree_extractor import assign_samples
assign_samples(paths, (exp.X, exp.y))
paths = [p for p in paths if p['coverage'] > 0]

lamb = 100
while lamb < 2000:
    ex = Extractor(paths, exp.X_train, exp.clf.predict(exp.X_train))
    tau = 4
    w, _, fidelity_train = ex.extract(lamb, tau)
    [idx] = np.nonzero(w)
    accuracy_test = ex.evaluate(w, exp.X_test, exp.y_test)
    fidelity_test = ex.evaluate(w, exp.X_test, exp.clf.predict(exp.X_test))

    f = open('result_maxnum.txt', 'a')
    f.write('%s,%s,%s,%s,%s' % (round(lamb, 5), len(idx), round(fidelity_train, 5), round(accuracy_test, 5), round(fidelity_test, 5)))
    f.close()
    
    print(lamb, len(idx), round(fidelity_train, 5), round(accuracy_test, 5), round(fidelity_test, 5))
    lamb += 100