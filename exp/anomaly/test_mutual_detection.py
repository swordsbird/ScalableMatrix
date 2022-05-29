import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances
import pandas as pd
import seaborn as sns

input_data = pickle.load(open('../../model/output/german0316_margin_0.5_alpha_0.95_multiply.pkl', 'rb'))
rules = input_data['paths']

s_conf = np.zeros(len(rules[0]['sample']))
for r in rules:
    s_conf += np.array(r['sample'])# * r['value']
s_conf /= np.max(np.abs(s_conf))
r_conf = np.array([r['confidence'] for r in rules])
mat = np.array([np.array(r['sample']) for r in rules])
mat = mat.astype(np.float32)

def mutual_anomaly_detection(mat, row_conf = None, col_conf = None, n_neighbors = 20, lambda_ = 0.85):
    if row_conf is None:
        row_conf = np.ones(mat.shape[0])
    if col_conf is None:
        col_conf = np.ones(mat.shape[1])

    mat_t = mat.copy().transpose()
    for i in range(mat.shape[0]):
        mat[i, :] /= mat[i].sum()
    for i in range(mat_t.shape[0]):
        mat_t[i, :] /= mat_t[i].sum()
    S = mat.shape[1]
    R = mat.shape[0]
    M = np.identity(R + S)
    M[R:, :R] = mat_t
    M[:R, R:] = mat
    A = np.ones(S + R) / (S + R)

    for i in range(M.shape[1]):
        M[:, i] /= M[:, i].sum()

    s_dist = pairwise_distances(X = mat_t, metric='euclidean')
    clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
    clf.fit(s_dist)
    col_lof = -clf.negative_outlier_factor_
    print('s lof', np.min(col_lof), np.max(col_lof))

    r_dist = pairwise_distances(X = mat, metric='euclidean')
    clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
    clf.fit(r_dist)
    row_lof = -clf.negative_outlier_factor_
    print('r lof', np.min(row_lof), np.max(row_lof))
    P = np.concatenate((row_lof, col_lof), axis = 0)
    P = P / P.sum()

    for i in range(300):
        MA = np.matmul(M, A)
        new_A = lambda_ * MA + (1 - lambda_) * P
        '''
        r_score = A[:R].copy() * 1e4
        s_score = A[R:].copy() * 1e4
        print('iter %d\n rule score: %.4f %.4f %.4f\n sample score: %.4f %.4f %.4f\n'%\
            (i, np.min(r_score), np.std(r_score), np.max(r_score), \
            np.min(s_score), np.std(s_score), np.max(s_score)))
        '''
        A = new_A

    return A[:R], A[R:]


'''
version 0420

def mutual_anomaly_detection(mat, row_conf = None, col_conf = None, n_neighbors = 20, lambda_ = 0.85):
    if row_conf is None:
        row_conf = np.ones(mat.shape[0])
    if col_conf is None:
        col_conf = np.ones(mat.shape[1])

    mat_t = mat.copy().transpose()
    for i in range(mat.shape[0]):
        mat[i, :] /= mat[i].sum()
    for i in range(mat_t.shape[0]):
        mat_t[i, :] /= mat_t[i].sum()
    S = mat.shape[1]
    R = mat.shape[0]
    M = np.identity(R + S)
    M[R:, :R] = mat_t
    M[:R, R:] = mat
    A = np.ones(S + R) / (S + R)

    for i in range(M.shape[1]):
        M[:, i] /= M[:, i].sum()

    for i in range(3):
        weighted_M = M * A
        s_dist = pairwise_distances(X = weighted_M[R:, :R], metric='euclidean') 
        clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
        clf.fit(s_dist)
        col_lof = -clf.negative_outlier_factor_

        r_dist = pairwise_distances(X = weighted_M[:R, R:], metric='euclidean')
        clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
        clf.fit(r_dist)
        row_lof = -clf.negative_outlier_factor_

        P = np.concatenate((row_lof, col_lof), axis = 0)
        P = P / P.sum()

        while True:
            MA = np.matmul(M, A)
            new_A = lambda_ * MA + (1 - lambda_) * P
            if np.std(A - new_A) < 1e-9:
                break
            A = new_A

    return A[:R], A[R:]
'''

'''
version 0420 v2
def mutual_anomaly_detection_bak(mat, row_conf = None, col_conf = None, n_neighbors = 20, lambda_ = 0.85):
    if row_conf is None:
        row_conf = np.ones(mat.shape[0])
    if col_conf is None:
        col_conf = np.ones(mat.shape[1])

    mat_t = mat.copy().transpose()
    for i in range(mat.shape[0]):
        mat[i, :] /= mat[i].sum()
    for i in range(mat_t.shape[0]):
        mat_t[i, :] /= mat_t[i].sum()
    S = mat.shape[1]
    R = mat.shape[0]
    M = np.identity(R + S)
    M[R:, :R] = mat_t
    M[:R, R:] = mat
    A = np.ones(S + R) / (S + R)

    for i in range(M.shape[1]):
        M[:, i] /= M[:, i].sum()

    s_dist = pairwise_distances(X = mat_t, metric='euclidean')
    clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
    clf.fit(s_dist)
    col_lof = -clf.negative_outlier_factor_

    r_dist = pairwise_distances(X = mat, metric='euclidean')
    clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
    clf.fit(r_dist)
    row_lof = -clf.negative_outlier_factor_

    for i in range(5):
        P = np.concatenate((row_lof, col_lof), axis = 0)
        P = P / P.sum()
        MA = np.matmul(M, A)
        weighted_M = M * A
        A = lambda_ * MA + (1 - lambda_) * P
        print('round', i, weighted_M[R:, :R].sum())
        s_dist = pairwise_distances(X = weighted_M[R:, :R], metric='euclidean') 
        clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
        clf.fit(s_dist)
        col_lof = -clf.negative_outlier_factor_

        print('iter', i, 'column lof', np.min(col_lof), np.max(col_lof))
        r_dist = pairwise_distances(X = weighted_M[:R, R:], metric='euclidean')
        clf = LocalOutlierFactor(metric="precomputed", n_neighbors=n_neighbors)
        clf.fit(r_dist)
        row_lof = -clf.negative_outlier_factor_

    return A[:R], A[R:]
'''
mat = np.array([np.array(r['sample']) for r in rules])
mat = mat.astype(np.float32)

r_lof, s_lof = mutual_anomaly_detection(mat, lambda_ = 0.99)