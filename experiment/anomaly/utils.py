import pandas as pd
import seaborn as sns
import numpy as np
import random

def plot_anomaly(mat, r_lof, s_lof, type = 'rule', y = 'max', anomaly_precent = 0.05, filter_num = 2):
    if type != 'rule':
        (r_lof, s_lof) = (s_lof, r_lof)
        mat = mat.transpose()
    ordered_s_lof = sorted(s_lof)[::-1][int(len(s_lof) * anomaly_precent)]
    px = []
    pmax = []
    pmean = []
    pmedian = []
    panum = []
    pnum = []
    ppercent = []
    corr = []
    for i in range(len(r_lof)):
        x = r_lof[i]
        idxes = np.flatnonzero(mat[i, :])
        if len(idxes) <= filter_num:
            continue
        pmedian.append(np.median(s_lof[idxes]))
        pmean.append(np.mean(s_lof[idxes]))
        pmax.append(np.max(s_lof[idxes]))
        pnum.append(len(idxes))
        panum.append(np.sum(s_lof[idxes] > ordered_s_lof))
        ppercent.append(np.sum(s_lof[idxes] > ordered_s_lof) / len(idxes))
        px.append(x)
        corr.append(s_lof[idxes[random.randint(0, len(idxes) - 1)]])
    df = pd.DataFrame({ 'x': px, 'max': pmax, 'mean': pmean, 'median': pmedian, 'anomaly_num': panum, 'num': pnum, 'percent': ppercent, 'corr': corr })
    sns.scatterplot(data=df, x='x', y=y)