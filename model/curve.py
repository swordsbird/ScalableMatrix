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

from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

lines = open('output/record_0529_2.txt', 'r').read().split('\n')
data = [[float(y) for y in x.split(',')] for x in lines if len(x) > 1]
data = np.array(data)

plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

df = pd.DataFrame({
    'lambda': [t[0] for t in data],
    'value': [t[2] for t in data],
    'type': ['anomaly' for t in data],
})
plt.figure(figsize=(15, 10))
sns.lineplot(data=df, x='lambda', y='value', hue='type', markers=True)
plt.savefig('curve_anomaly.png')
