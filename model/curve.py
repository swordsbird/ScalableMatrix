data = [[0, 0.1078],
[0.1, 0.0704],
[0.2, 0.0320],
[0.3, -0.00472],
[0.4, -0.04235],
[0.5, -0.08118],
[0.6, -0.11787],
[0.7, -0.15569],
[0.8, -0.19353],
[0.9, -0.23142],
[1.0, -0.26932],
[1.1, -0.3073],
[1.2, -0.3453],
[1.3, -0.3834],
[1.4, -0.4215],
[1.5, -0.4597],
[1.6, -0.4919],
[1.7, -0.5362],
[1.8, -0.5745],
[1.9, -0.6128]]

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

plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

df = pd.DataFrame({
    'x': [t[0] for t in data],
    'y': [t[1] for t in data],
})
plt.figure(figsize=(15, 10))
sns.lineplot(data=df, x='x', y='y', markers=True)
plt.savefig('obj_curve.png')
