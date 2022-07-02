import sys
sys.path.append('..')
from lib.tree_extractor import path_extractor
from lib.model_reduction_variant import Extractor
from lib.exp_model import ExpModel
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
pd.options.display.notebook_repr_html = False 
plt.rcParams['figure.dpi'] = 75 
sns.set_theme(style='darkgrid') 

data = open('data/reduction_minimize_0531.txt', 'r').read().split('\n')[:40]
data = [(i, float(x.split(',')[1]), float(x.split(',')[2])) for i, x in enumerate(data)]

df = pd.DataFrame({
    'x': [t[0] for t in data],
    'y':  [t[2] / 80 for t in data],
    'label': ['anomaly' for t in data],
})
plt.figure(figsize=(15, 10))
sns.lineplot(data=df, x='x', y='y', color='orange', markers=True)
plt.savefig('curve2.png')