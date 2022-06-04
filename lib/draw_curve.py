import pandas as pd
import numpy as np
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
