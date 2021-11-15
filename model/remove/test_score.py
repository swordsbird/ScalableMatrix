from utils import reader
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from collections import Counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")

# Preprocessing Libraries

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit






'''
from utils import reader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix, precision_score, recall_score, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

X,y=reader.getData()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state = 1)

smote = SMOTE(random_state = 101)
X_new, y_new = smote.fit_resample(X_train,y_train)
rf=RandomForestClassifier(n_estimators=50,max_depth=2,random_state=1)
rf.fit(X_new,y_new)

y_hat=rf.predict(X_test)
P=precision_score(y_test,y_hat)
R=recall_score(y_test,y_hat)
A=accuracy_score(y_test,y_hat)
print("A = %.2f %%\nP = %.2f %%\nR = %.2f %%\n"%(A*100,P*100,R*100))
'''
'''
index0=np.where(y==0)[0]
index1=np.where(y==1)[0]
N=min(len(index0),len(index1))
index0=index0[:N]
index1=index1[:N]
index=np.array(list(index0)+list(index1))
X_new=X[index]
y_new=y[index]

X_train,X_test,y_train,y_test=train_test_split(X_new,y_new,test_size = 0.1, random_state = 1)



rf=RandomForestClassifier(n_estimators=30,max_depth=6)
rf.fit(X_train,y_train)


y_hat=rf.predict(X_test)
confusion_matrix(y_hat,y_test)
'''