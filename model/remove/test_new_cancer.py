import os
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from matplotlib import pyplot as plt
from randomforest import RandomForestDemo
from reader import Reader
input_path = './data/cancer.csv'

df = pd.read_csv(input_path)
label = ['B', 'M']
labels = df['diagnosis']
labels=labels.apply(lambda x:1 if x=='M' else 0)
df = df.drop(['diagnosis','id'], axis = 1)
X_raw,X_test,y_raw,y_test  = train_test_split(df,
                                              labels,
                                              test_size=0.2,
                                              stratify = labels,
                                              random_state = 42)

N,M=X_raw.shape


class Temp_reader(Reader):
    @property
    def features(self):
        firstrow=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
        return firstrow[2:]
    
    @property
    def categories(self):
        return ['B','M']
    
    def val2float(self,feature,val)->float:
        return float(val)
    
    def getData(self):
        return X_raw.values,y_raw.values


reader=Temp_reader()

rf=RandomForestDemo(reader=reader,sampling_method=None,bootstrap=False,ccp_alpha=0.0,class_weight="balanced",criterion="entropy",max_depth=5,max_features="auto",max_leaf_nodes=None,max_samples=None,min_impurity_decrease=0.0,min_impurity_split=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=150,n_jobs=None,oob_score=False,random_state=42,verbose=0,warm_start=False)

_paths=rf.get_paths()
paths=[p for r in _paths for p in r]
paths.sort(key=lambda x:-x['coverage'])


def getRulesSet(paths,X,y,th_p,th_N):
    def path_score(path,X,y):
        ans=2*(y==int(path.get('output')))-1
        m=path.get('range')
        for key in m:
            ans=ans * (X[:,int(key)]>=m[key][0]) * (X[:,int(key)]<m[key][1])
        return ans
    
    Mat=np.array([path_score(p,X_raw.values,y_raw.values) for p in paths])
    
    SR=[]
    val=np.zeros(N)
    score=0
    def confidence(p):
        return np.max(p.get('distribution'))/p.get('coverage')
    
    def cmp_gt(val,val2,th_N):
        if(th_N<0):return False
        k=np.sum(val>=th_N)-np.sum(val2>=th_N)
        if k==0 :
            th=np.ones(len(val))*th_N
            _val=np.min([th,val],axis=0)
            _val2=np.min([th,val2],axis=0)
            return np.sum(_val)>np.sum(_val2)
            # return cmp_gt(val,val2,th_N-1)
        else :return k>0
    
    for i in range(len(paths)):
        path=paths[i]
        if(confidence(path)<th_p):continue
        new_val=val+Mat[i]
        if(cmp_gt(new_val,val,th_N)):
            val=new_val
            SR.append(paths[i])
    
    return SR


def SR_predict(SR,X):
    N,M=X.shape
    all_ans=[]
    def R_predict(R,X):
        ans=np.ones(X.shape[0],dtype=int)*(int(R.get('output'))+1)
        m=R.get('range')
        for key in m:
            ans=ans*(X[:,int(key)]>=m[key][0]) * (X[:,int(key)]<m[key][1])
        return ans
    
    all_ans=[]
    for rule in SR:
        all_ans.append(R_predict(rule,X))
    all_ans=np.array(all_ans)
    return np.array([np.argmax(np.bincount(all_ans[:,i])[1:]) for i in range(N)])



th_p=0.75
sim=[]
acc=[]
SR_N=[]
X_th_N=[1,2,3,4,5,6,7,8,9,10]
for th_N in X_th_N:
    SR=getRulesSet(paths,X_raw.values,y_raw.values,th_p=th_p,th_N=th_N)
    y_test_pred=SR_predict(SR,X_test.values)
    y_train_pred=SR_predict(SR,X_raw.values)
    sim.append(accuracy_score(rf.predict(X_raw),y_train_pred))
    acc.append(accuracy_score(y_test,y_test_pred))
    SR_N.append(len(SR))


plt.plot(X_th_N,sim)
plt.plot(X_th_N,acc)
plt.show()

plt.plot(X_th_N,SR_N)
plt.show()



th_N=5
sim=[]
acc=[]
SR_N=[]
X_th_p=[0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]
for th_p in X_th_p:
    SR=getRulesSet(paths,X_raw.values,y_raw.values,th_p=th_p,th_N=th_N)
    y_test_pred=SR_predict(SR,X_test.values)
    y_train_pred=SR_predict(SR,X_raw.values)
    sim.append(accuracy_score(rf.predict(X_raw),y_train_pred))
    acc.append(accuracy_score(y_test,y_test_pred))
    SR_N.append(len(SR))

plt.plot(X_th_p,sim)
plt.plot(X_th_p,acc)
plt.show()

plt.plot(X_th_p,SR_N)
plt.show()





'''
sim=accuracy(
             rf.pred(X_train),
             SR.pred(X_train)
)

acc=accurcacy(
             y_test,
             SR.pred(X_test)
)
'''