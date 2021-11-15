#  temporary backend server
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,classification_report
from matplotlib import pyplot as plt
from randomforest import RandomForestDemo
from sklearn.neighbors import LocalOutlierFactor
from reader import Reader
import random
import time
import json
from flask import Flask

random_state=42
random.seed(random_state)

input_path = './data/winequality-red.csv'

df = pd.read_csv(input_path)
labels = df['quality']
labels=labels.apply(lambda x:1 if x>6.5 else 0)
df = df.drop(['quality'], axis = 1)
X_raw,X_test,y_raw,y_test  = train_test_split(df,
                                              labels,
                                              test_size=0.2,
                                              stratify = labels,
                                              random_state = random_state)

N,M=X_raw.shape
class Temp_reader(Reader):
    @property
    def features(self):
        return ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    
    @property
    def categories(self):
        #  3,4,5(bad)  6,7,8(good)
        return ['bad','good']
    
    def val2float(self,feature,val)->float:
        return val
    
    def getData(self):
        return X_raw.values,y_raw.values


reader=Temp_reader()
#rf=RandomForestDemo(reader=reader,sampling_method="SMOTE",bootstrap=False,ccp_alpha=0.0,class_weight="balanced",criterion="entropy",max_depth=5,max_features="auto",max_leaf_nodes=None,max_samples=None,min_impurity_decrease=0.0,min_impurity_split=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=150,n_jobs=None,oob_score=False,random_state=42,verbose=0,warm_start=False)
rf=RandomForestDemo(reader=reader,sampling_method=None,n_estimators=200,random_state=random_state)
_paths=rf.get_paths()
paths=[p for r in _paths for p in r]

def path_score(path,X,y):
    ans=2*(y==int(path.get('output')))-1
    m=path.get('range')
    for key in m:
        ans=ans * (X[:,int(key)]>=m[key][0]) * (X[:,int(key)]<m[key][1])
    return ans

Mat=np.array([path_score(p,X_raw.values,y_raw.values) for p in paths]).astype('float')

for i in range(len(paths)):
    paths[i]['samples']=[str(t) for t in list(np.where(Mat[i]!=0)[0])]


n_estimators=rf.n_estimators
paths.sort(key=lambda x:-x['coverage'])

def getWeight(X_raw,y_raw,paths):
    # def path_score(path,X,y):
    #     ans=np.ones(len(y))
    #     m=path.get('range')
    #     for key in m:
    #         ans=ans * (X[:,int(key)]>=m[key][0]) * (X[:,int(key)]<m[key][1])
    #     return ans
    RXMat=np.abs(Mat)  # np.array([path_score(p,X_raw.values,y_raw.values) for p in paths])
    XRMat=RXMat.transpose()
    XXAnd=np.dot(XRMat,RXMat)
    XROne=np.ones(XRMat.shape)
    XXOr=2*np.dot(XROne,RXMat)-XXAnd
    XXOr=(XXOr+XXOr.transpose())/2
    # XXOr=2*n_estimators-XXAnd
    XXDis=1-XXAnd/XXOr
    # XXDis.sort(axis=0)
    K=int(np.ceil(np.sqrt(len(X_raw))))
    clf=LocalOutlierFactor(n_neighbors=K,metric="precomputed")
    clf.fit(XXDis)
    XW=-clf.negative_outlier_factor_
    # XW=1/np.sum(XXDis[:K,:],axis=0)
    # XW=XW/np.sum(XW)
    # plt.hist(XW);plt.xlabel('weight');plt.ylabel('frequency');plt.show()
    MXW,mXW=np.max(XW),np.min(XW)
    XW=1+(5-1)*(XW-mXW)/(MXW-mXW)
    return XW/np.sum(XW)

w=getWeight(X_raw,y_raw,paths)

def getRulesSet(th_N,sel_N):
    X=X_raw.values
    y=y_raw.values
    # def path_score(path,X,y):
    #     ans=2*(y==int(path.get('output')))-1
    #     m=path.get('range')
    #     for key in m:
    #         ans=ans * (X[:,int(key)]>=m[key][0]) * (X[:,int(key)]<m[key][1])
    #     return ans
    # Mat=np.array([path_score(p,X_raw.values,y_raw.values) for p in paths])
    # def confidence(p):
    #     return np.max(p.get('distribution'))/p.get('coverage')
    def calc_score(val):
        return np.dot(np.max([1-val/th_N,np.zeros(val.shape)],axis=0),w)
    coverage_list=np.array([paths[i]['coverage'] for i in range(len(paths))],dtype=int)
    def randomSelect(SR_idx,w):
        unselected=1-SR_idx
        all=np.dot(unselected,coverage_list)
        t=random.uniform(0,all)
        i=0
        while(t>=0 and i<len(SR_idx)):t-=coverage_list[i]*unselected[i];i+=1
        return i-1
    SR_idx=np.zeros(len(Mat),dtype=int)
    val=np.zeros(N,dtype=int)
    score=calc_score(val)
    decline=0
    T=0.05
    dT=T/1000
    while(decline<20):
        # weighted random select a rule
        idx=randomSelect(SR_idx,w)
        assert SR_idx[idx]<0.5
        # add the rule into SR and
        # remove a worst rule if necessary
        tmp_val=val+Mat[idx]
        idx2=None
        if np.sum(SR_idx)>=sel_N:
            bestScore=None
            best_idx2=None
            for idx2 in range(len(paths)):
                if(not SR_idx[idx2]):continue
                tmp_val-=Mat[idx2]
                tmp_score=calc_score(tmp_val)
                if(bestScore is None or bestScore>tmp_score):
                    bestScore=tmp_score
                    best_idx2=idx2
                tmp_val+=Mat[idx2]
            idx2=best_idx2
            tmp_score=bestScore
        else:
            tmp_score=calc_score(tmp_val)
        print("%.6f %02d %.6f %.6f %.6f"%(T,decline,np.exp((score-tmp_score)/T),tmp_score,score))
        if(np.log(random.uniform(0,1))<(score-tmp_score)/T):
            SR_idx[idx]=1
            val=val+Mat[idx]
            if(idx2 is not None):
                SR_idx[idx2]=0
                val=val-Mat[idx2]
            assert(np.all(np.dot(SR_idx,Mat)==val) )
            score=tmp_score
            decline=0
        else:
            decline+=1
        T=T*0.995
        # if(T<=0):T=dT
    return SR_idx,score

def SR_predict(SR_idx,X):
    SR=[paths[i] for i in range(len(SR_idx)) if SR_idx[i]>0 ]
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
    return np.array([np.argmax(np.bincount(all_ans[:,i])[1:])
        if len(np.bincount(all_ans[:,i]))>1 else -1
        for i in range(N)])

# SR_3_50,_=getRulesSet(3,50)
# SR_3_100,_=getRulesSet(3,100)
# SR_3_150,_=getRulesSet(3,150)
try:
    SR_3_50=np.load('cache/SR_3_50.npy')
    SR_3_100=np.load('cache/SR_3_100.npy')
    SR_3_150=np.load('cache/SR_3_150.npy')
except:
    SR_3_50,_=getRulesSet(3,50)
    SR_3_100,_=getRulesSet(3,100)
    SR_3_150,_=getRulesSet(3,150)
    np.save('cache/SR_3_50.npy',SR_3_50)
    np.save('cache/SR_3_100.npy',SR_3_100)
    np.save('cache/SR_3_150.npy',SR_3_150)

app = Flask(__name__)
@app.route('/data2')
def getdata2():
    global rf
    n_features=rf.n_features_
    n_categories=rf.n_categories
    features=[
        {
            "name":rf.features[i],
            "lbound":rf.feature_range[0][i],
            "rbound":rf.feature_range[1][i],
            "importance":rf.feature_importances_[i],
            "options":"+",
        }for i in range(n_features)
    ]
    SR=SR_3_100
    SR_paths=[paths[i] for i in range(len(paths)) if SR[i]  ]
    # merged_paths=rf.merge_paths(threshold)
    # import IPython;IPython.embed()
    categories=[
        {
            "name":str(rf.categories[k]),
            "total":str(rf.category_total[k]),
            "color":str(k)
        }
        for k in range(n_categories)
    ]
    return {
        "n_examples":rf.n_examples,
        "features":features,
        "categories":categories,
        "paths":SR_paths,
        "X":[list(rf.X[i]) for i in range(len(rf.X))],
        "y":list(rf.y.astype("str")),
    }

if __name__ == '__main__':
    getdata2()
    # print(getdata3())
    app.run(port=8000)
