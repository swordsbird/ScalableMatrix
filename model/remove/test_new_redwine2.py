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
n_estimators=rf.n_estimators
paths.sort(key=lambda x:-x['coverage'])

def getWeight(X_raw,y_raw,paths):
    def path_score(path,X,y):
        ans=np.ones(len(y))
        m=path.get('range')
        for key in m:
            ans=ans * (X[:,int(key)]>=m[key][0]) * (X[:,int(key)]<m[key][1])
        return ans
    RXMat=np.array([path_score(p,X_raw.values,y_raw.values) for p in paths])
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
    def path_score(path,X,y):
        ans=2*(y==int(path.get('output')))-1
        m=path.get('range')
        for key in m:
            ans=ans * (X[:,int(key)]>=m[key][0]) * (X[:,int(key)]<m[key][1])
        return ans
    Mat=np.array([path_score(p,X_raw.values,y_raw.values) for p in paths])
    def confidence(p):
        return np.max(p.get('distribution'))/p.get('coverage')
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

t0=time.time()
test_number=8

choice_sel_N=[20,50,100]
range_th_N=[1,2,3,4,5,6,7,8,9,10]
result_th_N=[]
# [0.09794544620725518, 0.1217316377785348, 0.1393753211190352, 0.1511310394373695, 0.15881478263881163, 0.17233199021869228, 0.17421559703505862, 0.19063152286658325, 0.1975559756440801, 0.21230634131013612, 0.05979050484743374, 0.08425057959134422, 0.09734992065934715, 0.1140755726392727, 0.12012201956391823, 0.12793596815715247, 0.13449736277369095, 0.14015349782922612, 0.1463159662170576, 0.15112646795627024, 0.01352939750743346, 0.05593778545099082, 0.06300446302851771, 0.07668686594542079, 0.08728194899095347, 0.09713363400315721, 0.10661088600325952, 0.11143900651099997, 0.11279717637942406, 0.1196995557170041]
acc_th_N=[]
# [0.86875, 0.884375, 0.9, 0.88125, 0.890625, 0.853125, 0.859375, 0.846875, 0.859375, 0.846875, 0.884375, 0.909375, 0.915625, 0.9125, 0.90625, 0.896875, 0.925, 0.90625, 0.909375, 0.89375, 0.884375, 0.915625, 0.921875, 0.925, 0.93125, 0.928125, 0.915625, 0.915625, 0.915625, 0.9125]
for sel_N in choice_sel_N:
    for th_N in range_th_N:
        for t in range(test_number):
            os.system('title "th_N=%d sel_N=[%d] (%d/%d) "'%(th_N,sel_N,t+1,test_number))
            SR_idx,score=getRulesSet(th_N,sel_N)
            result_th_N.append(score)
            y_test_pred=SR_predict(SR_idx,X_test.values)
            acc=accuracy_score(y_test,y_test_pred)
            acc_th_N.append(acc)


range_sel_N=[10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250]
choice_th_N=[1,3,5]
result_sel_N=[]
acc_sel_N=[]
for th_N in choice_th_N:
    for sel_N in range_sel_N:
        for t in range(test_number):
            os.system('title "th_N=[%d] sel_N=%d (%d/%d)"'%(th_N,sel_N,t+1,test_number))
            SR_idx,score=getRulesSet(th_N,sel_N)
            result_sel_N.append(score)
            y_test_pred=SR_predict(SR_idx,X_test.values)
            acc=accuracy_score(y_test,y_test_pred)
            acc_sel_N.append(acc)

t1=time.time()

print("Time: %d s"%(t1-t0))




saveobj={
    'test_number':test_number,
    'range_th_N':range_th_N,
    'choice_sel_N':choice_sel_N,
    'result_th_N':result_th_N,
    'result_sel_N':result_sel_N,
    'acc_sel_N':acc_sel_N,
    'acc_th_N':acc_th_N,
    'range_sel_N':range_sel_N,
    'choice_sel_N':choice_sel_N,
}

with open('tmpsave.json','w') as f:
    f.write(json.dumps(saveobj))

_acc_th_N=np.array(acc_th_N).reshape((len(choice_sel_N) ,len(range_th_N), test_number))
_result_th_N=np.array(result_th_N).reshape((len(choice_sel_N) ,len(range_th_N), test_number))

_acc_sel_N=np.array(acc_sel_N).reshape((len(choice_th_N) ,len(range_sel_N), test_number))
_result_sel_N=np.array(result_sel_N).reshape((len(choice_th_N) ,len(range_sel_N), test_number))

# acc - th_N
plt.subplot(221)
legend_sel_N=[]
__acc_th_N=np.mean(_acc_th_N,axis=2)
for i in range(len(choice_sel_N)):
    plt.plot(range_th_N,__acc_th_N[i])
    legend_sel_N.append('sel_N=%d'%choice_sel_N[i])

plt.xlabel('th_N')
plt.ylabel('acc')
plt.legend(legend_sel_N)
# plt.show()

# loss - th_N
plt.subplot(222)
__result_th_N=np.mean(_result_th_N,axis=2)
for i in range(len(choice_sel_N)):
    plt.plot(range_th_N,__result_th_N[i])

plt.xlabel('th_N')
plt.ylabel('loss')
plt.legend(legend_sel_N)
# plt.show()



# acc - sel_N
plt.subplot(223)
legend_th_N=[]
__acc_sel_N=np.mean(_acc_sel_N,axis=2)
for i in range(len(choice_th_N)):
    plt.plot(range_sel_N,__acc_sel_N[i])
    legend_th_N.append('th_N=%d'%choice_th_N[i])

plt.xlabel('sel_N')
plt.ylabel('acc')
plt.legend(legend_th_N)
# plt.show()

# loss - sel_N
plt.subplot(224)
__result_sel_N=np.mean(_result_sel_N,axis=2)
for i in range(len(choice_th_N)):
    plt.plot(range_sel_N,__result_sel_N[i])

plt.xlabel('sel_N')
plt.ylabel('loss')
plt.legend(legend_th_N)
plt.show()


