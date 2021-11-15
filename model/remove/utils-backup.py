import os
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
'''
def readFile():
    return load_iris(return_X_y=True)
all_props=['sepal length','sepal width','petal length','petal width']
categories=['setosa','versicolor','virginica']
'''

data_file=os.path.join('classification','train_set.csv')
all_props=['age','job','marital','education','balance','housing','loan','poutcome']
props=['age','job','marital','education','default','balance','housing','loan','contact'
        'day','month','duration','campaign','pdays','previous','poutcome']
len_all_props=len(all_props)
partial_props=['age','job','balance','loan','housing']
categories=['positive','negative']



def fun(prop_name,val)->float:
    if(prop_name=='age'):
        return val
    elif prop_name=='job':
        jobs=['management', 'technician', 'admin.', 'services', 'retired', 'student', 'blue-collar', 'unknown', 'entrepreneur', 'housemaid', 'self-employed', 'unemployed']
        if val in jobs :return jobs.index(val)+1
        else: return 0
        return val
    elif prop_name=='marital':
        if val=='married':return 1
        elif val=='divorced':return 2
        elif val=='single':return 3
        else: return 0
    elif prop_name=='education':
        if val=='tertiary':return 1
        elif val=='primary':return 2
        elif val=='secondary':return 3
        elif val=='unknown':return 4
        else: return 0
    elif prop_name=='balance':
        return val
    elif prop_name=='housing' or prop_name=='loan':
        if val=='yes':return 1
        elif val=='no':return 2
        else: return 0
    elif prop_name=='poutcome':
        if val=='unknown':return 0
        elif val=='other':return 1
        elif val=='failure':return 2
        elif val=='success':return 3
        else :return 4

def readFile(props=all_props):
    dataset=[]
    resultset=[]
    with open(data_file,'r') as f:
        reader=csv.DictReader(f)
        for line in reader:
            obj=[]
            for prop in props:
                obj.append(float(fun(prop,line[prop])))
            dataset.append(obj)
            resultset.append(int(line['y']))

    N=len(dataset)
    # import IPython;IPython.embed()
    vec=[i for i in range(N)]
    random.shuffle(vec)  # randomly shuffles the records
    dataset=[dataset[vec[i]] for i in range(N)]
    resultset=[resultset[vec[i]] for i in range(N)]
    return np.array(dataset),np.array(resultset)


def merge_paths(data,features,categories,threshold):
    paths=[]
    merged=False
    N=len(data)
    M=len(features)
    K=len(categories)
    # used for regulation A=sqrt(feature.importance)/feature.range
    A=np.array([f['rbound']-f['lbound']>0 and np.sqrt(f['importance'])/(f['rbound']-f['lbound']) or 0 for f in features])
    # import IPython;IPython.embed()
    # print("A=",A)
    # visit=False
    for i in range(N):
        merged=False
        for j in range(len(paths)):
            if(dist2(paths[j],data[i],A=A)<threshold):
                merged=True
                # if(not visit):print(paths[j],'\n---\n',data[i],'\n-----')
                paths[j]=_merge_paths(paths[j],data[i])
                # if(not visit):print(paths[j]);visit=True
                break
        if(not merged):paths.append(data[i])
    return [{
        'name':p['name'],
        'lrange':list(p['lrange']),
        'rrange':list(p['rrange']),
        'distribution':list(p['distribution']),
    }for p in paths]

def dist2(p1,p2,A=1):
    dl=(p1['lrange']-p2['lrange'])*A
    dr=(p1['rrange']-p2['rrange'])*A
    return np.dot(dl,dl)+np.dot(dr,dr)

def _merge_paths(p1,p2):
    p1['distribution']=p1['distribution']+p2['distribution']
    p1['repeat']=p1['repeat']+p2['repeat']
    return p1

