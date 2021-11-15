import os
import numpy as np
import random
import csv
from reader import Reader

class CustomerReader(Reader):
    @property
    def features(self):
        return ['age','job','marital','education','balance','housing','loan','poutcome']
    
    @property
    def categories(self):
        return ['positive','negative']

    def val2float(self,feature,val)->float:
        if(feature=='age'):
            return val
        elif feature=='job':
            jobs=['management', 'technician', 'admin.', 'services', 'retired', 'student', 'blue-collar', 'unknown', 'entrepreneur', 'housemaid', 'self-employed', 'unemployed']
            if val in jobs :return jobs.index(val)+1
            else: return 0
            return val
        elif feature=='marital':
            if val=='married':return 1
            elif val=='divorced':return 2
            elif val=='single':return 3
            else: return 0
        elif feature=='education':
            if val=='tertiary':return 1
            elif val=='primary':return 2
            elif val=='secondary':return 3
            elif val=='unknown':return 4
            else: return 0
        elif feature=='balance':
            return val
        elif feature=='housing' or feature=='loan':
            if val=='yes':return 1
            elif val=='no':return 2
            else: return 0
        elif feature=='poutcome':
            if val=='unknown':return 0
            elif val=='other':return 1
            elif val=='failure':return 2
            elif val=='success':return 3
            else :return 4

    def getData(self):
        data_file=os.path.join('data','train_set.csv')
        dataset=[]
        resultset=[]
        with open(data_file,'r') as f:
            reader=csv.DictReader(f)
            for line in reader:
                obj=[]
                for prop in self.features:
                    obj.append(float(self.val2float(prop,line[prop])))
                dataset.append(obj)
                resultset.append(int(line['y']))
        N=len(dataset)
        vec=[i for i in range(N)]
        random.shuffle(vec)  # randomly shuffles the records
        dataset=[dataset[vec[i]] for i in range(N)]
        resultset=[resultset[vec[i]] for i in range(N)]
        return np.array(dataset),np.array(resultset)
