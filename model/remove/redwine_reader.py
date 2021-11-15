import os
import numpy as np
import random
import csv
from reader import Reader

class RedwineReader(Reader):
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
        data_file=os.path.join('data','winequality-red.csv')
        dataset=[]
        resultset=[]
        with open(data_file,'r') as f:
            reader=csv.DictReader(f)
            for line in reader:
                obj=[]
                for prop in self.features:
                    obj.append(float(self.val2float(prop,line[prop])))
                dataset.append(obj)
                resultset.append(int(int(line['quality'])>5.5))
        N=len(dataset)
        vec=[i for i in range(N)]
        random.shuffle(vec)  # randomly shuffles the records
        dataset=[dataset[vec[i]] for i in range(N)]
        resultset=[resultset[vec[i]] for i in range(N)]
        return np.array(dataset),np.array(resultset)
