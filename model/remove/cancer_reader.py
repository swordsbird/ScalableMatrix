import os
import numpy as np
import random
import csv
from reader import Reader

class CancerReader(Reader):
    @property
    def features(self):
        # Net Income Flag全1，直接剃掉了
        firstrow=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

        return firstrow[2:]

    @property
    def categories(self):
        return ['B','M']

    def val2float(self,feature,val)->float:
        return float(val)

    def getData(self):
        data_file=os.path.join('data','cancer.csv')
        dataset=[]
        resultset=[]
        with open(data_file,'r',encoding="utf-8") as f:
            reader=csv.DictReader(f)
            for line in reader:
                obj=[]
                for prop in self.features:
                    obj.append(float(self.val2float(prop,line[prop])))
                dataset.append(obj)
                resultset.append(1 if line['diagnosis']=='M' else 0)
        N=len(dataset)
        vec=[i for i in range(N)]
        random.shuffle(vec)  # randomly shuffles the records
        dataset=[dataset[vec[i]] for i in range(N)]
        resultset=[resultset[vec[i]] for i in range(N)]
        return np.array(dataset),np.array(resultset)
