from sklearn.datasets import load_iris
from reader import Reader
class IrisReader(Reader):
    @property
    def features(self):
        return ['sepal length','sepal width','petal length','petal width']

    @property
    def categories(self):
        return ['setosa','versicolor','virginica']
    
    def getData(self):
        return load_iris(return_X_y=True)

    def val2float(self,feature,val)->float:
        return val