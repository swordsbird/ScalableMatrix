import abc
class Reader(abc.ABC):
    @property
    @abc.abstractclassmethod
    def features(self):
        pass

    @property
    @abc.abstractclassmethod
    def categories(self):
        pass
    
    @abc.abstractclassmethod
    def getData(self):
        pass

    @abc.abstractclassmethod 
    def val2float(self,feature,val)->float:
        pass

    