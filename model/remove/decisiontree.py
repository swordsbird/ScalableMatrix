import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import utils

class DecisionTreeDemo:
    def __init__(self):
        X,y= utils.readFile()
        self.feature_names=utils.all_props
        self.categories_names=utils.categories

        self.X=X
        self.y=y
        self.n_examples=len(self.y)

        self.categories=[]
        for iy in self.y:
            if not iy in self.categories:
                self.categories.append(iy)
        self.n_categories=len(self.categories)

        self.feature_range=[np.min(X,axis=0),np.max(X,axis=0)]

        estimator = DecisionTreeClassifier(max_leaf_nodes=50, random_state=0)
        estimator.fit(X, y)

        self.n_nodes = estimator.tree_.node_count
        self.children_left = estimator.tree_.children_left
        self.children_right = estimator.tree_.children_right
        self.feature = estimator.tree_.feature
        self.threshold = estimator.tree_.threshold

        self.estimator=estimator

        self.paths=[]
        self.node2path={}
        self.dfs(0,self.feature_range)
        self.n_paths=len(self.paths)
        self.paths=np.array(self.paths)
        print(self.paths)

        self.distribution=np.zeros((self.n_paths,self.n_categories))

        node_id=self.estimator.apply(X)
        for i in range(self.n_examples):
            path_id=self.node2path[node_id[i]]
            feature_id=self.y[i]
            self.distribution[path_id][feature_id]+=1

    def dfs(self,u,feature_range):
        if self.children_left[u]<0 or self.children_right[u]<0:
            self.node2path[u]=len(self.paths)
            # import IPython;IPython.embed()
            self.paths.append([x.copy() for x in feature_range])
        else:
            feature=self.feature[u]
            threshold=self.threshold[u]
            # import IPython;IPython.embed()

            # print(u,feature_range,feature,feature_range)
            tmp_min,tmp_max=feature_range[0][feature],feature_range[1][feature]

            feature_range[1][feature]=min(tmp_max,threshold)
            self.dfs(self.children_left[u],feature_range)
            feature_range[1][feature]=tmp_max

            feature_range[0][feature]=max(tmp_min,threshold)
            self.dfs(self.children_right[u],feature_range)
            feature_range[0][feature]=tmp_min
    
    def getPaths(self):
        return self.paths


if __name__ == "__main__":
    demo=DecisionTreeDemo()
