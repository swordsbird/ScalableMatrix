import time
from flask import Flask
from flask import request
import decisiontree
import randomforest
import numpy as np
import json
import utils
rf=randomforest.RandomForestDemo(n_estimators=50,max_depth=2)
# rf=randomforest.RandomForestDemo(n_estimators=100,bootstrap=False, class_weight='balanced_subsample',random_state=42)
# rf=randomforest.RandomForestDemo(n_estimators=5)

app = Flask(__name__)
@app.route('/data2')
def getdata2():
    try:
        threshold=float(request.args.get("threshold"))
        assert threshold>0 and threshold<=1
    except:
        threshold=0.50

    try:
        min_impurity_loss=float(request.args.get("min_impurity_loss"))
        assert min_impurity_loss>=0.0
    except:
        min_impurity_loss=0.0
    
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
    
    paths=rf.get_paths(min_impurity_loss)
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

    # import IPython;IPython.embed()
    # 需要将X转化为不包含int32的对象
    # import IPython;IPython.embed()
    return {
        "n_examples":rf.n_examples,
        "features":features,
        "categories":categories,
        "paths":paths,
        "X":[list(rf.X[i]) for i in range(len(rf.X))],
        "y":list(rf.y.astype("str")),
    }

@app.route('/data3')
def getdata3():
    global rf
    try:
        X=json.parse(request.args.get("X"))
        fid=json.parse(request.args.get("feature_id"))
    except:
        X=rf.X[0]
        fid=0
    return rf.getRange(X,fid)

@app.route('/data/corrcoef')
def get_pearson():
    R=np.corrcoef(rf.X.T)
    return {"R":[[round(ele,4) for ele in row] for row in R]}

if __name__ == '__main__':
    getdata2()
    # print(getdata3())
    app.run(port=8000)
