import os
import numpy as np

dataset_name="cancer"

if dataset_name=="iris":
    from iris_reader import IrisReader as DataReader
elif dataset_name=="customer":
    from customer_reader import CustomerReader as DataReader
elif dataset_name=="red wine":
    from redwine_reader import RedwineReader as DataReader
    configs={
        'n_estimators':200,
        'sampling_method': None,
        'random_state': 42,
    }
elif dataset_name=="bank":
    from bank_reader import BankReader as DataReader
elif dataset_name=="cancer":
    from cancer_reader import CancerReader as DataReader
    configs={
        'bootstrap': False, 
        'class_weight': 'balanced', 
        'criterion': 'entropy', 
        'max_depth': 5, 
        'n_estimators': 150, 
        'random_state': 42,
        'sampling_method': None,
    }

reader=DataReader()
