# OpenML CC-18

## Load in datasets

import openml
import sklearn
from rerf.rerfClassifier import rerfClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import math
from math import log

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


dict = {rerfClassifier(feature_combinations=1.5, image_height=None, image_width=None,
               max_depth=None, max_features='auto', # min_samples_split=1,
               n_estimators=100, n_jobs=None, oob_score=False,
               patch_height_max=None, patch_height_min=1, patch_width_max=None,
               patch_width_min=1, projection_matrix='RerF', random_state=None): 
                    {'n_estimators': 350, 
                    # 'min_samples_split': 1, 
                    'max_features': 36, 
                    'max_depth': 1536, 
                    'feature_combinations': 2}, 
        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, # min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100,
                n_jobs=None, oob_score=False, random_state=None,
                verbose=0, warm_start=False): 
                    {'n_estimators': 500, 
                     # 'min_samples_split': 2, 
                     'max_features': None, 
                     'max_depth': 961}
       }

keys, values = zip(*dict.items())

# get some data
openml.config.apikey = '204cdba18d110fd68ad24b131ea92030'
benchmark_suite = openml.study.get_suite('OpenML-CC18')

for task_id in benchmark_suite.tasks[67:]:  # iterate over all tasks
    try:
        f = open("SPORF_accuracies_opti-hyper_CC-18.txt","a")

        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        n_features = np.shape(X)[1]
        n_samples = np.shape(X)[0]

        print(task_id)
        print('Data set: %s: ' % (task.get_dataset().name))

        #train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        startTime = datetime.now()

        rerf_opti = rerfClassifier(**values[0])
        rerf_opti.fit(X_train, y_train)
        rerf_pred_opti = rerf_opti.predict(X_test)
        rerf_accuracy_opti = accuracy_score(y_test, rerf_pred_opti)
        print(rerf_accuracy_opti)

        rerf_opti_time = str(datetime.now() - startTime)
        print('Time: '+ str(datetime.now() - startTime))
        startTime = datetime.now()
            
        rerf_default = RandomForestClassifier()
        rerf_default.fit(X_train, y_train)
        rerf_pred_default = rerf_default.predict(X_test)
        rerf_accuracy_default = accuracy_score(y_test, rerf_pred_default)
        print(rerf_accuracy_default)

        rerf_default_time = str(datetime.now() - startTime)
        print('Time: '+ str(datetime.now() - startTime))
        startTime = datetime.now()

        rf_opti = RandomForestClassifier(**values[1])
        rf_opti.fit(X_train, y_train)
        rf_pred_opti = rf_opti.predict(X_test)
        rf_accuracy_opti = accuracy_score(y_test, rf_pred_opti)
        print(rf_accuracy_opti)

        rf_opti_time = str(datetime.now() - startTime)
        print('Time: '+ str(datetime.now() - startTime))
        startTime = datetime.now()
            
        rf_default = RandomForestClassifier()
        rf_default.fit(X_train, y_train)
        rf_pred_default = rf_default.predict(X_test)
        rf_accuracy_default = accuracy_score(y_test, rf_pred_default)
        print(rf_accuracy_default)

        rf_default_time = str(datetime.now() - startTime)
        print('Time: '+ str(datetime.now() - startTime))

        f.write('%i,%s,%s,%s,%s,%s,%f,%f,%f,%f,\n' % (task_id,task.get_dataset().name,rerf_opti_time,rerf_default_time,rf_opti_time,rf_default_time,rerf_accuracy_opti,rerf_accuracy_default,rf_accuracy_opti,rf_accuracy_default))
        f.close()
    except:
        print('Error in OpenML CC-18 dataset ' + str(task_id))





