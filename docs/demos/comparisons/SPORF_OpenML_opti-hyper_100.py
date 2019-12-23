# OpenML 100

## Load in datasets

import openml
import sklearn
import hyperparam_optimization as ho
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

# get some data
openml.config.apikey = '204cdba18d110fd68ad24b131ea92030'
benchmark_suite = openml.study.get_suite('OpenML100')

for task_id in benchmark_suite.tasks[92:93]:  # iterate over all tasks
    # try:
        # get some data
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        n_features = np.shape(X)[1]
        n_samples = np.shape(X)[0]

        print(task_id)
        print('Data set: %s: ' % (task.get_dataset().name))

        # build a classifier
        rerf = rerfClassifier()

        #specify max_depth and min_sample_splits ranges
        max_depth_array_rerf = (np.unique(np.round((np.linspace(2,n_samples,
                            10))))).astype(int)
        max_depth_range_rerf = np.append(max_depth_array_rerf, None)

        min_sample_splits_range_rerf = (np.unique(np.round((np.arange(1,math.log(n_samples),
                                    (math.log(n_samples)-2)/10))))).astype(int)

        # specify parameters and distributions to sample from
        rerf_param_dict = {"n_estimators": np.arange(50,550,50),
                      "max_depth": max_depth_range_rerf,
                      # "min_samples_split": min_sample_splits_range_rerf,
                      "feature_combinations": [1,2,3,4,5], 
                      "max_features": ["sqrt","log2", None, n_features**2]}

        #build another classifier
        rf = RandomForestClassifier()

        #specify max_depth and min_sample_splits ranges
        max_depth_array_rf = (np.unique(np.round((np.linspace(2,n_samples,
                            10))))).astype(int)
        max_depth_range_rf = np.append(max_depth_array_rf, None)

        min_sample_splits_range_rf = (np.unique(np.round((np.arange(2,math.log(n_samples),
                                    (math.log(n_samples)-2)/10))))).astype(int)

        # specify parameters and distributions to sample from
        rf_param_dict = {"n_estimators": np.arange(50,550,50),
                      "max_depth": max_depth_range_rf,
                      # "min_samples_split": min_sample_splits_range_rf, 
                      "max_features": ["sqrt","log2", None]}

        best_params = ho.hyperparameter_optimization_random(X, y, 
                                        (rerf, rerf_param_dict), (rf, rf_param_dict))

        #extract values from dict - seperate each classifier's param dict
        keys, values = zip(*best_params.items())

        f = open("SPORF_accuracies_opti-hyper_100-trial.txt","a")

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

        rf_opti = RandomForestClassifier(**values[1])
        rf_opti.fit(X_train, y_train)
        rf_pred_opti = rf_opti.predict(X_test)
        rf_accuracy_opti = accuracy_score(y_test, rf_pred_opti)
        print(rf_accuracy_opti)

        rf_opti_time = str(datetime.now() - startTime)
        print('Time: '+ str(datetime.now() - startTime))
        startTime = datetime.now()

        f.write('%i,%s,%s,%s,%f,%f,\n' % (task_id,task.get_dataset().name,rerf_opti_time,rf_opti_time,rerf_accuracy_opti,rf_accuracy_opti))
        f.close()
    # except:
    #     print('Error in OpenML 100 dataset ' + str(task_id))





