#%%
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import churn_analysis as utils
import basic_feature_analysis_pi as bsf
import course_utils as bd
import os
import csv

# on windows
#main_path = '..\..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows\\'

# on linux
main_path = '../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/'
label = 'Underdog Win'

def create_train_test(train_beg, train_end, test_beg, test_end, n, path, preprocess=True):
    '''
    Enter beginning and end years as well as n and get train and test df back
    '''
    train_years = list(np.arange(train_beg, train_end+1))
    test_years = list(np.arange(test_beg, test_end+1))

    train_df = bsf.construct_df(train_years, n, path)
    test_df = bsf.construct_df(test_years, n, path)

    X_train = train_df.drop(label, 1)
    y_train = train_df[label]
    X_test = test_df.drop(label, 1)
    y_test = test_df[label]

    '''if preprocess:
        X_train, X_test = preprocess_df(X_train, X_test)'''

    return X_train, X_test, y_train, y_test


def make_pipeline_rf(std=True, pca=True, poly=False):
    ## Ignoring poly and pca for now
    ## Assumes standard scalar
    pca = PCA()

    steps = [('scalar', sk.preprocessing.StandardScaler()),
             ('pca', pca), 
             ('rf', RandomForestClassifier())]
    pipeline = Pipeline(steps) 
    
    return pipeline

def make_pipeline_gbt(std=True, pca=True, poly=False):
    ## Ignoring poly and pca for now
    ## Assumes standard scalar
    pca = PCA()

    steps = [('scalar', sk.preprocessing.StandardScaler()),
             ('pca', pca), 
             ('gbt', GradientBoostingClassifier())]
    pipeline = Pipeline(steps) 
    
    return pipeline

def hyper_param_search(X_train, y_train, pipeline, param_grid, num_folds):
    '''
    Assumes X_train, X_test, y_train, y_test are already made and variables accessible to the function.
    Even though random forests have a built in CV of sorts with OOB, we will still cross validate. To minimize
    time wrt CV just make kfolds = 2. Evaluation metric is recall to find positives (empirically noticed that 
    model does not readily find positives)
    '''
    kfolds = KFold(n_splits = num_folds)
    model_grid_search_scaler = GridSearchCV(pipeline, param_grid=param_grid, cv = kfolds, scoring = 'precision')
    model_grid_search_scaler.fit(X_train, y_train)

    return model_grid_search_scaler

def best_model_to_csv(param_dict, model_string):
    '''
    After performing grid search and pipeline steps, print to file the best model's params. 
    '''
    output_file = model_string + "_params.csv"
    with open(output_file, 'a+') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in param_dict.items():
            writer.writerow([key, value])
    return None

def get_many_train_tests(path, n):
    
    stored_dfs = []
    
    X_train, X_test, y_train, y_test = create_train_test(2009, 2018, 2019, 2020, n, path)
    X_train1, X_test1, y_train1, y_test1 = create_train_test(2013, 2019, 2020, 2020, n, path)
    X_train2, X_test2, y_train2, y_test2 = create_train_test(2013, 2018, 2019, 2020, n, path)
    X_train3, X_test3, y_train3, y_test3 = create_train_test(2012, 2019, 2020, 2020, n, path)
    
    X_train5 = X_train.drop(columns=['underdog_rel_Odds Close'])
    X_test5 = X_test.drop(columns=['underdog_rel_Odds Close'])
    X_train6 = X_train1.drop(columns=['underdog_rel_Odds Close'])
    X_test6 = X_test1.drop(columns=['underdog_rel_Odds Close'])
    X_train7 = X_train2.drop(columns=['underdog_rel_Odds Close'])
    X_test7 = X_test2.drop(columns=['underdog_rel_Odds Close'])
    X_train8 = X_train3.drop(columns=['underdog_rel_Odds Close'])
    X_test8 = X_test3.drop(columns=['underdog_rel_Odds Close'])

    stored_dfs = [(X_train, X_test, y_train, y_test),
                  (X_train1, X_test1, y_train1, y_test1),
                  (X_train2, X_test2, y_train2, y_test2),
                  (X_train3, X_test3, y_train3, y_test3),
                  (X_train5, X_test5, y_train, y_test),
                  (X_train6, X_test6, y_train1, y_test1),
                  (X_train7, X_test7, y_train2, y_test2),
                  (X_train8, X_test8, y_train3, y_test3)]

    return stored_dfs

def model_bakeoff():

    #for ind, datagroup in enumerate(holder):
    holder = get_many_train_tests()
    for ind, datagroup in enumerate(holder):
    
        pipeline_rf = make_pipeline_rf()
        pipeline_gbt = make_pipeline_gbt()

        ## Choose x train, and y train from the datagroup
        ## RUNNING DUMB PARAMS
        rf = hyper_param_search(datagroup[0], datagroup[2], pipeline_rf, param_grid_rf, 6)

        best_predictions_rf = rf.best_estimator_.predict(datagroup[1])
        best_proba_rf = rf.best_estimator_.predict_proba(datagroup[1])

        fpr, tpr, thresholds = sk.metrics.roc_curve(datagroup[3], best_proba_rf[:,1])
        auc_score_rf = sk.metrics.auc(fpr, tpr)

        param_dict_rf = rf.best_params_
        param_dict_rf['auc'] = auc_score_rf
        param_dict_rf['precision'] = sk.metrics.precision_score(datagroup[3], best_predictions_rf)
        param_dict_rf['recall'] = sk.metrics.recall_score(datagroup[3], best_predictions_rf)
        param_dict_rf['datagroup'] = ind
        best_model_to_csv(param_dict_rf, 'rf3_prec')
        

        ## RUNNING DUMB PARAMS
        gbt = hyper_param_search(datagroup[0], datagroup[2], pipeline_gbt, param_grid_gbt, 6)

        best_predictions_gbt = gbt.best_estimator_.predict(datagroup[1])
        best_proba_gbt = gbt.best_estimator_.predict_proba(datagroup[1])

        fpr, tpr, thresholds = sk.metrics.roc_curve(datagroup[3], best_proba_gbt[:,1])
        auc_score_gbt = sk.metrics.auc(fpr, tpr)

        param_dict_gbt = gbt.best_params_
        param_dict_gbt['auc'] = auc_score_gbt
        param_dict_gbt['precision'] = sk.metrics.precision_score(datagroup[3], best_predictions_gbt)
        param_dict_gbt['recall'] = sk.metrics.recall_score(datagroup[3], best_predictions_gbt)
        param_dict_gbt['datagroup'] = ind
        best_model_to_csv(param_dict_gbt, 'gbt3_prec')

    return None




param_grid_rf = dict(pca__n_components = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25],
                     rf__n_estimators = [100, 150, 200],
                     rf__criterion = ['gini', 'entropy'],
                     rf__class_weight = [None, 'balanced', 'balanced_subsample'],
                     rf__max_features = ['sqrt', 'log2'],
                     rf__min_samples_split = [10, 30, 50, 100, 200, 300])

param_grid_rf2 = dict(pca__n_components = [10, 15],
                     rf__class_weight = [None, 'balanced'],
                     rf__max_features = ['sqrt', 'log2'],
                     rf__min_samples_split = [10])

param_grid_gbt = dict(pca__n_components = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25],
                     gbt__n_estimators = [100, 150, 200],
                     gbt__max_depth = [3,4,5],
                     gbt__max_features = ['sqrt', 'log2'],
                     gbt__min_samples_split = [5, 10, 50, 100, 200, 300],
                     gbt__min_samples_leaf = [5, 10, 50, 100, 200, 300])

param_grid_gbt2 = dict(pca__n_components = [10, 15],
                     gbt__max_depth = [3,4],
                     gbt__min_samples_leaf = [100, 200])

#model_bakeoff()
