
#%%
import sys
sys.path.append("../")
import pandas as pd
import numpy as np
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
import zz_pipeline_tests as pt
import joblib

main_path = '../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/'


def make_gbt(pca_num, mdep, mfeat, minss, minsl, nest, datagroup_num, pca_bool):
    
    xtrain, xtest, ytrain, ytest = datagroups[datagroup_num]

    scaler = sk.preprocessing.StandardScaler()
    xtrain_sc = scaler.fit_transform(xtrain)

    scaler2 = sk.preprocessing.StandardScaler()
    xtest_sc = scaler2.fit_transform(xtest)

    if pca_bool:
        pca = PCA(n_components=pca_num)    
        xtrain_clean = pca.fit_transform(xtrain_sc)
        xtest_clean = pca.transform(xtest_sc)
    else:
        xtrain_clean = xtrain_sc
        xtest_clean = xtest_sc

    gbt = GradientBoostingClassifier(n_estimators=nest, max_depth=mdep, max_features=mfeat, 
                                     min_samples_split=minss, min_samples_leaf=minsl)
    
    gbt.fit(xtrain_clean, ytrain)
    
    predictions = gbt.predict(xtest_clean)
    probabilities = gbt.predict_proba(xtest_clean)
    precision = sk.metrics.precision_score(ytest, predictions)
    recall = sk.metrics.recall_score(ytest, predictions)

    return gbt, predictions, probabilities, precision, recall, xtest_clean



def make_rf(pca_num, cw, crit, mf, mss, nest, datagroup_num, pca_bool):
    
    xtrain, xtest, ytrain, ytest = datagroups[datagroup_num]

    scaler = sk.preprocessing.StandardScaler()
    xtrain_sc = scaler.fit_transform(xtrain)

    scaler2 = sk.preprocessing.StandardScaler()
    xtest_sc = scaler2.fit_transform(xtest)

    if pca_bool:
        pca = PCA(n_components=pca_num)    
        xtrain_clean = pca.fit_transform(xtrain_sc)
        xtest_clean = pca.transform(xtest_sc)
    else:
        xtrain_clean = xtrain_sc
        xtest_clean = xtest_sc

    rf = RandomForestClassifier(class_weight=cw, criterion=crit, max_features=mf,
                                 min_samples_split=mss, n_estimators=nest)
    
    rf.fit(xtrain_clean, ytrain)
    
    predictions = rf.predict(xtest_clean)
    probabilities = rf.predict_proba(xtest_clean)
    precision = sk.metrics.precision_score(ytest, predictions)
    recall = sk.metrics.recall_score(ytest, predictions)

    return rf, predictions, probabilities, precision, recall, xtest_clean

#datagroups = pt.get_many_train_tests(main_path, 50)
max_depth = 5
max_features = 'sqrt'
min_samples_leaf = 50
min_samples_split = 50
n_estimators = 200
pca_n_components = 17
datagroup = 0

rf_cw = 'balanced'
rf_crit = 'gini'
rf_mf = 'sqrt'
rf_mss = 300
rf_nest = 150
rf_pca_n_components = 17
rf_datagroup = 5

# %%
#while ((prec < 0.5) or (rec < 0.025)):
#while (prec < 0.6):
precisions_all = []
recalls_all = []
precisions = []
recalls = []
for i in range(50, 51):
    prec_i = []
    rec_i = []
    datagroups = pt.get_many_train_tests(main_path, i)
    for j in range(20):
        gbt, preds, probs, prec, rec, xt = make_gbt(pca_num=17, mdep=5, mfeat='sqrt',
                                        minss=50, minsl=50, nest=200, 
                                        datagroup_num=datagroup, pca_bool=True)
    
        prec_i.append(prec)
        rec_i.append(rec)

        joblib.dump(gbt, 'saved_models\\gbt\\no_leakage\\msl50\\{}.pkl'.format(j))

    precisions.append(np.mean(prec_i))
    recalls.append(np.mean(rec_i))

    precisions_all.append(prec_i)
    recalls_all.append(rec_i)
#%%
joblib.dump(precisions, 'saved_models/stored_prec_rec/avg_precs_gbt.pkl')
joblib.dump(recalls, 'saved_models/stored_prec_rec/avg_recs_gbt.pkl')
joblib.dump(precisions_all, 'saved_models/stored_prec_rec/all_precs_gbt.pkl')
joblib.dump(recalls_all, 'saved_models/stored_prec_rec/all_recs_gbt.pkl')
#%%
#while prec < 0.35:
precisions_all = []
recalls_all = []
precisions = []
recalls = []
for i in range(50, 51):
    prec_i = []
    rec_i = []
    datagroups = pt.get_many_train_tests(main_path, i)
    for j in range(10):
        rf, preds, probs, prec, rec, xt = make_rf(pca_num=14, cw="balanced_subsample", crit="gini", mf="sqrt", 
                                         mss=100, nest=150, datagroup_num=0, pca_bool=True)
        prec_i.append(prec)
        rec_i.append(rec)

        #joblib.dump(rf, 'saved_models/rf/no_leakage/storage of models/12-3_dg1_pca14/{}.pkl'.format(j))


    precisions.append(np.mean(prec_i))
    recalls.append(np.mean(rec_i))

    precisions_all.append(prec_i)
    recalls_all.append(rec_i)
#%%
#plt.scatter(np.arange(5,51), precisions)
plt.scatter(np.arange(5,51), recalls)
joblib.dump(precisions, 'saved_models/stored_prec_rec/avg_precs.pkl')
joblib.dump(recalls, 'saved_models/stored_prec_rec/avg_recs.pkl')


#%%
joblib.dump(rf, 'saved_models\\rf_36prec48rec.pkl')
# %%
joblib.dump(gbt, 'saved_models\\gbt\\no_leakage\\mss50\\gbt_prec38rec042.pkl')
# %%
gbt_job_lib = joblib.load('saved_models\\gbt_prec62.pkl')
# %%
xtr,xte,ytr,yte = datagroups[0]
predictions = gbt_job_lib.predict(xt)
probabilities = gbt_job_lib.predict_proba(xt)
precision = sk.metrics.precision_score(yte, predictions)
recall = sk.metrics.recall_score(yte, predictions)


# %%
prec = 0
rec = 0
# %%
stored_xt = xt
# %%
np.savetxt("saved_models/rf/no_leakage/storage of models/12-3_dg1_pca14/xt.csv", xt, delimiter=",")
# %%
