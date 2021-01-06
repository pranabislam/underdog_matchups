
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
from sklearn.metrics import precision_score, recall_score
from basic_feature_analysis_pi import construct_df

main_path = '../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/'
datagroups = pt.get_many_train_tests(main_path, 50)
xtr,xte,ytr,yte = datagroups[2]

"""Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **stake - stake to bet on each game
    **thresh- confidence threshold for filtering bets
    **years- test years list
    returns an expected portfolio ROI
"""
def portfolio_roi(truths, probs, stake, years, thresh=0.5):
    # Get absolute odds from within function instead of passing them in as a parameter
    odds_df = pd.DataFrame()
    for year in years:
    # for year in [2019, 2020]:  <---- OLD CODE
        df = pd.read_csv(f'../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/{year}_stats_n50.csv')
        odds_df = pd.concat([odds_df, df])
    odds_df.index = range(len(odds_df))
    odds = odds_df[['Home Odds Close', 'Away Odds Close']].max(axis=1).values    
    
    # Initialize all necessary variables & data structures for computations
    portfolio = pd.DataFrame({'Actuals': truths, 'Preds': probs, 'Odds': odds})
    sum_bets, sum_winnings = 0, 0
    
    # Create df where predict_proba > threshold value (greater confidence of underdog winning)
    win_prob = portfolio['Preds'] > thresh
    good_bets = portfolio[win_prob]
    
    # Iterate over every row
    for index, row in good_bets.iterrows():
        # Track the total amount of money placed on bets
        
        # Weight each bet based on higher risk games
        #if row['Odds'] >= 3:
            #stake *= row['Odds']/2.0
        sum_bets += stake
        # print(row)

        if row['Actuals'] == 1:
            # Track the total amount of earnings from winning bets
            sum_winnings += (stake * row['Odds'])
        else:
            # If the underdog loses, the loss is already accounted for by tracking how much money was bet
            # and the fact that no money was won (comes into play during ROI colculation)
            continue
    
    # ROI calculation
    if sum_bets == 0:
        roi = 0
    else:
        roi = (sum_winnings - sum_bets) / sum_bets
    print(sum_winnings)
    print(sum_bets)
    return roi

def average_probs(prob_vectors, weights):
    '''
    Given a list of probabilities from predict_proba[:,1], and a list of weights for each
    probability vector, return the weighted average
    
    prob_vector: shape = (num_games x num diff models)
    weight vector: shape = (num diff models x 1)
    
    '''
    weights = np.array(weights)

    return prob_vectors @ weights


def random_guesser(n, p1):
    randomrois = []
    ## choose random rows fromm 2020 dataset
    df = pd.read_csv('../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/2020_stats_n50.csv')
    ## dot product underdog wins with stake * columns where win * underdog win... that should be it
    ## then calc ROI
    for i in range(10000):
        samples = df.sample(n)
        samples['udds'] = samples[['Home Odds Close', 'Away Odds Close']].max(axis=1).values
        samples['winnings'] = samples['udds']*samples['Underdog Win'] 
        
        randomguess = np.random.choice([0, 1], size=(n,), p=[p1,1-p1])
        total_invested = 100*np.sum(randomguess)
        total_winnings = 100*np.dot(samples['winnings'].to_numpy(), randomguess)
        randomrois.append(total_winnings / total_invested - 1)

    return randomrois

#%%

#%%
############################################################################
''' GBT '''
## datagroup 0 with NORMAL set
xtr,xte,ytr,yte = datagroups[0]
model_files = os.listdir("saved_models/gbt/no_leakage/msl100")
model_files.remove('probability_vec')
# model_files.remove('rf')
precisions = []
recalls = []
log_losses = []
aucs = []
rois = []


models = []
probabilities = []
## rn using each model from the pca = 13 components
xt = pd.read_csv('saved_models/pca_xtest/no_leakage/pca_x_test_n17.csv', header=None)

weights = [1/len(model_files)] * len(model_files)

for mf in model_files:
    gbt = joblib.load('saved_models/gbt/no_leakage/msl100/' + mf)
    models.append(gbt)
    probabilities.append(gbt.predict_proba(xt)[:,1])


    proba = gbt.predict_proba(xt)[:,1]
    fpr, tpr, thresholds = sk.metrics.roc_curve(yte, proba)
    aucs.append(sk.metrics.auc(fpr, tpr))

    log_losses.append(sk.metrics.log_loss(yte, proba))
    
    predictions = gbt.predict(xt)
    precisions.append(sk.metrics.precision_score(yte, predictions))
    recalls.append(sk.metrics.recall_score(yte, predictions))

    rois.append(portfolio_roi(yte, proba, 100, years=[2020], thresh=0.5))



probabilities_stacked = np.vstack(probabilities)
probabilities_stacked = probabilities_stacked.T

w_probs = average_probs(probabilities_stacked, weights)
returns = portfolio_roi(yte, w_probs, 100, years=[2020], thresh=0.50)



# %%
xt = pd.read_csv("saved_models/rf/no_leakage/x_test/pca_x_test_n14.csv", header=None)

gbt_job_lib = joblib.load('saved_models\\gbt1.pkl')
predictions = gbt_job_lib.predict(xt)
probabilities = gbt_job_lib.predict_proba(xt)
precision = precision_score(yte, predictions)
recall = recall_score(yte, predictions)

gbt2 = joblib.load('saved_models\\gbt1.pkl')
predictions2 = gbt2.predict(xt)
probabilities2 = gbt2.predict_proba(xt)
precision2 = precision_score(yte, predictions2)
recall2 = recall_score(yte, predictions2)
#%%
############################################################################
''' RANDOM FORESTS NOW '''
## datagroup 5 with smallest test set
xtr,xte,ytr,yte = datagroups[0]
## rn using each model from the pca = 14 components
xt = pd.read_csv("saved_models/rf/no_leakage/storage of models/12-3_dg0_pca14/xt.csv", header=None)

# model_files = os.listdir('saved_models/rf/no_leakage/storage of models/12-3_dg1_pca14')
model_files = ['4.pkl', '6.pkl', '7.pkl', '9.pkl', '8.pkl']
# model_files.remove('tree params.PNG')
# model_files.remove('xt.csv')
models = []
probabilities = []

precisions = []
recalls = []
log_losses = []
aucs = []
rois = []


weights = [1/len(model_files)] * len(model_files)

for mf in model_files:
    rf = joblib.load('saved_models/rf/no_leakage/storage of models/12-3_dg0_pca14/' + mf)
    models.append(rf)
    probabilities.append(rf.predict_proba(xt)[:,1])


    proba = rf.predict_proba(xt)[:,1]
    fpr, tpr, thresholds = sk.metrics.roc_curve(yte, proba)
    aucs.append(sk.metrics.auc(fpr, tpr))

    log_losses.append(sk.metrics.log_loss(yte, proba))
    
    predictions = rf.predict(xt)
    precisions.append(sk.metrics.precision_score(yte, predictions))
    recalls.append(sk.metrics.recall_score(yte, predictions))

    rois.append(portfolio_roi(yte, proba, 100, years=[2020], thresh=0.5))


probabilities_stacked = np.vstack(probabilities)
probabilities_stacked = probabilities_stacked.T

w_probs = average_probs(probabilities_stacked, weights)
returns = portfolio_roi(yte, w_probs, 100, years=[2020], thresh=0.50)
# %%
############################################################
''' Testing many different n values with random forest ''' 
''' Actually this might be a little complicated for file management 
    since i conducted pca on each...'''


#%%
## Constructing precision and recall curves for the random forest
from sklearn.metrics import plot_precision_recall_curve
# plt.title("Precision-Recall Curve for Random Forest")
plot_precision_recall_curve(rf, xt, yte)#kwargs={'title':"Precision-Recall Curve for Random Forest"})
plt.title("Precision-Recall Curve for Random Forest")
# %%
plot_precision_recall_curve(gbt, xt, yte)#kwargs={'title':"Precision-Recall Curve for Random Forest"})
plt.title("Precision-Recall Curve for Gradient Boosted Tree")




# %%
