# Logistic Regression 

import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

cols_drop = ['unique_game_id', 'DATE', 'Home Team', 'Away Team']
non_relative_columns = ['AWAY_CUM_WIN_PCT', 'AWAY_IP_AVAIL','AWAY_PRIOR_WIN_V_OPP',
                        'AWAY_ROLL_WIN_PCT', 'Away 3P', 'Away 3P%', 'Away 3PAr', 'Away AST',
                        'Away AST%', 'Away All Stars', 'Away BLK', 'Away BLK%', 'Away DRB',
                        'Away DRB%', 'Away DRtg', 'Away FG', 'Away FG%', 'Away FT', 'Away FT%',
                        'Away FTr', 'Away ORB', 'Away ORB%', 'Away ORtg', 'Away Odds Close',
                        'Away PF', 'Away STL', 'Away STL%', 'Away TOV', 'Away TOV%', 'Away TS%',
                        'Away eFG%', 'HOME_CUM_WIN_PCT', 'HOME_IP_AVAIL', 'HOME_PRIOR_WIN_V_OPP',
                        'HOME_ROLL_WIN_PCT', 'Home 3P', 'Home 3P%', 'Home 3PAr', 'Home AST',
                        'Home AST%', 'Home All Stars', 'Home BLK', 'Home BLK%', 'Home DRB',
                        'Home DRB%', 'Home DRtg', 'Home FG', 'Home FG%', 'Home FT', 'Home FT%',
                        'Home FTr', 'Home ORB', 'Home ORB%', 'Home ORtg', 'Home Odds Close',
                        'Home PF', 'Home STL', 'Home STL%', 'Home TOV', 'Home TOV%', 'Home TS%',
                        'Home eFG%']

def construct_df(years, n, main_path):
    '''
    Given that our files are located in "..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows",
    return concatenated df for a given rolling avg window (n) and a list of years
    
    years: type list of ints from 2009 - 2020
    n: type in from 5 to 50 inclusive
    '''
    
    filenames = [str(year) + "_stats_n" + str(n) + ".csv" for year in years]
    dfs = []

    for filename in filenames:
        dfs.append(pd.read_csv(main_path + filename))
    ## Currently assuming that we are dropping playstyle stuff so wont create robust working for it
    ## remember to drop 'Away Underdog' after all of this is done

    concatenated_df = pd.concat(dfs)#.drop(columns=cols_drop)
    concatenated_df.drop((cols_drop + list(concatenated_df.filter(regex = 'Home Team')) + list(concatenated_df.filter(regex = 'Away Team'))), axis = 1, inplace = True)
    concatenated_df['Away Underdog'].replace(0, -1, inplace=True)

    new_column_tuples = []
    for i in range(int(len(non_relative_columns)/2)):
        new_column_tuples.append((non_relative_columns[i],non_relative_columns[i+31]))

    for col_pair in new_column_tuples:
        concatenated_df['underdog_rel_' + col_pair[0][5:]] = \
            (concatenated_df[col_pair[0]] - concatenated_df[col_pair[1]])*concatenated_df['Away Underdog']

    concatenated_df['Home Relative Age Diff'] = concatenated_df['Home Relative Age Diff']*concatenated_df['Away Underdog']*1 
    concatenated_df.drop(columns=non_relative_columns, inplace=True)

    return concatenated_df

# generate dataframes for all rolling windows
train_dfs, test_dfs = {}, {}
for i in range(5,51):
    train_dfs[f'train_{i}'] = construct_df(range(2014,2019), i, os.getcwd()+'\\')
    test_dfs[f'test_{i}'] = construct_df(range(2019,2021), i, os.getcwd()+'\\')
    
X_trains, Y_trains, X_tests, Y_tests = {}, {}, {}, {}
for i in train_dfs:
    X_trains[i] = train_dfs[i].drop('Underdog Win', 1)
    Y_trains[i] = train_dfs[i]['Underdog Win']
for i in test_dfs:
    X_tests[i] = test_dfs[i].drop('Underdog Win', 1)
    Y_tests[i] = test_dfs[i]['Underdog Win']
    

#Use Kfold to create 5 folds
kfolds = KFold(n_splits = 5)

#Create a set of steps. All but the last step is a transformer (something that processes data). 
#Build a list of steps, where the first is StandardScaler and the second is LogisticRegression
#The last step should be an estimator.
steps = [('scaler', StandardScaler()),
         ('pca', PCA()),
         ('lr', LogisticRegression(solver = 'liblinear'))]

#Now set up the pipeline
pipeline = Pipeline(steps)

#Now set up the parameter grid, paying close to the correct convention here
parameters_scaler = dict(lr__C = [10**i for i in range(-3, 3)],
                         lr__class_weight = ['balanced', None],
                         lr__penalty = ['l1', 'l2'],
                         lr__fit_intercept = [True, False],
                         lr__max_iter = [100, 1000, 10000])

#Now run a grid search
lr_grid_search_scaler = GridSearchCV(pipeline, param_grid = parameters_scaler, cv = kfolds, scoring = 'f1')

predicts, predict_probas, precisions, recalls, f1s, aucs, accuracies = {}, {}, {}, {}, {}, {}, {}
for i, j in list(zip(X_trains, X_tests)):
    lr_grid_search_scaler.fit(X_trains[i], Y_trains[i])
    predicts[j] = lr_grid_search_scaler.predict(X_tests[j])
    predict_probas[j] = lr_grid_search_scaler.predict_proba(X_tests[j])
    precisions[j] = precision_score(Y_tests[j], predicts[j])
    recalls[j] = recall_score(Y_tests[j], predicts[j])
    f1s[j] = f1_score(Y_tests[j], predicts[j])
    aucs[j] = roc_auc_score(Y_tests[j], predicts[j])
    accuracies[j] = accuracy_score(Y_tests[j], predicts[j])
    
rois = {}
for i in predict_probas:
    rois[i] = portfolio_roi(Y_tests[i], predict_probas[i][:,1], 10, thresh = 0.6)
    




train = construct_df(range(2014,2019), 5, os.getcwd()+'\\')
test = construct_df(range(2019,2021), 5, os.getcwd()+'\\')

## plot feature importance

X_train = train.drop('Underdog Win', 1)
Y_train = train['Underdog Win']
X_test = test.drop('Underdog Win', 1)
Y_test = test['Underdog Win']



#Use Kfold to create 4 folds
kfolds = KFold(n_splits = 5)

#Create a set of steps. All but the last step is a transformer (something that processes data). 
#Build a list of steps, where the first is StandardScaler and the second is LogisticRegression
#The last step should be an estimator.

steps = [('scaler', StandardScaler()),
         ('pca', PCA()),
         ('lr', LogisticRegression(solver = 'liblinear'))]

#Now set up the pipeline
pipeline = Pipeline(steps)

#Now set up the parameter grid, paying close to the correct convention here
parameters_scaler = dict(lr__C = [10**i for i in range(-3, 3)],
                         lr__class_weight = ['balanced', None],
                         lr__penalty = ['l1', 'l2'],
                         lr__fit_intercept = [True, False],
                         lr__max_iter = [100, 1000, 10000])

#Now run a grid search
lr_grid_search_scaler = GridSearchCV(pipeline, param_grid = parameters_scaler, cv = kfolds, scoring = 'f1')
lr_grid_search_scaler.fit(X_train, Y_train)

#Print the score of the best model
best = lr_grid_search_scaler.best_score_
print(best)

best = lr_grid_search_scaler.best_estimator_.steps



##Voting classifier to combine models
clf1 = LogisticRegression(random_state=1, class_weight = 'balanced', solver = 'liblinear')
clf2 = RandomForestClassifier(random_state=1)
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft')
params = {'lr__C': [10**i for i in range(-3, 3)], 'rf__n_estimators': [20, 200]}
paramgrid = GridSearchCV(estimator=eclf, param_grid=params, cv=kfolds).fit(X_train, Y_train)
ypred = paramgrid.predict_proba(X_test)
recall_test = recall_score(Y_test, ypred)
precision_test = precision_score(Y_test, ypred)


plot_precision_recall_curve(paramgrid, X_test, Y_test)

##LogLoss
ypredtrain = paramgrid.predict_proba(X_train)

# evaluate predictions for a 0 true value
losses_0 = [log_loss([0], [x], labels=[0,1]) for x in ypred[:,1]]
# evaluate predictions for a 1 true value
losses_1 = [log_loss([1], [x], labels=[0,1]) for x in ypred[:,1]]

losses_0_train = [log_loss([0], [x], labels=[0,1]) for x in ypredtrain[:,1]]
# plot input to loss
new_x0, new_y0 = zip(*sorted(zip(ypred[:,1], losses_0)))
new_x1, new_y1 = zip(*sorted(zip(ypred[:,1], losses_1)))

new_x0train, new_y0train = zip(*sorted(zip(ypredtrain[:,1], losses_0_train)))

plt.plot(new_x0,new_y0, label='test')
plt.plot(new_x0train,new_y0train, '--', label='train')
plt.legend()
plt.plot(new_x1,new_y1, label='true=1')



losses = []
for i,j in zip(ypred[:,1], Y_test):
    losses.append(log_loss(j,i))
    
plt.plot(ypred[:,1], losses)
plt.legend()


"""Function that accepts the following arguments: 
    **truths - actual outcomes of underdog wins/losses, 
    **probs - model.predict_proba[:, 1] array (model's probability estimate of the underdog winning)
    **stake - stake to bet on each game
    **thresh- confidence threshold for filtering bets
    returns an expected portfolio ROI
"""
def portfolio_roi(truths, probs, stake, thresh=0.5):
    # Get absolute odds from within function instead of passing them in as a parameter
    odds_df = pd.DataFrame()
    for year in [2020]:
        df = pd.read_csv(os.getcwd() + '\\' + f'{year}_stats_n5.csv')
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
        sum_bets += stake

        if row['Actuals'] == 1:
            # Track the total amount of earnings from winning bets
            sum_winnings += (stake * row['Odds'])
        else:
            # If the underdog loses, the loss is already accounted for by tracking how much money was bet
            # and the fact that no money was won (comes into play during ROI colculation)
            continue
    
    # ROI calculation
    roi = (sum_winnings - sum_bets) / sum_bets

    return roi


roi = portfolio_roi(Y_test, ypred_proba[:,1], 10)



testguess = construct_df(range(2009,2019), 5, os.getcwd()+'\\')['Underdog Win']
p1 = sum(testguess)/len(testguess)
p2 = 1-sum(testguess)/len(testguess)




randY = construct_df(range(2020,2021), 5, os.getcwd()+'\\')['Underdog Win']
randomrois, randprecisions, randrecalls = [],[],[]
for i in range(1000):
    randomguess = np.random.choice([0, 1], size=(600,), p=[0.77,0.23])
    randomrois.append(portfolio_roi(randY, randomguess, 10))
    #randprecisions.append(precision_score(randY, randomguess))
    #randrecalls.append(recall_score(randY, randomguess))
    
plt.hist(randomrois)
np.mean(randomrois), np.mean(precisions)



def calculate_log_loss(class_ratio,multi=10000):
    
    #if sum(class_ratio)!=1.0:
    #    print("warning: Sum of ratios should be 1 for best results")
    #    class_ratio[-1]+=1-sum(class_ratio)  # add the residual to last class's ratio
    
    actuals=[]
    for i,val in enumerate(class_ratio):
        actuals=actuals+[i for x in range(int(val*multi))]
        

    preds=[]
    for i in range(multi):
        preds+=[class_ratio]

    return (log_loss(actuals, preds))

loss = calculate_log_loss([p1, p2], 100)
    


baseline = LogisticRegression()
baseline = baseline.fit(X_train, Y_train)
basepred = baseline.predict_proba(X_test)[:,1]
basepredict = baseline.predict(X_test)
baserec = recall_score(Y_test, basepredict)
baseprecision = precision_score(Y_test, basepredict)
baseloss = log_loss(Y_test, basepred)
baseroi = portfolio_roi(Y_test, basepred, 10)







