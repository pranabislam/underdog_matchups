import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from joblib import load

main_path = "../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/"
cols_drop = ['unique_game_id', 'DATE', 'Home Team', 'Away Team']
playstyles = ['Home Team Cut_offensive', 'Home Team Handoff_defensive',
                'Home Team Handoff_offensive', 'Home Team Isolation_defensive',
                'Home Team Isolation_offensive', 'Home Team Misc_offensive',
                'Home Team OffScreen_defensive', 'Home Team OffScreen_offensive',
                'Home Team PnRBH_defensive', 'Home Team PnRBH_offensive',
                'Home Team PnRRM_defensive', 'Home Team PnRRM_offensive',
                'Home Team Postup_defensive', 'Home Team Postup_offensive',
                'Home Team Putbacks_defensive', 'Home Team Putbacks_offensive',
                'Home Team Spotup_defensive', 'Home Team Spotup_offensive',
                'Home Team Transition_defensive', 'Home Team Transition_offensive',
                'Away Team Cut_offensive', 'Away Team Handoff_defensive',
                'Away Team Handoff_offensive', 'Away Team Isolation_defensive',
                'Away Team Isolation_offensive', 'Away Team Misc_offensive',
                'Away Team OffScreen_defensive', 'Away Team OffScreen_offensive',
                'Away Team PnRBH_defensive', 'Away Team PnRBH_offensive',
                'Away Team PnRRM_defensive', 'Away Team PnRRM_offensive',
                'Away Team Postup_defensive', 'Away Team Postup_offensive',
                'Away Team Putbacks_defensive', 'Away Team Putbacks_offensive',
                'Away Team Spotup_defensive', 'Away Team Spotup_offensive',
                'Away Team Transition_defensive', 'Away Team Transition_offensive']
def construct_df(years, n):
    '''
    Given that our files are located in "..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows",
    return concatenated df for a given rolling avg window (n) and a list of years
    
    years: type list of ints from 2009 - 2020
    n: type in from 5 to 50 inclusive
    '''
    ## Maybe special case for dfs including 2013- and 2014+ need to be included
    dfs = []
    for year in years:
        filename = str(year) + "_stats_n" + str(n) + ".csv"
        df = pd.read_csv(main_path + filename) 
        if year >= 2014:
            df.drop(columns=playstyles, inplace=True)
        dfs.append(df)
    for i in range(1, len(dfs)):
        if len(dfs[i].columns) != len(dfs[i-1].columns):
            print(i)

    concatenated_df = pd.concat(dfs).drop(columns=cols_drop)
    return concatenated_df

test = construct_df([2019, 2020], n=10)
x_test = test.drop(columns="Underdog Win")
y_test = test["Underdog Win"]


# Generate aucs
dump_dir = "./svm_models/"
dump_prefix = "svm_"
kernels = ["linear", "rbf", "poly"]
c_vals = [25, 50, 100, 200, 400, 800]
aucs = {}
for kernel in kernels:
    aucs[kernel] = []
    for c in c_vals:
        clf = load(dump_dir + dump_prefix + kernel + "_" + str(c) + ".joblib")
        fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(x_test))
        aucs[kernel].append(auc(fpr, tpr))
    plt.plot(c_vals, aucs[kernel], label=kernel)

plt.xlabel("Regularization Parameter C")
plt.ylabel("AUC")
plt.title("SVM AUCs")
plt.legend()
plt.savefig("../../images/SVM_AUCs.png")

        















