import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, roc_curve, auc
from joblib import load
from model_feature_selection.basic_feature_analysis_pi import construct_df
from model_evaluation.portfolio_roi import test_thresholds

main_path = "../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/"
dump_dir = "./svm_models/"
dump_prefix = "svm_cv_n"
drop_leaks = False
low_roll_num = 5
high_roll_num = 51
clf = {}
clf_smote = {}
clf_rbf = {}
clf_rbf_smote = {}
cv_results = {}

for i in range(low_roll_num, high_roll_num):
    clf[i] = load(dump_dir + dump_prefix + str(i) + "_linear.joblib")
    clf_smote[i] = load(dump_dir + dump_prefix + str(i) + "_linear_smote.joblib")
    clf_rbf[i] = load(dump_dir + dump_prefix + str(i) + "_rbf.joblib")
    clf_rbf_smote[i] = load(dump_dir + dump_prefix + str(i) + "_rbf_smote.joblib")

    # add missing columns
    cv_results_no_smote = pd.DataFrame(clf[i].cv_results_)
    cv_results_no_smote["param_gamma"] = "NA"

    cv_results_smote = pd.DataFrame(clf_smote[i].cv_results_)
    cv_results_smote["param_class_weight"] = "NA"
    cv_results_smote["param_gamma"] = "NA"

    cv_results_rbf_no_smote = pd.DataFrame(clf_rbf[i].cv_results_)

    cv_results_rbf_smote = pd.DataFrame(clf_rbf_smote[i].cv_results_)
    cv_results_rbf_smote["param_class_weight"] = "NA"

    cv_results[i] = pd.concat([cv_results_smote, cv_results_no_smote])
    cv_results[i]["param_kernel"] = "linear"
    cv_results[i] = pd.concat([cv_results[i], cv_results_rbf_no_smote, cv_results_rbf_smote])
    cv_results[i]["roll_num"] = i
df = pd.concat(cv_results.values())

df["balancing"] = df["param_class_weight"]
df["balancing"][df["balancing"] == "NA"] = "smote"
df["balancing"][df["balancing"].isnull()] = "none"
df["balancing"][df["balancing"] == "balanced"] = "class_weight"
df.reset_index(drop=True, inplace=True)
grouped_df = df.groupby(["param_kernel", "balancing", "param_C"])
max_f1 = grouped_df.max()["mean_test_score"].reset_index()
idxmax_f1 = list(grouped_df.idxmax()["mean_test_score"])
max_f1["param_gamma"] = list(df.iloc[idxmax_f1,]["param_gamma"])
max_f1["roll_num"] = list(df.iloc[idxmax_f1,]["roll_num"])

plt.figure(figsize=(20,10))
for kernel in ["linear", "rbf"]:
    for balancing in ["class_weight", "none", "smote"]:
        data = max_f1[(max_f1["param_kernel"] == kernel) & (max_f1["balancing"] == balancing)]
        label = ("kernel: " + kernel + ", balancing method: " + balancing)
        x_vals = np.log10(data["param_C"]).astype(int)
        y_vals = data["mean_test_score"]
        plt.plot(x_vals, y_vals, label=label)
        if kernel == "rbf":
            for x_val, y_val, roll_num, gamma in zip(x_vals, y_vals, data["roll_num"], data["param_gamma"]):
                plt.annotate(str(roll_num) + "," + str(gamma), (x_val, y_val + 0.01))
        else:
            for x_val, y_val, roll_num in zip(x_vals, y_vals, data["roll_num"]):
                plt.annotate(str(roll_num), (x_val, y_val + 0.01))
plt.xticks(x_vals)
plt.legend()
plt.xlabel("Log10 of Regularization Parameter C")
plt.ylabel("Maximum of Mean F1 Score")
plt.title("Maximum of Mean 5-Fold F1 Score over all Rolling Averages")
plt.savefig("../../images/SVM_f1_Comparisons.png")

opt_model_idx = max_f1.groupby(["param_kernel", "balancing"])["mean_test_score"].idxmax()
opt_models = max_f1.iloc[opt_model_idx,]
main_path = "../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/"
dump_prefix = "svm_final_n"
def create_svm_model(kernel, balancing, C, num_roll, gamma):
    train = construct_df(range(2009, 2019), n=num_roll, main_path=main_path, drop_leaks=drop_leaks)
    test = construct_df([2019, 2020], n=num_roll, main_path=main_path, drop_leaks=drop_leaks)
    x_train, x_test = train.drop(columns="Underdog Win"), test.drop(columns="Underdog Win")
    y_train, y_test = train["Underdog Win"], test["Underdog Win"]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    balance_param = None
    if balancing == "class_weight":
        balance_param = "balanced"
    elif balancing == "smote":
        smote = SMOTE(random_state=18)
        x_train_scaled, y_train_scaled = smote.fit_resample(x_train_scaled, y_train)
    if kernel == "rbf":
        clf = SVC(random_state=18, kernel=kernel, class_weight=balance_param, C=C, gamma=gamma)
    else:
        clf = LinearSVC(random_state=18, class_weight=balance_param, C=C)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    dec_func_vals = clf.decision_function(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, dec_func_vals)
    return classification_report(y_test, preds), test_thresholds(y_test, dec_func_vals, [0], main_path), auc(fpr, tpr)
for i, row in opt_models.iterrows():
    print(row["param_kernel"], row["balancing"], row["param_C"])
    class_rep, thresh_rep, auc_rep = create_svm_model(row["param_kernel"], row["balancing"], row["param_C"], row["roll_num"], row["param_gamma"])
    print(class_rep)
    print(thresh_rep)
    print("AUC=" + str(auc_rep))

