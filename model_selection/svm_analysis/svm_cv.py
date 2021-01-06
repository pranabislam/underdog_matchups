import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from joblib import dump
from model_feature_selection.basic_feature_analysis_pi import construct_df


main_path = "../../scraping/merging/cleaned_dfs_11-23/all_rolling_windows/"
dump_dir = "./svm_models/"
dump_prefix = "svm_cv_n"
drop_leaks = True
C = [.1, 1, 10, 100]
gamma = [.001, .01, .1, 1]
for i in range(5, 51):
	train = construct_df(range(2009, 2019), n=i, main_path=main_path, drop_leaks=drop_leaks)
	x_train = train.drop(columns="Underdog Win")
	y_train= train["Underdog Win"]
	scaler = StandardScaler()
	scaler.fit(x_train)
	x_train_scaled = scaler.transform(x_train)
	smote = SMOTE(random_state=18)
	x_res, y_res = smote.fit_resample(x_train_scaled, y_train)

	# linear unbalanced and weighted data
	tuned_params = [{"C": C, "class_weight": [None, "balanced"]}]
	clf = GridSearchCV(LinearSVC(random_state=18), tuned_params, scoring="f1")
	clf.fit(x_train_scaled, y_train)
	dump(clf, dump_dir + dump_prefix + str(i) + "_linear.joblib")

	# linear smote data
	tuned_params = [{"C": C}]
	clf = GridSearchCV(LinearSVC(random_state=18), tuned_params, scoring="f1")
	clf.fit(x_res, y_res)
	dump(clf, dump_dir + dump_prefix + str(i) + "_linear_smote.joblib")

	# rbf unbalanced and weighted data
	tuned_params = {"kernel": ["rbf"], "C": C, "gamma": gamma, "class_weight": [None, "balanced"]}
	clf = GridSearchCV(SVC(random_state=18), tuned_params, scoring="f1")
	clf.fit(x_train_scaled, y_train)
	dump(clf, dump_dir + dump_prefix + str(i) + "_rbf.joblib")

	# rbf smote
	tuned_params = {"kernel": ["rbf"], "C": C, "gamma": gamma, "class_weight": [None]}
	clf = GridSearchCV(SVC(random_state=18), tuned_params, scoring="f1")
	clf.fit(x_res, y_res)
	dump(clf, dump_dir + dump_prefix + str(i) + "_rbf_smote.joblib")






