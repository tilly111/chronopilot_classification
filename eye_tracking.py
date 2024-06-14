import os

import constants
import platform
import pandas as pd
import numpy as np
import torch as t
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress, combinations

from sympy.combinatorics.subsets import ksubsets

from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import LearningCurveDisplay

from utils.learner_pipeline import get_pipeline_for_features

from plotting_scripts.roc_curve_plotting import get_mccv_ROC_display

from imblearn.over_sampling import BorderlineSMOTE

from utils.feature_loader import load_eye_tracking_data

from sklearn.preprocessing import MinMaxScaler

# for dim reduction?

from classifier.scream_1 import train_nn_model, predict_NN


import shap

from plotting_scripts.plot_physio import plot_physio3D, plot_physio2D

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')


# load data
X, y = load_eye_tracking_data(make_binary=True)
X.drop(columns=["participant", "time", "robot"], inplace=True)

print(X.columns)
# plt.hist(y)
# plt.show()
# exit(12)

## select hyperparameters
num_splits = 500
roc_repeats = 500
number_trees = 1000


# TODO autoML to find the best classifier and hyperparameters


# play trough
# sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)
#
# acc_list = []
#
#
# for i, (train_index, test_index) in enumerate(sss.split(X, y)):
#     print(f"Fold {i}")
#     x_train, x_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
#     y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)
#
#     learner = ExtraTreesClassifier(n_estimators=number_trees)
#
#     pl_interpretable = get_pipeline_for_features(learner, x_train, y_train, list(x_train.columns))
#
#
#     pl_interpretable.fit(x_train, y_train.values.ravel())
#
#     y_pred = pl_interpretable.predict(x_test)
#
#     acc_list.append(accuracy_score(y_test, y_pred))
#
# print(f"Mean accuracy: {np.mean(acc_list)}")
# print(f"Std accuracy: {np.std(acc_list)}")
# print(f"Max accuracy: {np.max(acc_list)}")
# print(f"Min accuracy: {np.min(acc_list)}")


# plot roc auc curve
# learner = ExtraTreesClassifier(n_estimators=number_trees)
# pl_interpretable = get_pipeline_for_features(learner, X, y, list(X.columns))
#
# fig, axs = plt.subplots(1, 1, figsize=(7, 7))
# get_mccv_ROC_display(pl_interpretable, X, y, repeats=roc_repeats, ax=axs)
# plt.savefig(f"plots/roc_curve_repeats_{roc_repeats}_extra_tree_{number_trees}_all_features.pdf")
# plt.show()

# plot learning curve
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
#
# common_params = {
#     "X": X,
#     "y": y,
#     "train_sizes": np.linspace(0.1, 1.0, 30),
#     "cv": StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0),
#     "score_type": "both",
#     "n_jobs": os.cpu_count(),
#     "line_kw": {"marker": "o"},
#     "std_display_style": "fill_between",
#     "score_name": "roc_auc",
# }
#
# # for ax_idx, estimator in enumerate([naive_bayes, svc]):
# learner = ExtraTreesClassifier(n_estimators=number_trees)
# LearningCurveDisplay.from_estimator(learner, **common_params, ax=ax)
# handles, label = ax.get_legend_handles_labels()
# ax.legend(handles[:2], ["Training Score", "Test Score"])
# ax.set_title(f"Learning Curve for {learner.__class__.__name__}")
# plt.show()
