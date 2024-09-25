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

from imblearn.over_sampling import BorderlineSMOTE

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AffinityPropagation

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


study = "1"  # "1" or "2"
block_names = ["exp_T", "exp_MA", "exp_TU", "exp_PU"]  # "exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"
background_block_name = "exp_S"  # "baseline"
classifier_name = "KNN"  # "SVC", "DTC", "KNN", "GNB", "LR", "LDA", "RF", "GB", "AB", "XGB", "QDA", "NN"
individual_tag = "v3"
use_shap = False
feature_selection = "None"  # "RFECV", "SFS", None, "MANUAL"
neuro_kit = False
vizualize = False

# what features to use
manual_features = ["breathing rate", "bpm", "hr_mad"]

if study == "1":
    population = constants.SUBJECTS_STUDY_1
    # population = constants.SUBJECTS_STUDY_1_test
    # population = constants.SUBJECTS_STUDY_1_test2
    # population = constants.SUBJECTS_STUDY_1_over
    # population = constants.SUBJECTS_STUDY_1_only3
    # population = [24, 30, 36]
else:
    population = constants.SUBJECTS_STUDY_2  # constants.SUBJECTS_STUDY_2, GROUP_5

########################################################################################################################
# load data and labels
########################################################################################################################
# get background subtraction
if neuro_kit:
    ppg_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/ppg_nk.csv")
else:
    ppg_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/ppg.csv")
eda_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/eda.csv")
tmp_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/tmp.csv")
eda_features_bg.drop(columns=["subject"], inplace=True)
tmp_features_bg.drop(columns=["subject"], inplace=True)

x_bg = pd.concat([ppg_features_bg, eda_features_bg, tmp_features_bg], axis=1).reset_index().drop(columns=["index"])
# x_bg = eda_features_bg.reset_index().drop(columns=["index"])
x_bg.fillna(0, inplace=True)  # TODO hack to resolve nans

labels_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{background_block_name}.csv")

# load all features and labels
x_all = None
y_all = None

for block in block_names:
    labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")
    # labels["block_estimation"] = labels["block_estimation"] - labels_bg["block_estimation"] + 1
    # if block == "exp_MA":
    #     labels["block_estimation"] = 0
    # elif block == "exp_T":  # exp_TU exp_MA
    #     labels["block_estimation"] = 1
    # elif block == "exp_TU":
    #     labels["block_estimation"] = 0
    # else:
    #     labels["block_estimation"] = 3
    # print(block)

    if neuro_kit:
        ppg_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/ppg_nk.csv")
    else:
        ppg_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/ppg.csv")
    eda_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/eda.csv")
    tmp_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/tmp.csv")
    eda_features.drop(columns=["subject"], inplace=True)
    tmp_features.drop(columns=["subject"], inplace=True)

    x = pd.concat([ppg_features, eda_features, tmp_features], axis=1).reset_index().drop(columns=["index"])
    # x = eda_features.reset_index().drop(columns=["index"])
    x.fillna(0, inplace=True)  # TODO hack to resolve nans

    # add background subtraction
    # x.loc[:, x.columns != 'subject'] = x.loc[:, x.columns != 'subject'] - x_bg.loc[:, x_bg.columns != 'subject']
    x.loc[:, x.columns != 'subject'] = (x.loc[:, x.columns != 'subject'] - x_bg.loc[:, x_bg.columns != 'subject']) / x_bg.loc[:, x_bg.columns != 'subject']

    if x_all is None:
        x_all = x
        y_all = labels
    else:
        x_all = pd.concat([x_all, x], axis=0)
        y_all = pd.concat([y_all, labels], axis=0)

# if use_shap or feature_selection is not None:
if neuro_kit:
    all_features_c = constants.ALL_PPG_FEATURES_NEUROKIT_AVAILABLE + constants.ALL_EDA_FEATURES + constants.ALL_TMP_FEATURES
else:
    all_features_c = constants.ALL_PPG_FEATURES_HEARTPY + constants.ALL_EDA_FEATURES + constants.ALL_TMP_FEATURES
    # all_features_c = constants.ALL_EDA_FEATURES

if feature_selection == "MANUAL":
    all_features_c = manual_features
    manual_features.append("subject")
    x_all = x_all.drop(columns=[col for col in x_all.columns if col not in manual_features])

    # apply some additional dimensionality reduction aka smart down scaling
    # dim_reducer = Isomap(n_components=3, n_neighbors=5)
    # dim_reducer = LocallyLinearEmbedding(n_components=2, n_neighbors=2)
    # x_numpy = dim_reducer.fit_transform(x_all.drop(columns=["subject"]))
    # x_df = pd.DataFrame(data=x_numpy, columns=["dim1", "dim2", "dim3"])
    # x_df = pd.DataFrame(data=x_numpy, columns=["dim1", "dim2"])
    # x_all_tmp = x_all["subject"].reset_index().drop(columns=["index"])
    # x_all_tmp["dim1"] = x_df["dim1"]
    # x_all_tmp["dim2"] = x_df["dim2"]
    # # x_all_tmp["dim3"] = x_df["dim3"]
    # x_all = x_all_tmp

x_all_c0 = x_all[y_all["block_estimation"] == 0]
x_all_c1 = x_all[y_all["block_estimation"] == 1]

subjects_c0 = x_all_c0["subject"]
subjects_c1 = x_all_c1["subject"]

x_all_c0.drop(columns=["subject"], inplace=True)
x_all_c1.drop(columns=["subject"], inplace=True)



cluster_al = AffinityPropagation()

cluster_al.fit(x_all_c0)

print(cluster_al.labels_)
print(np.unique(cluster_al.labels_, return_counts=True))
print(f"largest clustered subgroup: {np.argmax(np.unique(cluster_al.labels_, return_counts=True)[1])}")

print(f"clustered subgroup: {np.unique(subjects_c0[cluster_al.labels_ == np.argmax(np.unique(cluster_al.labels_, return_counts=True)[1])].to_numpy())}")

cluster_al.fit(x_all_c1)

print(cluster_al.labels_)
print(np.unique(cluster_al.labels_, return_counts=True))
print(f"largest clustered subgroup: {np.argmax(np.unique(cluster_al.labels_, return_counts=True)[1])}")

print(f"clustered subgroup: {np.unique(subjects_c1[cluster_al.labels_ == np.argmax(np.unique(cluster_al.labels_, return_counts=True)[1])].to_numpy())}")


# plt.figure()
# plt.scatter(subjects, cluster_al.labels_)
# plt.show()
