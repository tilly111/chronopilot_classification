import constants
import platform
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib
from itertools import compress, combinations, product

from sympy.combinatorics.subsets import ksubsets

# from imblearn.over_sampling import BorderlineSMOTE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import shap
from sklearn.cluster import KMeans

from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_slice
from sklearn.model_selection import train_test_split

from plotting_scripts.plot_physio import plot_physio_violin

from plotting_scripts.plot_classifier import plot_classifier

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')

split = "participant"
label = "duration_estimate"  # "ppot" or "duration_estimate"
n_classes = 3
bls = True  # baseline subtraction
tag = "_bls" if bls else ""

if split == "only_one_minute":
    res = pd.read_csv(f"results/eye_tracking_{n_classes}_classes/{split}/accuracy_{label}_tw_20{tag}.csv", index_col=f"{split}")
    # conf_m_1 = pd.read_csv(f"results/eye_tracking_{n_classes}_classes/{split}/confusion_matrix_{label}_tw_20_minute_1.csv")
else:
    res = pd.read_csv(f"results/eye_tracking_{n_classes}_classes/{split}s/accuracy_{label}_tw_20{tag}.csv", index_col=f"{split}")
    # conf_m_1 = pd.read_csv(f"results/eye_tracking_{n_classes}_classes/{split}s/confusion_matrix_{label}_tw_20_{split}_1.csv")


if split == "time":
    fig, ax = plt.subplots()
    ax.boxplot([res.iloc[0], res.iloc[1], res.iloc[2]], labels=["1", "3", "5"])
    plt.xlabel("Time (min)")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
if split == "only_one_minute":
    fig, ax = plt.subplots()
    ax.boxplot([res.iloc[0], res.iloc[1], res.iloc[2], res.iloc[3], res.iloc[4]], labels=["1", "2", "3", "4", "5"])
    plt.xlabel("Minute of the experiment (min)")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
if split == "participant":
    print(res.shape)
    fig, ax = plt.subplots()
    ax.boxplot([res.iloc[0], res.iloc[1], res.iloc[2], res.iloc[3], res.iloc[4], res.iloc[5], res.iloc[6],
                res.iloc[7], res.iloc[8], res.iloc[9], res.iloc[10], res.iloc[11], res.iloc[12], res.iloc[13],
                res.iloc[14], res.iloc[15], res.iloc[16], res.iloc[17]], labels=res.index)  #
    plt.xlabel("Participant")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
    exit(2)


# print(conf_m_1.sum(axis=1))
# for robot in [1, 3, 5, 7, 9, 11, 13, 15]:
for robot in [1, 2, 3, 4, 5]:
    if split == "only_one_minute":
        res = pd.read_csv(f"results/eye_tracking_2_classes/{split}/accuracy_{label}_tw_20.csv", index_col=f"{split}")
        conf_m_1 = pd.read_csv(f"results/eye_tracking_2_classes/{split}/confusion_matrix_{label}_tw_20_minute_{robot}.csv")
    else:
        res = pd.read_csv(f"results/eye_tracking_2_classes/{split}s/accuracy_{label}_tw_20.csv", index_col=f"{split}")
        conf_m_1 = pd.read_csv(f"results/eye_tracking_2_classes/{split}s/confusion_matrix_{label}_tw_20_{split}_{robot}.csv")

    acc_list = []
    for i in range(conf_m_1.shape[0]):
        acc = (conf_m_1.iloc[i]["TP"] + conf_m_1.iloc[i]["TN"]) / (conf_m_1.iloc[i]["TP"] + conf_m_1.iloc[i]["TN"] + conf_m_1.iloc[i]["FP"] + conf_m_1.iloc[i]["FN"])
        # print(
        #     f"distribution: {conf_m_1.iloc[i]['TP'] + conf_m_1.iloc[i]['FN']}, {conf_m_1.iloc[i]['FP'] + conf_m_1.iloc[i]['TN']}")
        # print(f"acc: {acc}; class split: {(conf_m_1.iloc[i]['TP'] + conf_m_1.iloc[i]['FN'])/conf_m_1.iloc[i].sum()}, {(conf_m_1.iloc[i]['FP'] + conf_m_1.iloc[i]['TN'])/conf_m_1.iloc[i].sum()}")
        acc_list.append(acc)
    print(f"{split}(s) {robot}: accuracy mean: {np.mean(acc_list):.4f} -- distribution {(conf_m_1.iloc[0]['TP'] + conf_m_1.iloc[0]['FN'])/conf_m_1.iloc[0].sum():.2f}, {(conf_m_1.iloc[0]['FP'] + conf_m_1.iloc[0]['TN'])/conf_m_1.iloc[0].sum():.2f}")