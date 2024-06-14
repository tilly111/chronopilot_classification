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
from imblearn.under_sampling import NeighbourhoodCleaningRule, CondensedNearestNeighbour

from sklearn.preprocessing import MinMaxScaler

from plotting_scripts.plot_physio import two_sided_violin_plot, plot_physio_violin

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


def build_data(population, block_names, feature_names, study="1", background_block_name="exp_S"):
    x = None
    y = None
    for b in block_names:
        label = pd.read_csv(f"/Volumes/Data/chronopilot/scream_experiment/study1_ts_features/labels/{b}.csv")
        for p in population:
            df_ppg = pd.read_csv(f"/Volumes/Data/chronopilot/scream_experiment/study1_ts_features/{b}/subject-{p}_ppg.csv")
            df_eda = pd.read_csv(f"/Volumes/Data/chronopilot/scream_experiment/study1_ts_features/{b}/subject-{p}_eda.csv")
            df_eda.drop(columns="subject", inplace=True)
            df = pd.concat([df_ppg, df_eda], axis=1)
            df = df[["subject"] + feature_names]
            # df["label"] = label.loc[label["subject"] == p, "label"].values[0]
            if x is None:
                x = df
                y = pd.DataFrame({"subject": np.ones((df.shape[0],)) * p, "block_estimation": np.ones((df.shape[0],)) * label.loc[label["subject"] == p, "block_estimation"].values[0]})
            else:
                x = pd.concat([x, df], axis=0)
                y = pd.concat([y, pd.DataFrame({"subject": np.ones((df.shape[0],)) * p, "block_estimation": np.ones((df.shape[0],)) * label.loc[label["subject"] == p, "block_estimation"].values[0]})], axis=0)

    print(f"shapes x: {x.shape}, y: {y.shape}")
    print(f"columns: {x.columns}, {y.columns}")
    return x, y



# df = pd.read_csv("/Volumes/Data/chronopilot/scream_experiment/study1_ts_features/exp_PU/subject-2_ppg.csv")
#
# print(df)
x, y = build_data(constants.SUBJECTS_STUDY_1, ["exp_T", "exp_MA", "exp_TU", "exp_PU"], constants.ALL_PPG_FEATURES_HEARTPY + constants.ALL_EDA_FEATURES)
x.replace([np.inf, -np.inf], np.nan, inplace=True)
x.fillna(0, inplace=True)
x.reset_index(inplace=True)
plot_physio_violin(constants.ALL_PPG_FEATURES_HEARTPY + constants.ALL_EDA_FEATURES, x, y)

