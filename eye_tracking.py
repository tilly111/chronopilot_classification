import os

import constants
import platform
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import torch as t
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress, combinations
from sklearn.decomposition import PCA

from sympy.combinatorics.subsets import ksubsets

from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
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

from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_slice

from sklearn.preprocessing import MinMaxScaler

import shap

from plotting_scripts.plot_physio import plot_physio3D, plot_physio2D


# load data
# X, y = load_eye_tracking_data(make_binary=True, load_preprocessed=True)
# X.drop(columns=["participant", "time", "robot"], inplace=True)


def fit_classifer(x_train, x_test, y_train, y_test, number_trees=1000):
    learner = ExtraTreesClassifier(n_estimators=number_trees)
    # learner = SVC(kernel="rbf", C=1.0, probability=True)

    pl_interpretable = get_pipeline_for_features(learner, x_train, y_train, list(x_train.columns))

    pl_interpretable.fit(x_train, y_train.values.ravel())

    y_pred = pl_interpretable.predict(x_test)
    # a_list.append(accuracy_score(y_test, y_pred))

    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), pl_interpretable

if __name__ == '__main__':
    matplotlib.use('QtAgg')
    ## select hyperparameters
    num_splits = 500
    roc_repeats = 500
    number_trees = 1000
    n_classes = 2

    # X, y = load_eye_tracking_data(number_of_classes=n_classes, load_preprocessed=False, include_meta_label=True)
    X, y = load_eye_tracking_data_slice(number_of_classes=n_classes, load_preprocessed=True)

    # use indiviual times experiments
    # y = y.loc[y["time"] == 5]
    # X = X.loc[X["time"] == 5]
    # y.drop(columns=["time", "robot", "participant"], inplace=True)
    # X.drop(columns=["time", "robot", "participant"], inplace=True)

    # use preprocessing: the best subset
    # X = X[['sub_max_speed_fix', 'sub_mean_dispersion_fix', 'sub_mean_duration_fix', 'sub_mean_speed', 'sub_min_dispersion_fix', 'sub_min_speed_fix', 'sub_number_clusters_fix']]
    # pca = PCA()  # n_components=7
    # X_pp = pca.fit_transform(X)
    # X = pd.DataFrame(X_pp, columns=[f"PCA_{i}" for i in range(7)])

    # upsampling the data
    # sm = BorderlineSMOTE(random_state=42)  # random_state=42
    # X, y = sm.fit_resample(X, y)

    print(f"X data shape: {X.shape}")
    print(f"y data shape: {y.shape}\n")
    print(f"distribution of y: {np.unique(y, return_counts=True)}")
    plt.hist(y)
    plt.show()

    ## play trough
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)

    acc_list = []
    futures = []
    conf_m = np.zeros((n_classes, n_classes))

    pbar = tqdm(total=num_splits)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            # print(f"Fold {i}")
            x_train, x_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
            y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)
            # upsampling the data
            sm = BorderlineSMOTE()  # random_state=42
            x_train, y_train = sm.fit_resample(x_train, y_train)

            futures.append(executor.submit(
                fit_classifer,
                x_train,
                x_test,
                y_train,
                y_test))
        def _cb(future):
            pbar.update(1)

        for future in futures:
            future.add_done_callback(_cb)

        as_completed(futures)

        for future in futures:
            acc, conf_m_tmp, learner = future.result()
            acc_list.append(acc)
            conf_m += conf_m_tmp
            # todo get best lerner and do shap analysis
        conf_m /= num_splits
    pbar.close()
    print("\n")
    print(f"Mean accuracy: {np.mean(acc_list)}")
    print(f"Std accuracy: {np.std(acc_list)}")
    print(f"Max accuracy: {np.max(acc_list)}")
    print(f"Min accuracy: {np.min(acc_list)}")

    if n_classes == 2:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_m,
                                      display_labels=["slow", "fast"])
    elif n_classes == 3:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_m,
                                      display_labels=["slow", "medium", "fast"])

    disp.plot()
    plt.show()


    # plot roc auc curve
    # learner = ExtraTreesClassifier(n_estimators=number_trees)
    # pl_interpretable = get_pipeline_for_features(learner, X, y, list(X.columns))
    #
    # fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    # get_mccv_ROC_display(pl_interpretable, X, y, repeats=roc_repeats, ax=axs)
    # # plt.savefig(f"plots/eye_tracking_analysis/roc_curve_repeats_{roc_repeats}_extra_tree_{number_trees}_all_features.pdf")
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
