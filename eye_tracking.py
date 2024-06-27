import os

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import platform

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import LearningCurveDisplay

from utils.learner_pipeline import get_pipeline_for_features

from sklearn.feature_selection import VarianceThreshold
from plotting_scripts.roc_curve_plotting import get_mccv_ROC_display

from imblearn.over_sampling import BorderlineSMOTE

from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_slice

from sklearn.metrics import auc, get_scorer

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

import shap

from plotting_scripts.plot_physio import plot_physio3D, plot_physio2D


# load data
# X, y = load_eye_tracking_data(make_binary=True, load_preprocessed=True)
# X.drop(columns=["participant", "time", "robot"], inplace=True)


def fit_classifer(learner, x_train, x_test, y_train, y_test, n_classes=2):
    # todo adjust accordingly to the number of classes
    # for two classes: ExtraTreesClassifier: bootstrap=True, max_features=0.4908846305986305, n_estimators=512, warm_start=True
    #  increase parameters because 512 is upper limit of autoML and more trees = better ;)
    # learner = ExtraTreesClassifier(bootstrap=True, max_features=0.4908846305986305, n_estimators=1024, warm_start=True)
    #learner = ExtraTreesClassifier(n_estimators=1024)
    # data_pre_processor = VarianceThreshold()
    learner_c = clone(learner)

    learner_c.fit(x_train, y_train.values.ravel())

    y_pred = learner_c.predict(x_test)
    if n_classes == 2:
        scorer = get_scorer("accuracy")  # roc_auc
    else:
        scorer = get_scorer("accuracy")  # roc_auc_ovr

    # return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), pl_interpretable
    return scorer(learner_c, x_test, y_test), confusion_matrix(y_test, y_pred)  # , pl_interpretable

if __name__ == '__main__':
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
    elif platform.system() == "Linux":
        matplotlib.use('TkAgg')
    num_splits = 500
    n_classes = 2

    X, y = load_eye_tracking_data(number_of_classes=n_classes, load_preprocessed=True)
    # X, y = load_eye_tracking_data_slice(number_of_classes=n_classes, load_preprocessed=True)

    # use indiviual times experiments
    # y = y.loc[y["time"] == 5]
    # X = X.loc[X["time"] == 5]
    # y.drop(columns=["time", "robot", "participant"], inplace=True)
    # X.drop(columns=["time", "robot", "participant"], inplace=True)

    # use preprocessing: the best subset
    # X = X[['sub_max_diameter2d', 'sub_max_speed_fix', 'sub_mean_duration_fix', 'sub_mean_speed', 'sub_mean_speed_fix', 'sub_min_diameter2d', 'sub_number_clusters_fix']]
    # pca = PCA()  # n_components=7
    # X_pp = pca.fit_transform(X)
    # X = pd.DataFrame(X_pp, columns=[f"PCA_{i}" for i in range(7)])

    # upsampling the data
    # sm = BorderlineSMOTE(random_state=42)  # random_state=42
    # X, y = sm.fit_resample(X, y)

    print(f"X data shape: {X.shape}")
    print(f"y data shape: {y.shape}\n")
    print(f"distribution of y: {np.unique(y, return_counts=True)}")
    _, counts = np.unique(y, return_counts=True)
    majority_class = np.max(counts)/X.shape[0]
    # plt.hist(y)
    # plt.show()
    learner = ExtraTreesClassifier(n_estimators=1024)
    preprocessor = MinMaxScaler()
    pl_interpretable = get_pipeline_for_features(learner, preprocessor)

    ## play trough
    # sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)

    acc_list = []
    futures = []
    conf_m = np.zeros((n_classes, n_classes))

    pbar = tqdm(total=num_splits)
    m_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=m_workers) as executor:
        # for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        for seed in range(num_splits):
            X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=0.8, random_state=seed)
            # print(f"Fold {i}")
            #x_train, x_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
            #y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)
            # upsampling the data
            # sm = BorderlineSMOTE()  # random_state=42
            # x_train, y_train = sm.fit_resample(x_train, y_train)

            futures.append(executor.submit(
                fit_classifer, learner,
                X_train,
                X_val,
                y_train,
                y_val,
                n_classes))
        def _cb(future):
            pbar.update(1)

        for future in futures:
            future.add_done_callback(_cb)

        as_completed(futures)

        for future in futures:
            acc, conf_m_tmp = future.result()  # , learner
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
    plt.figure()
    plt.hist(acc_list, label="Accuracy")
    plt.xlabel("Accuracy")  # 0.5410447761
    upper_lim = np.max(np.unique(acc_list, return_counts=True)[1])*10
    plt.vlines(majority_class, 0, upper_lim, colors="red", label="Majority class", linestyles="--")
    plt.legend()
    plt.savefig(
        f"plots/eye_tracking_analysis/accuracy_hist_repeats_{num_splits}_extra_tree_{n_classes}_classes_n_features:{X.shape[0]}.pdf")

    if n_classes == 2:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_m,
                                      display_labels=["slow", "fast"])
    elif n_classes == 3:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_m,
                                      display_labels=["slow", "medium", "fast"])

    disp.plot()
    plt.savefig(
        f"plots/eye_tracking_analysis/confusion_matrix_repeats_{num_splits}_extra_tree_{n_classes}_classes_n_features:{X.shape[1]}.pdf")
    # plt.show()

    # plot roc auc curve
    # learner = ExtraTreesClassifier(n_estimators=number_trees)
    # pl_interpretable = get_pipeline_for_features(learner, X, y, list(X.columns))

    # fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    # get_mccv_ROC_display(pl_interpretable, X, y, repeats=num_splits, ax=axs)  #
    # plt.savefig(f"plots/eye_tracking_analysis/roc_curve_repeats_{num_splits}_extra_tree_{n_classes}_classes_n_features:{X.shape[0]}.pdf")
    plt.show()

