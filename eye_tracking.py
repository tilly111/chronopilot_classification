import pandas as pd
import numpy as np
import os
import platform
import constants
import matplotlib
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import shap
from utils.splitting import analysis_test_split_tw

from utils.splitting import leave_one_subject_out_cv
from utils.feature_loader import load_eye_tracking_data_tw, load_eye_tracking_data_baseline
from utils.splitting import train_test_split_tw
from utils.learner_pipeline import fit_classifier, get_pipeline_for_features, fit_classifier_cf, \
    get_pipeline_from_config
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import matplotlib.pyplot as plt

matplotlib.use('QtAgg')
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)


n_classes = 2
tw = 2  # time window in seconds
label = "duration_estimate"  # "ppot" or "duration_estimate"
scoring = "accuracy"  # "roc_auc"?
use_shap = False
bls = True  # baseline subtraction
tag = "_bls" if bls else ""
dir_path = "results/without_histgradient"
number_of_repeats = 1
init_length = 30  # in seconds
K = 50

X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, include_meta_label=True,
                                 tw=tw, label_name=[label], bls=True)
x_analysis, x_test, y_analysis, y_test = analysis_test_split_tw(X, y)


# X = X[X["time"] == 5]
# y = y[y["time"] == 5]

# x_analysis.drop(columns=["slice", "participant", "time", "robot"], inplace=True)
# y_analysis.drop(columns=["participant", "time", "robot"], inplace=True)

config = f"{dir_path}/eye_tracking_{n_classes}_classes/autoML_classifiers/{scoring}_naml_history_tw_{tw}_label_{label}_bls.csv"
pl_interpretable = get_pipeline_from_config(config, scoring)

# trained = clone(pl_interpretable).fit(x_analysis.values, y_analysis.values.ravel())
# NOTE only person does not work at all << 50 % accuracy
# NOTE with some data of a person it works like a charm
shap_save_frame = None
metrics_save_frame = None
for p in x_test["participant"].unique():
    # for p in [18]:
    # NOTE select how to select the training data
    # x_train = X[((X["participant"] != p) | (X["slice"] < init_length / tw))].copy()
    # y_train = y[((y["participant"] != p) | (X["slice"] < init_length / tw))].copy()
    # x_test = X[((X["participant"] == p) & (X["slice"] >= init_length / tw))].copy()
    # y_test = y[((y["participant"] == p) & (X["slice"] >= init_length / tw))].copy()
    x_train = X[((X["participant"] == p) & (X["slice"] < init_length / tw))].copy()
    y_train = y[((y["participant"] == p) & (X["slice"] < init_length / tw))].copy()
    x_test = X[((X["participant"] == p) & (X["slice"] >= init_length / tw))].copy()
    y_test = y[((y["participant"] == p) & (X["slice"] >= init_length / tw))].copy()
    # x_train = X[((X["participant"] != p))].copy()
    # y_train = y[((y["participant"] != p))].copy()
    # x_test = X[((X["participant"] == p))].copy()
    # y_test = y[((y["participant"] == p))].copy()

    x_test.drop(columns=["slice", "participant", "time", "robot"], inplace=True)
    y_test.drop(columns=["participant", "time", "robot"], inplace=True)
    x_train.drop(columns=["slice", "participant", "time", "robot"], inplace=True)
    y_train.drop(columns=["participant", "time", "robot"], inplace=True)

    trained = clone(pl_interpretable).fit(x_train.values, y_train.values.ravel())

    y_pred = trained.predict(x_test.values)
    y_pred_proba = trained.predict_proba(x_test.values)

    # roc = roc_auc_score(y_test, y_pred_proba[:, 1]) if n_classes == 2 else 0
    print(f"participant {p}: {accuracy_score(y_test, y_pred)}, {f1_score(y_test, y_pred, average='weighted')}")
    print(f"train labels: {y_train[label].value_counts()}")
    print(f"test labels: {y_test[label].value_counts()}")
    print(f"# different labels: {y_test[label].value_counts().shape[0]}")
    # print(f"predicted labels: {y_pred}")
    c_m = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(c_m)
    if y_test[label].value_counts().shape[0] > 1:
        metrics_save_frame = pd.DataFrame(
            data=[accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), c_m[0, 0], c_m[0, 1],
                  c_m[1, 0], c_m[1, 1]], index=["accuracy", "f1", "tp", "fp", "fn", "tn"],
            columns=[f"participant_{p}"]) if metrics_save_frame is None else pd.concat(
            [metrics_save_frame, pd.DataFrame(
                data=[accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), c_m[0, 0],
                      c_m[0, 1],
                      c_m[1, 0], c_m[1, 1]], index=["accuracy", "f1", "tp", "fp", "fn", "tn"],
                columns=[f"participant_{p}"])], axis=1)

    # TODO individual shap values:: KernelExplainer
    if use_shap:
        # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        explainer = shap.Explainer(trained.predict, x_test)  # x_analysis.values
        shap_values = explainer(x_test)
        # print(shap_values.feature_names)
        mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        # print(mean_shap_values)
        shap_save_frame = pd.DataFrame(data=mean_shap_values, index=shap_values.feature_names,
                                       columns=[f"participant_{p}"]) if shap_save_frame is None else pd.concat(
            [shap_save_frame,
             pd.DataFrame(data=mean_shap_values, index=shap_values.feature_names, columns=[f"participant_{p}"])],
            axis=1)
        # for i, feature in enumerate(shap_values.feature_names):
        #     print(f"{feature}: {mean_shap_values[i]}")
        # shap.plots.bar(shap_values, show=True)
        # fig, axs = plt.subplots(2)
        # plt.axes(axs[0])
        # print(shap_values.shape)
        # axs[0].set_title(f"participant {p} wrong")
        shap.plots.beeswarm(shap_values, show=True)  # Display the summary_plot of the label “0”.
        #
        # explainer = shap.Explainer(trained.predict, x_test)  # x_analysis.values
        # shap_values = explainer(x_test_correct)
        # plt.axes(axs[1])
        # axs[1].set_title(f"participant {p} correct")
        # shap.plots.beeswarm(shap_values, show=False)  # Display the summary_plot of the label “0”.

        # plt.show()
print(metrics_save_frame)

print(f"mean accuracy: {metrics_save_frame.loc['accuracy'].mean()}")
print(f"mean f1: {metrics_save_frame.loc['f1'].mean()}")
# if use_shap:
#     print(shap_save_frame)
#     shap_save_frame.to_csv(f"/Users/tillaust/PycharmProjects/chronopilot_classification/results/shap/individuals_{n_classes}_{label}_{tw}_bls.csv")
