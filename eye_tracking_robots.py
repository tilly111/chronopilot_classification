import sys

import pandas as pd
import numpy as np
import os
import platform
import matplotlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from utils.feature_loader import load_eye_tracking_data_tw
from utils.splitting import train_test_split_tw, analysis_test_split_tw, leave_one_subject_out_cv
from utils.learner_pipeline import fit_classifier, get_pipeline_for_features, fit_classifier_cf, get_pipeline_from_config
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import matplotlib.pyplot as plt


def fit_classifier_parallel(X, y, pl_interpretable, i_train, i_validation, n_classes):
    x_train = X.iloc[i_train]
    y_train = y.iloc[i_train]
    x_validation = X.iloc[i_validation]
    y_validation = y.iloc[i_validation]

    trained = clone(pl_interpretable).fit(x_train.values, y_train.values.ravel())
    y_pred = trained.predict(x_validation.values)
    y_pred_proba = trained.predict_proba(x_validation.values)

    # NOTE: ovo and macro insensitive to class inbalance for roc_auc, current solution is sensitive
    roc = roc_auc_score(y_validation, y_pred_proba[:, 1]) if n_classes == 2 else \
          roc_auc_score(y_validation, y_pred_proba, multi_class='ovr', average='macro')
    cm = confusion_matrix(y_validation, y_pred)

    return accuracy_score(y_validation, y_pred), roc, \
           f1_score(y_validation, y_pred, average='weighted'), trained, cm


if __name__ == '__main__':
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        plt.rcParams.update({'font.size': 22})
        dir_path = "results/without_histgradient"
    elif platform.system() == "Linux":
        dir_path = "results"
        matplotlib.use('TkAgg')

    # # params to choose
    n_classes = int(sys.argv[1])
    tw = int(sys.argv[2])  # time window in seconds
    label = str(sys.argv[4])  # "ppot" or "duration_estimate"
    scoring = str(sys.argv[3])  # "roc_auc"?
    bls = True  # baseline subtraction
    tag = "_bls" if bls else ""
    number_of_robot = int(sys.argv[5])
    rob_id = 2 * number_of_robot + 1
    number_of_repeats = 100
    workers = os.cpu_count()

    print(f"setting: {n_classes}, {tw}, {scoring}, {label}")
    if n_classes == 3 and scoring == 'roc_auc':
        scoring_load = "accuracy"
    else:
        scoring_load = scoring
    config = f"{dir_path}/eye_tracking_{n_classes}_classes/autoML_classifiers/{scoring_load}_naml_history_tw_{tw}_label_{label}_bls.csv"
    # sort config by accuracy
    # config = config.sort_values(by="accuracy", ascending=False)
    # pipeline_config = config["pipeline"].iloc[0]
    pl_interpretable = get_pipeline_from_config(config, scoring_load)

    print(pl_interpretable)
    X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, include_meta_label=True,
                                     tw=tw, label_name=[label], bls=bls)

    # check what kind of participants are possible
    possible_participants = X[X['robot'] == rob_id]['participant'].unique()
    X = X[X['participant'].isin(possible_participants)]
    y = y[y['participant'].isin(possible_participants)]

    splits = leave_one_subject_out_cv(X, y, "robot")

    # print(X.iloc[splits[0][0]]["robot"].unique())
    # print(X.iloc[splits[0][1]]["robot"].unique())
    train_idxs = splits[number_of_robot][0]
    test_idxs = splits[number_of_robot][1]

    X.drop(columns=["slice", "participant", "time", "robot"], inplace=True)
    y.drop(columns=["participant", "time", "robot"], inplace=True)

    acc_all = []
    roc_all = []
    f1_all = []
    classifier_all = []
    pbar = tqdm(total=number_of_repeats)
    # cv = StratifiedShuffleSplit(n_splits=number_of_repeats)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fit_classifier_parallel, X, y, pl_interpretable, train_idxs, test_idxs, n_classes) for _ in range(number_of_repeats)]

        def _cb(future):
            pbar.update(1)


        for future in futures:
            future.add_done_callback(_cb)

        # await results
        as_completed([f for f in futures])

    # close pbar
    pbar.close()

    test_cm_00, test_cm_01, test_cm_02 = [], [], []
    test_cm_10, test_cm_11, test_cm_12 = [], [], []
    test_cm_20, test_cm_21, test_cm_22 = [], [], []

    for i, future in enumerate(futures):
        acc, roc, f1, classifier, cm = future.result()
        acc_all.append(acc)
        roc_all.append(roc)
        f1_all.append(f1)
        classifier_all.append(classifier)
        if n_classes == 2:
            test_cm_00.append(cm[0, 0])
            test_cm_01.append(cm[0, 1])
            test_cm_10.append(cm[1, 0])
            test_cm_11.append(cm[1, 1])
        else:
            test_cm_00.append(cm[0, 0])
            test_cm_01.append(cm[0, 1])
            test_cm_02.append(cm[0, 2])
            test_cm_10.append(cm[1, 0])
            test_cm_11.append(cm[1, 1])
            test_cm_12.append(cm[1, 2])
            test_cm_20.append(cm[2, 0])
            test_cm_21.append(cm[2, 1])
            test_cm_22.append(cm[2, 2])


    #print(f"mean accuracy: {np.mean(acc_all):.4f} $\pm$ {np.std(acc_all):.4f}")  # \u00B1
    #print(f"mean ROC AUC: {np.mean(roc_all):.4f} $\pm$ {np.std(roc_all):.4f}")
    #print(f"mean F1-score: {np.mean(f1_all)}")

    if n_classes == 2:
        save_frame = pd.DataFrame(data={"accuracy": acc_all, "roc_auc": roc_all, "f1_score": f1_all,
                                        "test_cm_00": test_cm_00, "test_cm_01": test_cm_01,
                                        "test_cm_10": test_cm_10, "test_cm_11": test_cm_11})
    else:
        save_frame = pd.DataFrame(data={"accuracy": acc_all, "roc_auc": roc_all, "f1_score": f1_all,
                                        "test_cm_00": test_cm_00, "test_cm_01": test_cm_01, "test_cm_02": test_cm_02,
                                        "test_cm_10": test_cm_10, "test_cm_11": test_cm_11, "test_cm_12": test_cm_12,
                                        "test_cm_20": test_cm_20, "test_cm_21": test_cm_21, "test_cm_22": test_cm_22})

    save_frame.to_csv(f"{dir_path}/eye_tracking_{n_classes}_classes/active_robot/{scoring}_eye_tracking_{n_classes}_classes_tw_{tw}_label_{label}{tag}_{rob_id}.csv")