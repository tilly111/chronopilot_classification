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
import shap

from utils.feature_loader import load_eye_tracking_data_tw
from utils.splitting import train_test_split_tw, analysis_test_split_tw
from utils.learner_pipeline import fit_classifier, get_pipeline_for_features, fit_classifier_cf, get_pipeline_from_config
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import joblib

import matplotlib.pyplot as plt


def fit_classifier_parallel(x_analysis, y_analysis, pl_interpretable, use_shap, i_train, i_validation, n_classes):
    x_train = x_analysis.iloc[i_train]
    y_train = y_analysis.iloc[i_train]
    x_validation = x_analysis.iloc[i_validation]
    y_validation = y_analysis.iloc[i_validation]

    trained = clone(pl_interpretable).fit(x_train.values, y_train.values.ravel())
    y_pred = trained.predict(x_validation.values)
    y_pred_proba = trained.predict_proba(x_validation.values)

    if use_shap:  # TODO
        explainer = shap.KernelExplainer(trained.predict_proba, shap.sample(x_train.values, 50))
        shap_value = explainer(x_train.iloc[0:33])  # TODO: shouldnt we use x_validation here?

        # returns probability for class 0 and 1, but we only need one bc q = 1 - p
        shap_value.values = shap_value.values[:, :, 1]
        shap_value.base_values = shap_value.base_values[:, 1]

        shap_value = shap_value.abs.mean(axis=0).values

    # NOTE: ovo and macro insensitive to class inbalance for roc_auc, current solution is sensitive
    roc = roc_auc_score(y_validation, y_pred_proba[:, 1]) if n_classes == 2 else \
          roc_auc_score(y_validation, y_pred_proba, multi_class='ovr', average='macro')

    return accuracy_score(y_validation, y_pred), roc, \
           f1_score(y_validation, y_pred, average='weighted'), trained, None  # shap_value if use_shap else None


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
    use_shap = False
    bls = False  # baseline subtraction
    only_pupil = True  # only use pupil data
    tag = "_bls" if bls else ""
    tag_pupil = "_only_pupil" if only_pupil else ""
    number_of_repeats = 100
    workers = os.cpu_count()

    print(f"setting: {n_classes}, {tw}, {scoring}, {label}, {tag_pupil}")
    if n_classes == 3 and scoring == 'roc_auc':
        scoring_load = "accuracy"
    else:
        scoring_load = scoring
    config = f"{dir_path}/eye_tracking_{n_classes}_classes/autoML_classifiers/{scoring_load}_naml_history_tw_{tw}_label_{label}{tag}{tag_pupil}.csv"
    # sort config by accuracy
    # config = config.sort_values(by="accuracy", ascending=False)
    # pipeline_config = config["pipeline"].iloc[0]
    pl_interpretable = get_pipeline_from_config(config, scoring_load)

    print(pl_interpretable)
    X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, include_meta_label=True,
                                     tw=tw, label_name=[label], bls=bls, only_pupil=only_pupil)
    
    x_analysis, x_test, y_analysis, y_test = analysis_test_split_tw(X, y)
    x_analysis.drop(columns=["slice", "participant", "time", "robot"], inplace=True)
    y_analysis.drop(columns=["participant", "time", "robot"], inplace=True)
    x_test.drop(columns=["slice", "participant", "time", "robot"], inplace=True)
    y_test.drop(columns=["participant", "time", "robot"], inplace=True)

    if use_shap:
        shap_values = pd.DataFrame(data=np.zeros((1, len(x_analysis.columns))), columns=x_analysis.columns)


    acc_all = []
    roc_all = []
    f1_all = []
    classifier_all = []
    pbar = tqdm(total=number_of_repeats)
    cv = StratifiedShuffleSplit(n_splits=number_of_repeats)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fit_classifier_parallel, x_analysis, y_analysis, pl_interpretable, use_shap, i_train, i_validation, n_classes) for i_train, i_validation in cv.split(x_analysis, y_analysis)]

        def _cb(future):
            pbar.update(1)


        for future in futures:
            future.add_done_callback(_cb)

        # await results
        as_completed([f for f in futures])

    # close pbar
    pbar.close()

    for i, future in enumerate(futures):
        acc, roc, f1, classifier, shap_value = future.result()
        acc_all.append(acc)
        roc_all.append(roc)
        f1_all.append(f1)
        classifier_all.append(classifier)
        if use_shap:
            shap_values.loc[i] = shap_value

    #print(f"mean accuracy: {np.mean(acc_all):.4f} $\pm$ {np.std(acc_all):.4f}")  # \u00B1
    #print(f"mean ROC AUC: {np.mean(roc_all):.4f} $\pm$ {np.std(roc_all):.4f}")
    #print(f"mean F1-score: {np.mean(f1_all)}")

    test_accs = []
    test_rocs = []
    test_cm_00, test_cm_01, test_cm_02 = [], [], []
    test_cm_10, test_cm_11, test_cm_12 = [], [], []
    test_cm_20, test_cm_21, test_cm_22 = [], [], []

    for c in classifier_all:
        test_roc = roc_auc_score(y_test, c.predict_proba(x_test.values)[:, 1]) if n_classes == 2 else \
                   roc_auc_score(y_test, c.predict_proba(x_test.values), multi_class='ovr', average='macro')
        test_rocs.append(test_roc)
        test_acc = accuracy_score(y_test, c.predict(x_test.values))
        test_accs.append(test_acc)
        # NOTE: confusion matrix depends on number of classes
        if n_classes == 2:
            test_cm = confusion_matrix(y_test, c.predict(x_test.values), labels=[0, 1])
            test_cm_00.append(test_cm[0, 0])
            test_cm_01.append(test_cm[0, 1])
            test_cm_10.append(test_cm[1, 0])
            test_cm_11.append(test_cm[1, 1])
        else:
            test_cm = confusion_matrix(y_test, c.predict(x_test.values), labels=[0, 1, 2])
            test_cm_00.append(test_cm[0, 0])
            test_cm_01.append(test_cm[0, 1])
            test_cm_02.append(test_cm[0, 2])
            test_cm_10.append(test_cm[1, 0])
            test_cm_11.append(test_cm[1, 1])
            test_cm_12.append(test_cm[1, 2])
            test_cm_20.append(test_cm[2, 0])
            test_cm_21.append(test_cm[2, 1])
            test_cm_22.append(test_cm[2, 2])

    if n_classes == 2:
        save_frame = pd.DataFrame(data={"accuracy": acc_all, "roc_auc": roc_all, "f1_score": f1_all,
                                        "test_accuracy": test_accs, "test_roc_auc": test_rocs,
                                        "test_cm_00": test_cm_00, "test_cm_01": test_cm_01,
                                        "test_cm_10": test_cm_10, "test_cm_11": test_cm_11})
    else:
        save_frame = pd.DataFrame(data={"accuracy": acc_all, "roc_auc": roc_all, "f1_score": f1_all,
                                        "test_accuracy": test_accs, "test_roc_auc": test_rocs,
                                        "test_cm_00": test_cm_00, "test_cm_01": test_cm_01, "test_cm_02": test_cm_02,
                                        "test_cm_10": test_cm_10, "test_cm_11": test_cm_11, "test_cm_12": test_cm_12,
                                        "test_cm_20": test_cm_20, "test_cm_21": test_cm_21, "test_cm_22": test_cm_22})
    save_frame.to_csv(f"results/eye_tracking_{n_classes}_classes/all/{scoring}_eye_tracking_{n_classes}_classes_tw_{tw}_label_{label}_{tag}_{number_of_repeats}{tag_pupil}.csv")
    
    # dump best classifier
    clf = classifier_all[np.argmax(test_accs)]
    joblib.dump(classifier_all[np.argmax(test_accs)],
                f"results/models/{scoring}_eye_tracking_{n_classes}_classes_tw_{tw}_label_{label}_{tag}_{number_of_repeats}{tag_pupil}.pkl",
                compress=1)
    
    if use_shap:
        for clf in classifier_all:
            # explainer = shap.Explainer(clf)
            explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(x_analysis.values, 200))  # x_analysis.values
            shap_values = explainer.shap_values(x_test)
            # shap.summary_plot(shap_values, x_test)
            shap.summary_plot(shap_values[0], x_test)  # Display the summary_plot of the label “0”.
            plt.show()
        # shap_values = shap_values / number_of_repeats
        # shap_values = shap_values.T
        # shap_values["mean"] = shap_values.mean(axis=1)
        # # shap_values = shap_values.rename(columns={0: "shap_values"})
        # shap_values = shap_values.sort_values(by="mean", ascending=False)
        # print(shap_values)
        # shap_values.to_csv(f"results/eye_tracking_{n_classes}_classes/shap/shap_eye_tracking_{n_classes}_classes_tw_{tw}_label_{label}_{number_of_repeats}.csv")