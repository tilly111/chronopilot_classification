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
from utils.splitting import train_test_split_tw
from utils.learner_pipeline import fit_classifier, get_pipeline_for_features, fit_classifier_cf, get_pipeline_from_config
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import matplotlib.pyplot as plt


def fit_classifier_parallel(x_analysis, y_analysis, pl_interpretable, use_shap, i_train, i_validation):
    x_train = x_analysis.iloc[i_train]
    y_train = y_analysis.iloc[i_train]
    x_validation = x_analysis.iloc[i_validation]
    y_validation = y_analysis.iloc[i_validation]

    trained = clone(pl_interpretable).fit(x_train.values, y_train.values.ravel())
    y_pred = trained.predict(x_validation)
    y_pred_proba = trained.predict_proba(x_validation)

    if use_shap:
        explainer = shap.KernelExplainer(trained.predict_proba, shap.sample(x_train.values, 50))
        shap_value = explainer(x_train.iloc[0:33])  # TODO: shouldnt we use x_validation here?

        # returns probability for class 0 and 1, but we only need one bc p = 1 - p
        shap_value.values = shap_value.values[:, :, 1]
        shap_value.base_values = shap_value.base_values[:, 1]

        shap_value = shap_value.abs.mean(axis=0).values

    return accuracy_score(y_validation, y_pred), roc_auc_score(y_validation, y_pred_proba[:, 1]), \
           f1_score(y_validation, y_pred, average='weighted'), shap_value if use_shap else None


if __name__ == '__main__':
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        plt.rcParams.update({'font.size': 22})
    elif platform.system() == "Linux":
        matplotlib.use('TkAgg')

    # # params to choose
    n_classes = 2
    tw = 20  # time window in seconds
    label = "duration_estimate"  # "ppot" or "duration_estimate"
    scoring = "accuracy"  # "roc_auc"?
    use_shap = True
    bls = True  # baseline subtraction
    tag = "_bls" if bls else ""
    number_of_repeats = 2

    config = f"results/eye_tracking_{n_classes}_classes/autoML_classifiers/naml_history_eye_tracking_{n_classes}_classes_tw_{tw}_label_{label}.csv"
    # sort config by accuracy
    # config = config.sort_values(by="accuracy", ascending=False)
    # pipeline_config = config["pipeline"].iloc[0]
    pl_interpretable = get_pipeline_from_config(config, scoring)

    print(pl_interpretable)
    X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, include_meta_label=True,
                                     tw=tw, label_name=[label], bls=bls)
    x_analysis, _, y_analysis, _ = train_test_split_tw(X, y)
    x_analysis.drop(columns=["slice", "participant", "time", "robot"], inplace=True)
    y_analysis.drop(columns=["participant", "time", "robot"], inplace=True)

    if use_shap:
        shap_values = pd.DataFrame(data=np.zeros((1, len(x_analysis.columns))), columns=x_analysis.columns)


    acc_all = []
    roc_all = []
    f1_all = []
    pbar = tqdm(total=number_of_repeats)
    futures = []
    cv = StratifiedShuffleSplit(n_splits=number_of_repeats)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # for i_train, i_validation in cv.split(x_analysis, y_analysis):
        #
        #     futures.append(executor.submit(
        #         fit_classifier, x_analysis, y_analysis, pl_interpretable, use_shap, i_train, i_validation
        #          )
        #     )

        futures = [executor.submit(fit_classifier_parallel, x_analysis, y_analysis, pl_interpretable, use_shap, i_train, i_validation) for i_train, i_validation in cv.split(x_analysis, y_analysis)]

        def _cb(future):
            pbar.update(1)


        for future in futures:
            future.add_done_callback(_cb)

        # await results
        as_completed([f for f in futures])

    # close pbar
    pbar.close()

    for i, future in enumerate(futures):
        acc, roc, f1, shap_value = future.result()

        acc_all.append(acc)
        roc_all.append(roc)
        f1_all.append(f1)
        if use_shap:
            shap_values.loc[i] = shap_value

    print(f"mean accuracy: {np.mean(acc_all):.4f} $\pm$ {np.std(acc_all):.4f}")  # \u00B1
    print(f"mean ROC AUC: {np.mean(roc_all):.4f} $\pm$ {np.std(roc_all):.4f}")
    print(f"mean F1-score: {np.mean(f1_all)}")

    save_frame = pd.DataFrame(data={"accuracy": acc_all, "roc_auc": roc_all, "f1_score": f1_all})
    save_frame.to_csv(f"results/eye_tracking_{n_classes}_classes/all/metrics_eye_tracking_{n_classes}_classes_tw_{tw}_label_{label}_{tag}_{number_of_repeats}.csv")

    if use_shap:
        shap_values = shap_values / number_of_repeats
        shap_values = shap_values.T
        shap_values["mean"] = shap_values.mean(axis=1)
        # shap_values = shap_values.rename(columns={0: "shap_values"})
        shap_values = shap_values.sort_values(by="mean", ascending=False)
        print(shap_values)
        shap_values.to_csv(f"results/eye_tracking_{n_classes}_classes/shap/shap_eye_tracking_{n_classes}_classes_tw_{tw}_label_{label}_{number_of_repeats}.csv")