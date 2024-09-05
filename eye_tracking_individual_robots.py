import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from utils.splitting import leave_one_subject_out_cv
from utils.feature_loader import load_eye_tracking_data_tw
from utils.learner_pipeline import fit_classifier, get_pipeline_for_features, fit_classifier_cf
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # params to choose
    n_classes = 3
    tw = 20  # time window in seconds
    label = "duration_estimate"  # "ppot" or "duration_estimate"
    include_meta_label = True
    save_results = True
    scoring = "accuracy"  # "roc_auc"?
    m_workers = os.cpu_count()
    print(f"Utilizing {m_workers} worker(s)...")
    num_splits = 10
    bls = True  # baseline subtraction
    tag = "_bls" if bls else ""

    # Load data
    X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True,
                                     include_meta_label=include_meta_label, tw=tw, label_name=[label], bls=bls)
    y_with_meta = y.copy()
    y.drop(columns=["participant", "time", "robot"], inplace=True)
    X.drop(columns=["participant", "time", "robot", "slice"], inplace=True)


    # Leave one subject out cross-validation
    # splits = leave_one_subject_out_cv(X, y, "robot")
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)

    # select learner
    learner = ExtraTreesClassifier(criterion='entropy',
                                   max_features=0.9197700535609098,
                                   min_samples_leaf=3, min_samples_split=14,
                                   n_estimators=512, warm_start=True)
    preprocessor = None
    pl_interpretable = get_pipeline_for_features(learner, preprocessor)

    accuracy_result_frame = pd.DataFrame(columns=["accuracy"] * num_splits)

    acc_list = []
    shap_values_list = []
    futures = []
    # with ProcessPoolExecutor(max_workers=m_workers) as executor:
        # for seed in range(num_splits):
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
        y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)

        acc, _, shap_values, classifier = fit_classifier_cf(learner, X_train, X_test, y_train, y_test, scoring=scoring, use_shap=False, n_classes=n_classes)

        # individual robots
        y_preds = classifier.predict(X_test.values)
        print(np.unique(y_with_meta["robot"]))
        if n_classes == 2:
            res_robots = pd.DataFrame(columns=["count", "TP", "FN", "FP", "TN"], index=np.unique(y_with_meta["robot"]).astype(int), data=0)
        else:
            res_robots = pd.DataFrame(columns=["count", "P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"], index=np.unique(y_with_meta["robot"]).astype(int), data=0)
        # mapping is positive = slow/underestimation, negative = fast/overestimation
        res_robots.index.name = "robots"
        for j, idx in enumerate(test_index):
            # res_robots.iloc[y_with_meta.iloc[idx]["robot"]]
            if n_classes == 2:
                if y_with_meta.iloc[idx][label] == 0:
                    if y_with_meta.iloc[idx][label] == y_preds[j]:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "TP"] += 1
                    else:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "FN"] += 1
                elif y_with_meta.iloc[idx][label] == 1:
                    if y_with_meta.iloc[idx][label] == y_preds[j]:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "TN"] += 1
                    else:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "FP"] += 1
                res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "count"] += 1
            else:
                if y_with_meta.iloc[idx][label] == 0:
                    if y_preds[j] == 0:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P11"] += 1
                    elif y_preds[j] == 1:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P12"] += 1
                    elif y_preds[j] == 2:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P13"] += 1
                elif y_with_meta.iloc[idx][label] == 1:
                    if y_preds[j] == 0:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P21"] += 1
                    elif y_preds[j] == 1:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P22"] += 1
                    elif y_preds[j] == 2:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P23"] += 1
                elif y_with_meta.iloc[idx][label] == 2:
                    if y_preds[j] == 0:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P31"] += 1
                    elif y_preds[j] == 1:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P32"] += 1
                    elif y_preds[j] == 2:
                        res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "P33"] += 1
                res_robots.loc[int(y_with_meta.iloc[idx]["robot"]), "count"] += 1

        if n_classes == 2:
            res_robots["accuracy"] = (res_robots["TP"] + res_robots["TN"]) / res_robots["count"]
        else:
            res_robots["accuracy"] = (res_robots["P11"] + res_robots["P22"] + res_robots["P33"]) / res_robots["count"]
        print(res_robots)


        acc_list.append(acc)
        shap_values_list.append(shap_values)

        if save_results:
            res_robots.to_csv(
                f"results/eye_tracking_{n_classes}_classes/individual_robots/confusion_matrix_{label}_tw_{tw}_split_{i}{tag}.csv")

    # accuracy_result_frame.index.name = "robot"
    print(accuracy_result_frame)
    # if save_results:
    #     accuracy_result_frame.to_csv(f"results/eye_tracking_{n_classes}_classes/individual_robots/accuracy_{label}_tw_{tw}.csv")