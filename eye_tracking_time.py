import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from utils.splitting import leave_one_subject_out_cv
from utils.feature_loader import load_eye_tracking_data_tw
from utils.learner_pipeline import fit_classifier, get_pipeline_for_features
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

if __name__ == '__main__':
    # params to choose
    n_classes = 2
    tw = 20  # time window in seconds
    label = "ppot"  # "ppot" or "duration_estimate"
    include_meta_label = True
    scoring = "accuracy"  # "roc_auc"?
    m_workers = os.cpu_count()
    print(f"Utilizing {m_workers} worker(s)...")
    num_splits = 10
    bls = True  # baseline subtraction
    tag = "_bls" if bls else ""

    # Load data
    X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True,
                                     include_meta_label=include_meta_label, tw=tw, label_name=[label], bls=bls)

    # Leave one subject out cross-validation
    splits = leave_one_subject_out_cv(X, y, "time")

    # select learner
    learner = ExtraTreesClassifier(criterion='entropy',
                                   max_features=0.9197700535609098,
                                   min_samples_leaf=3, min_samples_split=14,
                                   n_estimators=512, warm_start=True)
    preprocessor = None
    pl_interpretable = get_pipeline_for_features(learner, preprocessor)

    accuracy_result_frame = pd.DataFrame(columns=["accuracy"] * num_splits)

    # fit classifier for each split
    for _, ind_idx in splits:
        if n_classes == 2:
            conf_m_result_frame = pd.DataFrame(columns=["TP", "FN", "FP", "TN"])
        else:
            conf_m_result_frame = pd.DataFrame(columns=["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"])
        time_duration = int(y.iloc[ind_idx]['time'].unique()[0])
        # print(f"train: {train_idx}")
        # print(f"test: {test_idx}")
        print("\n")
        print("--------------------------------------------------")
        # print(f"train on {y.iloc[train_idx]['time'].unique()}")
        print(f"test on {y.iloc[ind_idx]['time'].unique()}")
        # print("--------------------------------------------------")
        # X_train = X.iloc[train_idx].drop(columns=["slice", "slice_fix", "participant", "time", "robot"])
        X_individual_min = X.iloc[ind_idx].drop(columns=["slice", "participant", "time", "robot"])
        # y_train = y.iloc[train_idx].drop(columns=["participant", "time", "robot"])
        y_individual_min = y.iloc[ind_idx].drop(columns=["participant", "time", "robot"])

        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        # print(X_train.columns)
        # print(y_train.columns)
        # print("--------------------------------------------------")

        acc_list = []
        conf_m_list = []
        shap_values_list = []
        futures = []
        pbar = tqdm(total=num_splits)
        with ProcessPoolExecutor(max_workers=m_workers) as executor:
            for seed in range(num_splits):
                X_train, X_test, y_train, y_test = train_test_split(X_individual_min, y_individual_min,
                                                                    stratify=y_individual_min, test_size=0.2,
                                                                    random_state=seed)
                futures.append(
                    executor.submit(
                        fit_classifier, learner, X_train, X_test, y_train, y_test, scoring=scoring, use_shap=False, n_classes=n_classes
                    )
                )


            def _cb(future):
                pbar.update(1)


            for future in futures:
                future.add_done_callback(_cb)

            as_completed(futures)
            for future in futures:
                acc, conf_m_tmp, shap_values = future.result()
                acc_list.append(acc)
                if n_classes == 2:
                    conf_m_result_frame.loc[len(conf_m_result_frame)] = [conf_m_tmp[0, 0], conf_m_tmp[0, 1],
                                                                         conf_m_tmp[1, 0], conf_m_tmp[1, 1]]
                else:
                    conf_m_result_frame.loc[len(conf_m_result_frame)] = [conf_m_tmp[0, 0], conf_m_tmp[1, 0],
                                                                         conf_m_tmp[2, 0], conf_m_tmp[0, 1],
                                                                         conf_m_tmp[1, 1], conf_m_tmp[2, 1],
                                                                         conf_m_tmp[0, 2], conf_m_tmp[1, 2],
                                                                         conf_m_tmp[2, 2]]
                shap_values_list.append(shap_values)
        pbar.close()

        conf_m_result_frame.to_csv(
            f"results/eye_tracking_{n_classes}_classes/times/confusion_matrix_{label}_tw_{tw}_time_{time_duration}{tag}.csv",
            index=False)

        accuracy_result_frame.loc[int(y.iloc[ind_idx]['time'].unique()[0])] = acc_list

    accuracy_result_frame.index.name = "time"
    print(accuracy_result_frame)
    accuracy_result_frame.to_csv(f"results/eye_tracking_{n_classes}_classes/times/accuracy_{label}_tw_{tw}{tag}.csv")
