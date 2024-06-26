import platform
import constants
import os
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import auc, get_scorer
from sklearn.base import clone

from utils.learner_pipeline import get_pipeline_for_features

from utils.feature_loader import load_eye_tracking_data

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')
elif platform.system() == "Windows":
    # TODO
    pass

def get_learning_curve(learner, X, y, seed, schedule, n_classes):
    auc_train = []
    auc_val = []
    for i, a in enumerate(schedule):
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=a, stratify=y, random_state=seed)
        l_copy = clone(learner)
        l_copy.fit(X_train, y_train.values.ravel())
        if n_classes == 2:
            auc_train.append(roc_auc_score(y_train, l_copy.predict_proba(X_train)[:, 1]))
            auc_val.append(roc_auc_score(y_val, l_copy.predict_proba(X_val)[:, 1]))
        else:
            auc_train.append(roc_auc_score(y_train, l_copy.predict_proba(X_train), multi_class="ovr", average='macro'))
            auc_val.append(roc_auc_score(y_val, l_copy.predict_proba(X_val), multi_class="ovr", average='macro'))
    return auc_train, auc_val

def get_learning_curves(learner, X, y, repeats, n_classes, first_anchor=0.1, last_anchor=0.8, steps=10, filename=None):
    schedule = np.linspace(first_anchor, last_anchor, steps)

    if filename is None or not os.path.isfile(filename):
        lcs = None
    else:
        lcs = pd.read_csv(filename)

    pbar = tqdm(total=repeats)
    futures = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for seed in range(repeats):
            if lcs is None or len(lcs) < seed:
                futures.append(
                    (seed,
                     executor.submit(
                         get_learning_curve,
                         learner,
                         X,
                         y,
                         seed,
                         schedule,
                         n_classes
                     )
                     )
                )
            else:
                pbar.update(1)

        # Attach the callback to each future
        def _cb(future):
            pbar.update(1)

        for seed, future in futures:
            future.add_done_callback(_cb)

        # await results
        as_completed([f for k, f in futures])

    # close pbar
    pbar.close()

    # store results
    if futures:
        lcs_new = pd.DataFrame([future.result()[1] for seed, future in futures], columns=schedule)
        lcs = lcs_new if lcs is None else pd.concat([lcs, lcs_new], axis=0)
    lcs.to_csv(filename, index=False)

    return lcs

# does the fitting of the model and returns the score
def get_score_for_features(classifier, X, y, feature_list, repeats, n_classes):
    X_red = X[feature_list]
    pl_interpretable = get_pipeline_for_features(classifier, X, y, feature_list)
    if n_classes == 2:
        scorer = get_scorer("roc_auc")
    else:
        scorer = get_scorer("roc_auc_ovr")

    results = []
    for seed in range(repeats):
        X_train, X_val, y_train, y_val = train_test_split(X_red, y, stratify=y, train_size=0.8, random_state=seed)
        l = clone(pl_interpretable).fit(X_train, y_train.values.ravel())
        results.append(scorer(l, X_val, y_val))
    return results

# schedules getting the scores for the feature combinations
def get_scores_for_feature_combinations_based_on_previous_selections(classifier, X, y, repeats_per_size, df_last_stage,
                                                                     num_combos_from_last_stage, n_classes):
    if df_last_stage is None:
        combos_for_k = [[c] for c in X.columns]
    else:
        accepted_combos_last_stage = df_last_stage["combo"][:num_combos_from_last_stage]
        combos_for_k = []
        for accepted_prev_combo in accepted_combos_last_stage:
            for c in X.columns:
                if c not in accepted_prev_combo:  # and c > accepted_prev_combo[-1]:
                    new_combo = sorted(accepted_prev_combo + [c])
                    if new_combo not in combos_for_k:
                        combos_for_k.append(new_combo)

    pbar = tqdm(total=len(combos_for_k))
    rows = []

    # Using ThreadPoolExecutor to parallelize the function calls
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        # Submit tasks to the executor
        futures = [
            executor.submit(get_score_for_features, classifier, X, y, combo, repeats_per_size[len(combo)], n_classes)
            for combo in combos_for_k
        ]

        # Attach the callback to each future
        def _cb(future):
            pbar.update(1)

        for future in futures:
            future.add_done_callback(_cb)

        # await results
        as_completed(futures)

        # Process results as they complete
        for combo, future in zip(combos_for_k, futures):
            scores_for_combo = future.result()
            rows.append([combo, scores_for_combo, np.mean(scores_for_combo)])

    pbar.close()

    return pd.DataFrame(rows, columns=["combo", "scores", "score_mean"]).sort_values("score_mean", ascending=False)


def get_scores_for_feature_combinations(classifier, X, y, max_size, repeats_per_size, num_combos_from_last_stage, n_classes):
    dfs = {}

    for k in range(1, max_size + 1):

        path = f"results/eye_tracking_{n_classes}_classes/feature_combinations/feature_selection_results_{k}.csv"
        if os.path.isfile(path):
            dfs[k] = pd.read_csv(path)
            dfs[k]["combo"] = [json.loads(e.replace("'", '"')) for e in dfs[k]["combo"]]
            dfs[k]["scores"] = [json.loads(e) for e in dfs[k]["scores"]]
        else:

            combos_from_last_stage = None if k == 1 else [set()]
            if k == 1:
                df_for_last_k = None
            if k > 1:
                df_for_last_k = dfs[k - 1]  # .drop(columns=attributes_excluded_in_multivar_importance)
            dfs[k] = get_scores_for_feature_combinations_based_on_previous_selections(classifier, X, y,
                                                                                      repeats_per_size, df_for_last_k,
                                                                                      num_combos_from_last_stage[
                                                                                          k] if k > 1 else 0, n_classes)
        dfs[k].to_csv(path, index=False)
    return dfs


## TODO from felix

if __name__ == '__main__':
    ## select hyperparameters
    number_trees = 1000
    n_classes = 3

    # load data
    X, y = load_eye_tracking_data(number_of_classes=n_classes, load_preprocessed=True)
    # X.drop(columns=["participant", "time", "robot"], inplace=True)  # drop setting information
    max_feature_set_size = X.shape[1]

    learner = ExtraTreesClassifier(n_estimators=number_trees)

    df_auc_results_per_feature_combo = get_scores_for_feature_combinations(
        learner,
        X,
        y,
        max_feature_set_size,
        repeats_per_size={i: 5 for i in range(1, max_feature_set_size + 1)},
        num_combos_from_last_stage={i: 10 if i < 10 else (5 if i < 20 else 2) for i in range(2, max_feature_set_size + 1)},
        n_classes=n_classes
    )

    k_s = list(range(1, len(df_auc_results_per_feature_combo) + 1))
    best_scores_per_k = []
    best_combos_per_k = []
    print(f"shape: {len(df_auc_results_per_feature_combo)}")

    for k in k_s:
        # print(k)
        df_fs = df_auc_results_per_feature_combo[k]
        # print(df_fs.iloc[0])
        # TODO why is this so????
        try:
            best_scores_per_k.append(df_fs.iloc[0]["scores"])
            best_combos_per_k.append(df_fs.iloc[0]["combo"])
        except:
            print("No best combo for k", k)
            best_scores_per_k.append([])
            best_combos_per_k.append([])
        print(k, np.mean(best_scores_per_k[-1]), np.std(best_scores_per_k[-1]))

    # plot best combos
    fig, ax = plt.subplots(figsize=(10, 3))
    mu = np.array([np.mean(v) for v in best_scores_per_k])
    std = np.array([np.std(v) for v in best_scores_per_k])
    print(std)
    ax.plot(k_s, mu)
    ax.fill_between(k_s, mu - std, mu + std, alpha=0.2)
    for k, combo, score in zip(k_s, best_combos_per_k, mu):
        print("Chosen feature combinations for", k, score, str(combo))  # , rotation=90)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("AUC ROC")
    # ax.set_ylim([0.6, 0.8])
    ax.axhline(max(mu), color="black", linestyle="--")
    plt.savefig(f"plots/eye_tracking_analysis/feature_selection{n_classes}_classes.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()


    ks_for_lcs = range(1, max_feature_set_size + 1)

    lcs = {}  # learning classifier system for each k
    for k in ks_for_lcs:
        combo = best_combos_per_k[k-1]
        lc_file = f"results/eye_tracking_{n_classes}_classes/lcs/lcs_{k}.csv"
        print(f"Get curves for {k} features with combo {combo}.")
        lcs[k] = get_learning_curves(
            learner=get_pipeline_for_features(learner, X, y, combo),
            X=X[combo],
            y=y,
            repeats=500,
            n_classes=n_classes,
            first_anchor=0.05,
            last_anchor=0.9,
            steps=10,
            filename=lc_file
        )
    # plot learning curves
    fig, ax = plt.subplots(figsize=(16, 6))
    # ax.plot(schedule, lc[0].mean(axis=1), label="train AUC")
    for k in [7, 8, 9, 10, 11, 12]:  # , 4, 8, 16]:  # TODO adjust here
        schedule, lc = [float(v) for v in lcs[k].columns], lcs[k].values
        mu = lc.mean(axis=0)
        std = lc.std(axis=0)
        ax.plot(schedule, mu, label=f"{k} features")
        ax.fill_between(schedule, mu - std, mu + std, alpha=0.3)
    ax.set_title(f"Learning Curves for Validation AUROC")
    ax.legend()
    ax.set_xlim([0, 1.6])
    ax.set_ylabel("AUC ROC")
    # ax.set_ylim([0.45,0.8])
    ax.axhline(0.725, color="blue", linestyle="--")
    ax.axhline(0.5, color="red", linestyle="--")
    plt.show()

