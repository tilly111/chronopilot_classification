import os

from sklearn.ensemble import HistGradientBoostingClassifier

import platform
from tqdm import tqdm
import matplotlib
import pandas as pd
import naiveautoml
import logging
from sklearn.model_selection import train_test_split
from utils.splitting import train_test_split_tw, analysis_test_split_tw
from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_tw
import matplotlib.pyplot as plt
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


def plot_history(naml):
    scoring = naml.task.scoring["name"]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(naml.history["time"], naml.history[scoring])
    ax.axhline(naml.history[scoring].max(), linestyle="--", color="black", linewidth=1)
    max_val = naml.history[scoring].max()
    median_val = naml.history[scoring].median()
    ax.set_ylim([median_val, max_val + (max_val - median_val)])
    plt.show()


def calc_best_classifier(n_classes, label, tw, scoring):
    if n_classes == 3 and scoring == "roc_auc":
        return
    print(f"current configuration: {n_classes} classes, {tw} seconds, {label}, {scoring}")
    naml = naiveautoml.NaiveAutoML(max_hpo_iterations=1024, show_progress=True, scoring=scoring,
                                   max_hpo_iterations_without_imp=100, num_cpus=1,
                                   kwargs_as={'excluded_components': {"learner": ["HistGradientBoostingClassifier"]}})

    X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, include_meta_label=True,
                                     tw=tw, label_name=[label], bls=bls)
    # drop meta data
    X.drop(columns=["robot", "participant", "slice", "time"], inplace=True)
    # y.drop(columns=["robot", "participant", "time"], inplace=True)

    x_analysis, _, y_analysis, _ = analysis_test_split_tw(X, y)

    y_analysis = y_analysis[label].to_numpy().ravel()

    naml.fit(x_analysis, y_analysis)

    #print("---------------------------------")
    #print(naml.chosen_model)
    #print("---------------------------------")
    #print(naml.history)

    naml.history.to_csv(
        f"results/eye_tracking_{n_classes}_classes/autoML_classifiers/{scoring}_naml_history_tw_{tw}_label_{label}{tag}.csv")

if __name__ == "__main__":
    # do logging
    logger = logging.getLogger('naiveautoml')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    include_meta_label = True
    bls = True  # baseline subtraction
    tag = "_bls" if bls else ""
    # os.cpu_count()
    pbar = tqdm(total=56)
    futures = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for n_classes in [2, 3]:
            for label in ["duration_estimate", "ppot"]:
                for tw in [5, 10, 15, 20, 30, 45, 60]:
                    for scoring in ["accuracy", "roc_auc"]:
                        executor.submit(calc_best_classifier, n_classes, label, tw, scoring)
        # Attach the callback to each future
        def _cb(future):
            pbar.update(1)


    #for n_classes in [2, 3]:
    #    for label in ["duration_estimate", "ppot"]:
    #        for tw in [5, 10, 15, 20, 30, 45, 60]:
            #            for scoring in ["accuracy", "roc_auc"]:
            #        if n_classes == 3 and scoring == "roc_auc":
            #           continue
            #       print(f"current configuration: {n_classes} classes, {tw} seconds, {label}, {scoring}")
            #       naml = naiveautoml.NaiveAutoML(max_hpo_iterations=1024, show_progress=True, scoring=scoring, max_hpo_iterations_without_imp=100, num_cpus=1, kwargs_as={'excluded_components': {"learner": ["HistGradientBoostingClassifier"]}})
            #
            #       X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, include_meta_label=True, tw=tw, label_name=[label], bls=bls)
            #       # drop meta data
            #       X.drop(columns=["robot", "participant", "slice", "time"], inplace=True)
            #       # y.drop(columns=["robot", "participant", "time"], inplace=True)
            #
            #       x_analysis, _, y_analysis, _ = analysis_test_split_tw(X, y)
            #
            #       y_analysis = y_analysis[label].to_numpy().ravel()
            #
            #       naml.fit(x_analysis, y_analysis)
            #
            #       print("---------------------------------")
            #       print(naml.chosen_model)
            #       print("---------------------------------")
            #       print(naml.history)
            #
    #       naml.history.to_csv(f"results/eye_tracking_{n_classes}_classes/autoML_classifiers/{scoring}_naml_history_tw_{tw}_label_{label}{tag}.csv")
    #               # plot_history(naml)