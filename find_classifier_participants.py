import platform
import matplotlib
import pandas as pd
import naiveautoml
import logging
from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_tw
from utils.splitting import leave_one_subject_out_cv
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    # do logging
    logger = logging.getLogger('naiveautoml')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    naml = naiveautoml.NaiveAutoML(max_hpo_iterations=20, show_progress=True, scoring="accuracy")

    n_classes = 2
    tw = 20  # time window in seconds
    label = "duration_estimate"  # "ppot" or "duration_estimate"
    include_meta_label = True

    X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True,
                                     include_meta_label=include_meta_label, tw=tw, label_name=[label])

    # Leave one subject out cross-validation
    splits = leave_one_subject_out_cv(X, y, "participant")

    # select learner
    for train_idx, test_idx in splits:
        # print(f"train: {train_idx}")
        # print(f"test: {test_idx}")
        print("\n")
        print("--------------------------------------------------")
        print(f"train on {y.iloc[train_idx]['participant'].unique()}")
        print(f"test on {y.iloc[test_idx]['participant'].unique()}")
        X_train = X.iloc[train_idx].drop(columns=["slice", "slice_fix", "participant", "time", "robot"])
        y_train = y.iloc[train_idx].drop(columns=["participant", "time", "robot"])

        y_train = y_train.to_numpy().ravel()

        naml.fit(X_train.values, y_train)

        print("---------------------------------")
        print(naml.chosen_model)
        print("---------------------------------")
        print(naml.history)

        naml.history.to_csv(f"results/eye_tracking_{n_classes}_classes/autoML_classifiers/naml_history_tw_{tw}_label_{label}_participant_{int(y.iloc[test_idx]['participant'].unique()[0])}.csv")