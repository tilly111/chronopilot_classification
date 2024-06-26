import platform
import matplotlib
import pandas as pd
import naiveautoml
import logging
from sklearn.model_selection import train_test_split
from utils.feature_loader import load_eye_tracking_data
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

    n_classes = 3
    X, y = load_eye_tracking_data(number_of_classes=n_classes, load_preprocessed=True)

    # preselecting the best subset
    X = X[['sub_max_speed_fix', 'sub_mean_dispersion_fix', 'sub_mean_duration_fix', 'sub_mean_speed',
           'sub_min_dispersion_fix', 'sub_min_speed_fix', 'sub_number_clusters_fix']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y = y.to_numpy().ravel()

    naml.fit(X, y)

    print("---------------------------------")
    print(naml.chosen_model)
    print("---------------------------------")
    print(naml.history)

    naml.history.to_csv("results/autoML_classifiers/naml_history_eye_tracking_3_classes.csv")
    plot_history(naml)