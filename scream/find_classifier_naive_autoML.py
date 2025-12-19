import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import argparse
import platform
import csv
from pathlib import Path
import matplotlib
import naiveautoml
import logging
from utils.feature_loader import load_scream_data
import numpy as np
import matplotlib.pyplot as plt
from utils.learner_pipeline import pipeline_to_full_str


def calc_best_classifier(participant_id, scoring, study, blocks, baseline_subtraction, base_dir, data_dir):
    # if os.path.exists(
    #     f"{base_dir}/classifiers_naive_autoML/study={study}/blocks={blocks}/{scoring}_baseline_subtraction={baseline_subtraction}_label=duration_estimate_{participant_id}.csv"):
    #     print(
    #         f"Skipping existing configuration: study={study}, blocks={blocks}, scoring={scoring}, baseline subtraction={baseline_subtraction}, participant_id={participant_id}")
    #     return
    
    print(f"current configuration: study={study}, blocks={blocks}, scoring={scoring}, baseline subtraction={baseline_subtraction}, participant_id={participant_id}")
    
    X, y = load_scream_data(study=study, baseline_subtraction=baseline_subtraction)
    
    # filter blocks
    X = X[(X["Block"].isin(blocks))]
    y = y[(y["Block"].isin(blocks))]
    # NOTE remove nan values
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]
    
    # make the loso split
    x_split = X[(X["Participant"] != participant_id)].copy()
    y_split = y[(y["Participant"] != participant_id)].copy()
    x_test = X[(X["Participant"] == participant_id)].copy()
    y_test = y[(y["Participant"] == participant_id)].copy()
    
    # drop meta data
    x_split.drop(columns=["Participant", "Block", "Slice"], inplace=True)
    
    # NOTE REMOVE NAN VALUES
    x_analysis = x_split.reset_index(drop=True)
    y_analysis = y_split.reset_index(drop=True)
    mask = ~x_analysis.isna().any(axis=1)
    x_analysis = x_analysis[mask]
    y_analysis = y_analysis[mask]
    y_analysis = y_analysis["duration_estimate"].to_numpy().ravel()
    
    labels, counts = np.unique(y_analysis, return_counts=True)
    
    # , kwargs_as={'excluded_components': {"learner": ["HistGradientBoostingClassifier"]}}
    naml = naiveautoml.NaiveAutoML(max_hpo_iterations=1024,
                                   max_hpo_iterations_without_imp=100,
                                   max_hpo_time_without_imp=100 * 65,  # allow 65 seconds per iteration
                                   timeout_candidate=300,
                                   scoring=scoring,
                                   evaluation_fun="mccv",
                                   kwargs_evaluation_fun={"n_splits": 20},
                                   # kwargs_as={'excluded_components': {"learner": ["HistGradientBoostingClassifier"]}},
                                   num_cpus=1,
                                   show_progress=True,
                                   random_state=42)
    
    naml.fit(x_analysis, y_analysis)
    
    # print("---------------------------------")
    # print(naml.chosen_model)
    # print("---------------------------------")
    # print(naml.history)
    
    dir_path = Path(f"{base_dir}/classifiers_naive_autoML/study={study}/blocks={blocks}")
    dir_path.mkdir(parents=True, exist_ok=True)
    hist = naml.history
    for i, row in hist.iterrows():
        hist.loc[i, 'pipeline'] = pipeline_to_full_str(row["pipeline"])
    hist.to_csv(
        f"{base_dir}/classifiers_naive_autoML/study={study}/blocks={blocks}/{scoring}_baseline_subtraction={baseline_subtraction}_label=duration_estimate_{participant_id}.csv",
        quoting=csv.QUOTE_ALL)
    print(f"Saved configuration: study={study}, blocks={blocks}, scoring={scoring}, baseline subtraction={baseline_subtraction}, participant_id={participant_id}")


if __name__ == "__main__":
    # for interactive plots
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
        plt.rcParams.update({'font.size': 20})
    elif platform.system() == "Linux":
        def is_headless():
            return os.environ.get("DISPLAY", "") == ""
        
        if not is_headless():
            matplotlib.use('QtAgg')
    elif platform.system() == "Windows":
        print("Windows is not supported!")
        exit(12)
        
    # params to choose
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--label', type=str, default="duration_estimate")
        parser.add_argument('--scoring', type=str, default="accuracy")
        parser.add_argument('--participant_id', type=int, default=2)  # participant id to leave out
        parser.add_argument('--study', type=int, default=1)
        parser.add_argument('--blocks', nargs="+", type=str, default=["T", "MA"])
        parser.add_argument('--baseline_subtraction', action='store_true', help="Enable baseline subtraction")
        parser.add_argument('--no_baseline_subtraction', dest='baseline_subtraction', action='store_false',
                            help="Disable baseline subtraction")
        parser.set_defaults(baseline_subtraction=True)
        parser.add_argument('--data_dir', type=str, default="/Volumes/Data/chronopilot/2024_scream")
        parser.add_argument('--base_dir', type=str, default="/Volumes/Data/chronopilot/2024_scream/results")
        
        args = parser.parse_args()
    else:
        print("Please provide arguments to run ...")
        exit(12)
    args.blocks.sort()
    
    # do logging
    logger = logging.getLogger('naiveautoml')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # participant_id, scoring, study, blocks, baseline_subtraction, base_dir, data_dir
    calc_best_classifier(participant_id=args.participant_id,
                         scoring=args.scoring,
                         study=args.study,
                         blocks=args.blocks,
                         baseline_subtraction=args.baseline_subtraction,
                         base_dir=args.base_dir,
                         data_dir=args.data_dir
                         )
