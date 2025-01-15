#!/bin/bash

# TODO: make sure the env is properly sourced
# source ~/PycharmProjects/watchplant_classification/venv/bin/activate
# source venv/bin/activate

for n_classes in 2 3; do
    for tw in 2 5 10 15 20 30 45 60; do
        for scoring in "accuracy" "roc_auc"; do
            for label in "duration_estimate" "ppot"; do
                for experiment_time in 0 1 2; do
                  python eye_tracking_time.py $n_classes $tw $scoring $label $robot
                done
            done
        done
    done
done
