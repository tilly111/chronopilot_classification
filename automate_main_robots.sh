#!/bin/bash

# TODO: make sure the env is properly sourced
# source ~/PycharmProjects/watchplant_classification/venv/bin/activate
# source venv/bin/activate

for n_classes in 2 3; do
    for tw in 2 5 10 15 20 30 45 60; do
        for scoring in "accuracy" "roc_auc"; do
            for label in "duration_estimate" "ppot"; do
                for robot in 0 1 2 3 4 5 6 7; do
                  python eye_tracking_robots.py $n_classes $tw $scoring $label $robot
                done
            done
        done
    done
done
