#!/bin/bash

# TODO: make sure the env is properly sourced
# source ~/PycharmProjects/watchplant_classification/venv/bin/activate
# source venv/bin/activate

for n_classes in 2 3; do
    for tw in 2 5 10 15 20 30 45 60; do
        for scoring in "accuracy" "roc_auc"; do
            for label in "duration_estimate" "ppot"; do
                for participant in 2 4 6 10 13 14 15 16 18 21 22 23 5 11 12 17 3; do
                  python eye_tracking_time.py $n_classes $tw $scoring $label $participant
                done
            done
        done
    done
done
