#!/bin/bash

# TODO: make sure the env is properly sourced
# source ~/PycharmProjects/watchplant_classification/venv/bin/activate
# source venv/bin/activate

for n_classes in 2 3; do
    for tw in 2 5 10 15 20 30 45 60; do
        for scoring in "accuracy"; do
            for label in "duration_estimate" "ppot"; do
                for participant in 1 2 3 4 5 6 7 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
                  python eye_tracking_participants.py $n_classes $tw $scoring $label $participant
                done
            done
        done
    done
done
