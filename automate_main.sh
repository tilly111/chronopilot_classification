#!/bin/bash

source ~/PycharmProjects/watchplant_classification/venv/bin/activate

for n_classes in 2 3; do
    for tw in 2 5 10 15 20 30 45 60; do
        for scoring in "accuracy" "roc_auc"; do
            for label in "duration_estimate" "ppot"; do
                python main.py $n_classes $tw $scoring $label
            done
        done
    done
done
