#!/bin/bash

# TODO: make sure the env is properly sourced
# source ~/PycharmProjects/watchplant_classification/venv/bin/activate
# source venv/bin/activate

for n_classes in 2 3; do
    for tw in 1; do  # 2 5 10 15 20 30 45 60
        for scoring in "accuracy"; do  # "roc_auc"
            for label in "duration_estimate" "ppot"; do
                python main.py $n_classes $tw $scoring $label
            done
        done
    done
done
