#!/bin/bash

# TODO: make sure the env is properly sourced
# source ~/PycharmProjects/watchplant_classification/venv/bin/activate
# source venv/bin/activate

for n_classes in 3; do  # 2
    for tw in 1 2 5 10 15 20 30 45 60; do  #
        for scoring in "accuracy"; do  # "roc_auc"
            for label in "arousal"; do  # "duration_estimate" "ppot"
                python main.py $n_classes $tw $scoring $label
            done
        done
    done
done
