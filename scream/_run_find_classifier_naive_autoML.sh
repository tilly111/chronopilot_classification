#!/bin/bash

source ../venv/bin/activate

participants_study_1=(2 4 7 8 11 16 19 22 24 25 28 29 30 31 32 33 35 36 37 39 40 41 42 43 44 47)
participants_study_2=(1 2 3 4 5 6 7 9 11 12 13 16 18 23 25 26 27 29 31 32 33 41 42 45 46 47)

combinations=(
  "exp_T exp_MA"
  "exp_T exp_TU"
  "exp_T exp_PU"
  "exp_T exp_S"
  "exp_MA exp_TU"
  "exp_MA exp_PU"
  "exp_MA exp_S"
  "exp_TU exp_PU"
  "exp_TU exp_S"
  "exp_PU exp_S"

  "exp_T exp_MA exp_TU"
  "exp_T exp_MA exp_PU"
  "exp_T exp_MA exp_S"
  "exp_T exp_TU exp_PU"
  "exp_T exp_TU exp_S"
  "exp_T exp_PU exp_S"
  "exp_MA exp_TU exp_PU"
  "exp_MA exp_TU exp_S"
  "exp_MA exp_PU exp_S"
  "exp_TU exp_PU exp_S"

  "exp_T exp_MA exp_TU exp_PU"
  "exp_T exp_MA exp_TU exp_S"
  "exp_T exp_MA exp_PU exp_S"
  "exp_T exp_TU exp_PU exp_S"
  "exp_MA exp_TU exp_PU exp_S"

  "exp_T exp_MA exp_TU exp_PU exp_S"
)

for participant_id in "${participants_study_1[@]}"; do
  for combo in "${combinations[@]}"; do
    combo_array=($combo)
    python find_classifier_naive_autoML.py --participant_id "$participant_id" --study 1 --blocks ${combo_array[@]} --no_baseline_subtraction
  done
done

for participant_id in "${participants_study_2[@]}"; do
  for combo in "${combinations[@]}"; do
    combo_array=($combo)
    python find_classifier_naive_autoML.py --participant_id "$participant_id" --study 2 --blocks ${combo_array[@]}
  done
done