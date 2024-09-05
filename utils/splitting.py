import pandas as pd


def leave_one_subject_out_cv(X:pd.DataFrame, y:pd.DataFrame, criterium: str):
    """Leave one subject out cross-validation
        This method allows to split the dataset in a leave one subject out fashion depending on the criterium;
        The criterium can be "robot" or "participant" and it will split the dataset accordingly.
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Labels
    criterium : String
        The criterium to split the dataset, can be "robot" or "participant" or "time"

    Returns
    -------
    splits : list
        List of tuples with train and test indexes
    """
    # get all subgroups of the dataset
    subgroups = y[criterium].unique()
    print(f"individual {criterium}s: {subgroups}")

    # create the splits
    splits = []
    for subgroup in subgroups:
        train_idx = y[y[criterium] != subgroup].index
        test_idx = y[y[criterium] == subgroup].index
        splits.append((train_idx, test_idx))

    return splits


def only_one_minute(X:pd.DataFrame, y:pd.DataFrame, criterium: str, minute=1, tw=20):
    lower_bound = int((60 / tw) * (minute - 1))
    upper_bound = int((60 / tw) * minute)

    print(f"lower bound: {lower_bound}, upper bound: {upper_bound}")

    subgroups = X[criterium].unique()
    print(f"individual {criterium}s: {subgroups}")
    idx = X[(X['slice'] >= lower_bound) & (X['slice'] < upper_bound)].index

    return idx
