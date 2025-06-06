import numpy as np
from utils.feature_loader import load_eye_tracking_data_tw
from sklearn.model_selection import train_test_split

# n_classes = 3
slicing = "tw"  # time window = "tw", old slicing after minutes = "minutes", no slicing = None
# tw = 1  # time window in seconds
label = "arousal"  # "ppot" or "duration_estimate", "arousal"
include_meta_label = True
bls = False  # baseline subtraction
only_pupil = True

for n_classes in [3]:
    for tw in [1, 2, 5, 10, 15, 20, 30, 45, 60]:
        if slicing == "minute":
            # X, y = load_eye_tracking_data_slice(number_of_classes=n_classes, load_preprocessed=False)
            pass
        elif slicing == "tw":
            X, y = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=False, include_meta_label=include_meta_label, tw=tw, label_name=[label], bls=bls, only_pupil=only_pupil)
            if not include_meta_label:
                X.drop(columns=["slice"], inplace=True)
        else:
            # X, y = load_eye_tracking_data(number_of_classes=n_classes, load_preprocessed=False)
            pass

        if not include_meta_label:
            X.drop(columns=["participant", "time", "robot"], inplace=True)  # drop setting information
        # print(f"features: {X.columns}")
        # print(f"labels: {y.columns}")
        # print(f"shape: {X.shape} {y.shape}")

        if not include_meta_label:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

            # print(f"train: x: {x_train.shape} y: {y_train.shape}")
            # print(f"test: x: {x_test.shape} y: {y_test.shape}")
            # print(f"min: {np.min(x_train)} {np.min(x_test)}")
            # print(f"max: {np.max(x_train)} {np.max(x_test)}")
            #
            # print(np.unique(y_train, return_counts=True))
            # print(np.unique(y_test, return_counts=True))

        if slicing == "minute":
            x_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_train_slice.csv", index=False)
            x_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_test_slice.csv", index=False)
            y_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_train_slice.csv", index=False)
            y_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_test_slice.csv", index=False)
        elif slicing == "tw":
            if bls:
                tag = "_bls"
            else:
                tag = ""
            if only_pupil:
                tag_pupil = "_only_pupil"
            else:
                tag_pupil = ""
            if include_meta_label:
                X.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_tw_{tw}_label_{label}_withMetaData{tag}{tag_pupil}.csv", index=False)
                y.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_tw_{tw}_label_{label}_withMetaData{tag}{tag_pupil}.csv", index=False)
                print(f"y distribution:\n{y['arousal'].value_counts()}")
            else:
                x_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_train_tw_{tw}_label_{label}{tag}{tag_pupil}.csv", index=False)
                x_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_test_tw_{tw}_label_{label}{tag}{tag_pupil}.csv", index=False)
                y_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_train_tw_{tw}_label_{label}{tag}{tag_pupil}.csv", index=False)
                y_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_test_tw_{tw}_label_{label}{tag}{tag_pupil}.csv", index=False)
        else:
            x_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_train.csv", index=False)
            x_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_test.csv", index=False)
            y_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_train.csv", index=False)
            y_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_test.csv", index=False)