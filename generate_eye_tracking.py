import numpy as np
from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_slice
from sklearn.model_selection import train_test_split

n_classes = 3
slicing = True

if slicing:
    X, y = load_eye_tracking_data_slice(number_of_classes=n_classes, load_preprocessed=False)
else:
    X, y = load_eye_tracking_data(number_of_classes=n_classes, load_preprocessed=False)


X.drop(columns=["participant", "time", "robot"], inplace=True)  # drop setting information

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

print(f"train: x: {x_train.shape} y: {y_train.shape}")
print(f"test: x: {x_test.shape} y: {y_test.shape}")

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

if slicing:
    x_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_train_slice.csv", index=False)
    x_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_test_slice.csv", index=False)
    y_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_train_slice.csv", index=False)
    y_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_test_slice.csv", index=False)
else:
    x_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_train.csv", index=False)
    x_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/X_test.csv", index=False)
    y_train.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_train.csv", index=False)
    y_test.to_csv(f"preprocessed_data/eye_tracking_{n_classes}_classes/y_test.csv", index=False)