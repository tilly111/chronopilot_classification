from sklearn.ensemble import ExtraTreesClassifier
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

labels = pd.read_csv("/Volumes/Data/chronopilot/helicopter/new_label/labels_3_classes.csv")
y = labels["PassageOfTimeSlowFast"].to_numpy()
X = None


for i in range(12):
    data = pd.read_csv(f"/Volumes/Data/chronopilot/helicopter/features/helicopter_experiment/ppg_nk/start_posttest/subjectID_{i+1}.csv")

    data.drop(columns=["Unnamed: 0"], inplace=True)

    X = data if X is None else pd.concat([X, data], axis=0)

X.fillna(0, inplace=True)
print(X.shape, y.shape)

clf = ExtraTreesClassifier(n_estimators=1024)

fitted = clf.fit(X, y)

y_pred = fitted.predict(X)

print(accuracy_score(y, y_pred))


joblib.dump(fitted, "/helicopter_3_classes.pkl")