import platform
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from itertools import cycle

from utils.feature_loader import load_scream_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_tw
from sklearn.model_selection import train_test_split

from sklearn.metrics import RocCurveDisplay

from utils.learner_pipeline import get_pipeline_for_features
from plotting_scripts.roc_curve_plotting import get_mccv_ROC_display

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')
elif platform.system() == "Windows":
    # TODO
    pass


## load training/validation
X, y = load_eye_tracking_data_tw(number_of_classes=2, load_preprocessed=True)
print(y.shape)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# classifier = LogisticRegression()
classifier = ExtraTreesClassifier(n_estimators=1000)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
print(y_onehot_test.shape)  # (n_samples, n_classes)

print(label_binarizer.classes_)


class_id = 2  # fast
class_of_interest = "fast"

fig, ax = plt.subplots(figsize=(6, 6))
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, class_name in enumerate(["slow", "medium", "fast"]):
    if i == 0:
        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, i],
            y_score[:, i],
            name=f"{class_name} vs the rest",
            #color=colors[i],
            ax=ax,
            plot_chance_level=True,
        )
    else:
        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, i],
            y_score[:, i],
            name=f"{class_name} vs the rest",
            # color=colors[i],
            ax=ax,
            plot_chance_level=False,
        )


_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)",
)

plt.show()