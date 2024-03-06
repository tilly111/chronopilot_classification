import constants
import pandas as pd
import numpy as np

from sympy.combinatorics.subsets import ksubsets

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler


study = "2"  # "1" or "2"
block_names = ["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]
classifier_name = "XGB"  # "SVC", "DTC", "KNN", "GNB", "LR", "LDA", "RF", "GB", "AB", "XGB", "QDA"


####
all_acc = []

########################################################################################################################
# load data and labels
########################################################################################################################
# get background subtraction
ppg_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/baseline/ppg.csv")
eda_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/baseline/eda.csv")
tmp_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/baseline/tmp.csv")
eda_features_bg.drop(columns=["subject"], inplace=True)
tmp_features_bg.drop(columns=["subject"], inplace=True)

x_bg = pd.concat([ppg_features_bg, eda_features_bg, tmp_features_bg], axis=1).reset_index().drop(columns=["index"])
x_bg.fillna(0, inplace=True)  # TODO hack to resolve nans

# load all features and labels
x_all = []
y_all = []

for block in block_names:
    labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")

    ppg_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/ppg.csv")
    eda_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/eda.csv")
    tmp_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/tmp.csv")
    eda_features.drop(columns=["subject"], inplace=True)
    tmp_features.drop(columns=["subject"], inplace=True)

    x = pd.concat([ppg_features, eda_features, tmp_features], axis=1).reset_index().drop(columns=["index"])
    x.fillna(0, inplace=True)  # TODO hack to resolve nans

    # add background subtraction
    x.loc[:, x.columns != 'subject'] = x.loc[:, x.columns != 'subject'] - x_bg.loc[:, x_bg.columns != 'subject']

    x_all.append(x)
    y_all.append(labels)


########################################################################################################################
# set up training and test set
########################################################################################################################
for i, block in enumerate(block_names):
    # print(f"block: {block}")
    x_train = pd.concat([x_tmp for j, x_tmp in enumerate(x_all) if j != i], axis=0).drop(columns=["subject"]).to_numpy()
    y_train = pd.concat([y_tmp for j, y_tmp in enumerate(y_all) if j != i], axis=0).drop(columns=["subject"]).to_numpy()
    x_test = x_all[i].drop(columns=["subject"]).to_numpy()
    y_test = y_all[i]["block_estimation"].drop(columns=["subject"]).to_numpy()

    # print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    # print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

    # apply scaling
    min_max = MinMaxScaler()
    x_train = min_max.fit_transform(x_train)
    x_test = min_max.transform(x_test)

    if classifier_name == "SVC":
        estimator = SVC(decision_function_shape='ovr', kernel="linear", class_weight='balanced', probability=True)  # , class_weight='balanced'
    elif classifier_name == "DTC":
        estimator = DecisionTreeClassifier(criterion="entropy", splitter="best", class_weight='balanced')
    elif classifier_name == "KNN":
        estimator = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', metric='cosine', p=2)
    elif classifier_name == "GNB":
        estimator = GaussianNB()
    elif classifier_name == "LR":
        estimator = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
    elif classifier_name == "LDA":
        estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif classifier_name == "RF":
        estimator = RandomForestClassifier(criterion="gini", n_estimators=50, bootstrap=True, n_jobs=-1)  # , class_weight='balanced'  , n_estimators=2000
    elif classifier_name == "GB":
        estimator = GradientBoostingClassifier(loss='exponential', n_estimators=50, learning_rate=0.2)  # 2000
    elif classifier_name == "AB":
        estimator = AdaBoostClassifier(n_estimators=50)  # n_estimators=3000, learning_rate=0.2
    elif classifier_name == "XGB":
        estimator = XGBClassifier(n_estimators=50, booster='gbtree')  # 1000
    elif classifier_name == "QDA":
        estimator = QuadraticDiscriminantAnalysis()
        estimator = estimator.fit(x_train, y_train)

    selector = estimator.fit(x_train, y_train)

    y_pred = selector.predict(x_test)

    # TODO integrate shap
    # if use_shap:
    #     number_of_features = len(all_features)
    #     x_t = pd.DataFrame(data=x_test, columns=all_features)
    #     x_train_wn = pd.DataFrame(data=x_train, columns=all_features)
    #     if classifier_name in ["DTC", "KNN", "GNB", "QDA", "RF", "AB"]:
    #         explainer = shap.KernelExplainer(selector.predict_proba, x_train_wn)
    #         shap_value = explainer(x_t)
    #         # returns probability for class 0 and 1, but we only need one bc p = 1 - p
    #         shap_value.values = shap_value.values[:, :, 1]
    #         shap_value.base_values = shap_value.base_values[:, 1]
    #     else:
    #         explainer = shap.Explainer(selector, x_train_wn)
    #         shap_value = explainer(x_t)
    #
    #     print_frame = pd.DataFrame(data=np.zeros((1, for_printing_number_of_features)),
    #                                columns=for_printing_all_features)
    #     print_frame[shap_value.feature_names] = shap_value.abs.mean(axis=0).values
    #     for z in print_frame.columns:
    #         print(f"{print_frame[z].to_numpy().squeeze()}")
    #
    #     # plt.figure()
    #     # shap.plots.bar(shap_value, max_display=number_of_features, show=True)
    #     # print(print_frame)
    #     shap_values = print_frame  # pd.concat([shap_values, print_frame], axis='index', ignore_index=True)
    #     # print(shap_values)
    # print(f"accuracy: {selector.score(x_test, y_test)}")
    print(f"{selector.score(x_test, y_test)}")
    # all_acc.append(selector.score(x_test, y_test))
    # print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")

    # print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred, labels=[0, 1])}")
# print(f"mean accuracy: {np.mean(all_acc)}")