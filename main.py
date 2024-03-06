import constants
import platform
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

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

import shap

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')


study = "2"  # "1" or "2"
block_names = ["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]
classifier_name = "SVC"  # "SVC", "DTC", "KNN", "GNB", "LR", "LDA", "RF", "GB", "AB", "XGB", "QDA"
use_shap = False
feature_selection = "None"  # "RFECV", "SFS", None
neuro_kit = True


####
all_acc = []

########################################################################################################################
# load data and labels
########################################################################################################################
# get background subtraction
if neuro_kit:
    ppg_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/baseline/ppg_nk.csv")
else:
    ppg_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/baseline/ppg.csv")
eda_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/baseline/eda.csv")
tmp_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/baseline/tmp.csv")
eda_features_bg.drop(columns=["subject"], inplace=True)
tmp_features_bg.drop(columns=["subject"], inplace=True)

x_bg = pd.concat([ppg_features_bg, eda_features_bg, tmp_features_bg], axis=1).reset_index().drop(columns=["index"])
# x_bg = eda_features_bg.reset_index().drop(columns=["index"])
x_bg.fillna(0, inplace=True)  # TODO hack to resolve nans

# load all features and labels
x_all = None
y_all = None

for block in block_names:
    labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")

    if neuro_kit:
        ppg_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/ppg_nk.csv")
    else:
        ppg_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/ppg.csv")
    eda_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/eda.csv")
    tmp_features = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block}/tmp.csv")
    eda_features.drop(columns=["subject"], inplace=True)
    tmp_features.drop(columns=["subject"], inplace=True)

    x = pd.concat([ppg_features, eda_features, tmp_features], axis=1).reset_index().drop(columns=["index"])
    # x = eda_features.reset_index().drop(columns=["index"])
    x.fillna(0, inplace=True)  # TODO hack to resolve nans

    # add background subtraction
    x.loc[:, x.columns != 'subject'] = x.loc[:, x.columns != 'subject'] - x_bg.loc[:, x_bg.columns != 'subject']

    if x_all is None:
        x_all = x
        y_all = labels
    else:
        x_all = pd.concat([x_all, x], axis=0)
        y_all = pd.concat([y_all, labels], axis=0)

if use_shap or feature_selection is not None:
    if neuro_kit:
        all_features = constants.ALL_PPG_FEATURES_NEUROKIT + constants.ALL_EDA_FEATURES + constants.ALL_TMP_FEATURES
    else:
        all_features = constants.ALL_PPG_FEATURES_HEARTPY + constants.ALL_EDA_FEATURES + constants.ALL_TMP_FEATURES
    # all_features = constants.ALL_EDA_FEATURES

########################################################################################################################
# set up training and test set
########################################################################################################################
for p_t in ksubsets(constants.SUBJECTS_STUDY_2, 1):  # 3
    p_test = list(p_t)
    p_train = list(set(constants.SUBJECTS_STUDY_2) - set(p_test))
    # print(f"p_test: {p_test}, p_train: {p_train}")
    # print(f"p_test: {p_test}")
    x_train = x_all.loc[x_all["subject"].isin(p_train)].drop(columns=["subject"]).to_numpy()
    y_train = y_all.loc[y_all["subject"].isin(p_train)]["block_estimation"].drop(columns=["subject"]).to_numpy()
    x_test = x_all.loc[x_all["subject"].isin(p_test)].drop(columns=["subject"]).to_numpy()
    y_test = y_all.loc[y_all["subject"].isin(p_test)]["block_estimation"].drop(columns=["subject"]).to_numpy()

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

    if feature_selection == "RFECV":
        selector = RFECV(estimator, step=1, cv=len(p_train),
                         min_features_to_select=1)  # TODO seems to decrease the performance
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        # result_df = pd.DataFrame(zip(all_features, selector.support_, selector.ranking_))
        # result_df.columns = ["features", "used", "ranking"]
        # remove unused features for shap viz
        # all_features = list(compress(all_features, selector.support_))
        selector = estimator.fit(x_train, y_train)
    elif feature_selection == "SFS":
        selector = SequentialFeatureSelector(estimator, n_features_to_select='auto', direction="forward",
                                             tol=0.0001)  # 'auto' , n_jobs=-1
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        # result_df = pd.DataFrame(zip(all_features, selector.get_support()))
        # result_df.columns = ["features", "used"]
        # remove unused features for shap viz
        # all_features = list(compress(all_features, selector.get_support()))
        selector = estimator.fit(x_train, y_train)
    else:
        selector = estimator.fit(x_train, y_train)

    y_pred = selector.predict(x_test)

    # TODO integrate shap
    if use_shap:
        number_of_features = len(all_features)
        x_t = pd.DataFrame(data=x_test, columns=all_features)
        x_train_wn = pd.DataFrame(data=x_train, columns=all_features)
        if classifier_name in ["DTC", "KNN", "GNB", "QDA", "RF", "AB"]:
            explainer = shap.KernelExplainer(selector.predict_proba, x_train_wn)
            shap_value = explainer(x_t)
            # returns probability for class 0 and 1, but we only need one bc p = 1 - p
            shap_value.values = shap_value.values[:, :, 1]
            shap_value.base_values = shap_value.base_values[:, 1]
        else:
            explainer = shap.Explainer(selector, x_train_wn)
            shap_value = explainer(x_t)

        print_frame = pd.DataFrame(data=np.zeros((1, number_of_features)),
                                   columns=all_features)
        print_frame[shap_value.feature_names] = shap_value.abs.mean(axis=0).values
        for z in print_frame.columns:
            print(f"{print_frame[z].to_numpy().squeeze()}")

        plt.figure()
        shap.plots.bar(shap_value, max_display=number_of_features, show=True)
        plt.show()
        # print(print_frame)
        # shap_values = print_frame  # pd.concat([shap_values, print_frame], axis='index', ignore_index=True)
        # print(shap_values)
    # print(f"accuracy: {selector.score(x_test, y_test)}")
    print(f"{selector.score(x_test, y_test)}")
    # all_acc.append(selector.score(x_test, y_test))
    # print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
    #
    # print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred, labels=[0, 1])}")
