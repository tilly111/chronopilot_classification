import numpy as np
import constants
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from sympy.utilities.iterables import multiset_permutations
from sympy.combinatorics.subsets import ksubsets


from sklearn.preprocessing import MinMaxScaler


# for interactive plots
# matplotlib.use('QtAgg')  # TODO made for MacOs prbably needs to be adjusted -- only for plots
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)



########################################################################################################################
# load data
########################################################################################################################
X = pd.read_csv("preprocessed_data/helicopter/X.csv")
y = pd.read_csv("preprocessed_data/helicopter/y.csv")

p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

acc_list = []

for p_t in ksubsets(p, 1):
    p_test = list(p_t)
    p_train = [ele for ele in p if ele not in p_test]

    X_train = X.loc[X["ParticipantID"].isin(p_train)].drop(columns=["ParticipantID", "Session"])
    X_test = X.loc[X["ParticipantID"].isin(p_test)].drop(columns=["ParticipantID", "Session"])
    y_train = y.loc[y["ParticipantID"].isin(p_train)].drop(columns=["ParticipantID", "Session"]).values.flatten()
    y_test = y.loc[y["ParticipantID"].isin(p_test)].drop(columns=["ParticipantID", "Session"]).values.flatten()

    print(f"shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")

    # estimator = SVC(decision_function_shape='ovr', kernel="linear", class_weight='balanced')
    estimator = RandomForestClassifier(n_estimators=2000, bootstrap=True)
    selector = estimator.fit(X_train, y_train)
    y_pred = selector.predict(X_test)

    acc_list.append(selector.score(X_test, y_test))

    print(f"accuracy: {selector.score(X_test, y_test)}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")

    print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred)}")

    if p_test[0] == 1:  # save model
        joblib.dump(selector, "results/helicopter/helicopter_model.joblib")


print(f"mean acc: {np.mean(acc_list)}")  # List of accuracies for each participant test set

# How to load the model
import joblib
loaded_model = joblib.load("path/to/model/helicopter_model.joblib")
# predict using the loaded model
prediction = loaded_model.predict(X_test)


print(f"loaded model accuracy: {loaded_model.score(X_test, y_test)}")  # should be the same as the last accuracy


    # if classifier_name == "SVC":
    #
    # elif classifier_name == "DTC":
    #     estimator = DecisionTreeClassifier(criterion="entropy", splitter="best", class_weight='balanced')
    # elif classifier_name == "KNN":
    #     estimator = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', metric='cosine', p=2)
    # elif classifier_name == "GNB":
    #     estimator = GaussianNB()
    # elif classifier_name == "LR":
    #     estimator = LogisticRegression(penalty=None, max_iter=100, solver='lbfgs')
    # elif classifier_name == "LDA":
    #     estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    # elif classifier_name == "RF":
    #     estimator = RandomForestClassifier(criterion="gini", n_estimators=2000, bootstrap=True)  # , class_weight='balanced'  , n_jobs=-1
    # elif classifier_name == "GB":
    #     estimator = GradientBoostingClassifier(loss='exponential', n_estimators=1000, learning_rate=0.2)
    # elif classifier_name == "AB":
    #     estimator = AdaBoostClassifier(n_estimators=3000)  # n_estimators=1000, learning_rate=0.2
    # elif classifier_name == "XGB":
    #     estimator = XGBClassifier(n_estimators=1000, booster='gbtree')
    # elif classifier_name == "QDA":
    #     estimator = QuadraticDiscriminantAnalysis()
    #
    # if feature_selection == "RFECV":
    #     selector = RFECV(estimator, step=1, cv=5, min_features_to_select=1)  # TODO seems to decrease the performance
    #     selector = selector.fit(x_train, y_train)
    # elif feature_selection == "SFS":
    #     selector = SequentialFeatureSelector(estimator, n_features_to_select='auto', direction="forward")  # 'auto' , n_jobs=-1
    #     x_train = selector.fit_transform(x_train, y_train)
    #     # selector.support_ = np.array([True, False, True, True, False, False, True, True, False, False, False, False, True, True,
    #     #                      True, True, True, True, True, False, False, False, False, False])
    #     # x_train = selector.transform(x_train)
    #     x_test = selector.transform(x_test)
    #     result_df = pd.DataFrame(zip(all_features, selector.get_support()))
    #     result_df.columns = ["features", "used"]
    #     print(result_df)
    #     if save_flag:
    #         result_df.to_csv("results/" + experiment_name + "/" + classifier_name + "_" + feature_selection + "_" + str(p_test) + "_used_features.csv")
    #     selector = estimator.fit(x_train, y_train)
    # else:
    #     selector = estimator.fit(x_train, y_train)
    #
    #
    # if feature_selection == "RFECV":
    #     result_df = pd.DataFrame(zip(all_features, selector.support_, selector.ranking_))
    #     result_df.columns = ["features", "used", "ranking"]
    #     print(result_df)
    #     if save_flag:
    #         result_df.to_csv("results/" + experiment_name + "/" + classifier_name + "_" + feature_selection + "_" + str(
    #             p_test) + "_used_features.csv")
    #
    # y_pred = selector.predict(x_test)
    #
    # # print("y_pred:", y_pred)
    # # print("y_test:", y_test)
    #
    # print(f"-----------------{p_test}-----------------")
    #
    # print(f"accuracy: {selector.score(x_test, y_test)}")
    # print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
    #
    # print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred)}")
    #
    # if feature_selection is None:
    #     cm = confusion_matrix(y_test, y_pred)
    #     df_metrics = pd.DataFrame(data=np.zeros((2, 4), dtype=float), index=[0, 1],
    #                               columns=['Accuracy', 'F1 Score', 'Confusion Matrix', 'Confusion Matrix cont'])
    #     df_metrics["Accuracy"][1] = np.nan
    #     df_metrics["F1 Score"][1] = np.nan
    #     df_metrics["Accuracy"][0] = selector.score(x_test, y_test)
    #     df_metrics["F1 Score"][0] = f1_score(y_test, y_pred, average='weighted')
    #     df_metrics["Confusion Matrix"] = cm[:, 0]
    #     df_metrics["Confusion Matrix cont"] = cm[:, 1]
    #     if save_flag:
    #         df_metrics.to_csv(
    #             "results/" + experiment_name + "/" + classifier_name + "_None_" + str(p_test) + "_scores.csv")
    # else:
    #     cm = confusion_matrix(y_test, y_pred)
    #     df_metrics = pd.DataFrame(data=np.zeros((2, 4), dtype=float), index=[0, 1],
    #                               columns=['Accuracy', 'F1 Score', 'Confusion Matrix', 'Confusion Matrix cont'])
    #     df_metrics["Accuracy"][1] = np.nan
    #     df_metrics["F1 Score"][1] = np.nan
    #     df_metrics["Accuracy"][0] = selector.score(x_test, y_test)
    #     df_metrics["F1 Score"][0] = f1_score(y_test, y_pred, average='weighted')
    #     df_metrics["Confusion Matrix"] = cm[:, 0]
    #     df_metrics["Confusion Matrix cont"] = cm[:, 1]
    #     if save_flag:
    #         df_metrics.to_csv(
    #             "results/" + experiment_name + "/" + classifier_name + "_" + feature_selection + "_" + str(
    #                 p_test) + "_scores.csv")