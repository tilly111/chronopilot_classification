import constants
import platform
import pandas as pd
import numpy as np
import torch as t
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from sympy.combinatorics.subsets import ksubsets

from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from imblearn.over_sampling import BorderlineSMOTE

from sklearn.preprocessing import MinMaxScaler

# for dim reduction?

from classifier.scream_1 import train_nn_model, predict_NN


import shap

from plotting_scripts.plot_physio import plot_physio3D, plot_physio2D

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')


study = "1"  # "1" or "2"
block_names = ["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]  # "exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"
background_block_name = "exp_S"  # "baseline"
classifier_name = "XGB"  # "SVC", "DTC", "KNN", "GNB", "LR", "LDA", "RF", "GB", "AB", "XGB", "QDA", "NN"
individual_tag = "v3"
use_shap = False
feature_selection = "RFECV"  # "RFECV", "SFS", None, "MANUAL"
neuro_kit = False
vizualize = False

# what features to use
manual_features = ["breathing rate", "bpm", "hr_mad"]
tmp_all_acc = []
tmp_all_cm = []

if study == "1":
    # population = constants.SUBJECTS_STUDY_1
    population = constants.SUBJECTS_STUDY_1_test
    # population = constants.SUBJECTS_STUDY_1_over
    # population = constants.SUBJECTS_STUDY_1_only3
    # population = [24, 30, 36]
else:
    population = constants.SUBJECTS_STUDY_2  # constants.SUBJECTS_STUDY_2, GROUP_5

########################################################################################################################
# load data and labels
########################################################################################################################
# get background subtraction
if neuro_kit:
    ppg_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/ppg_nk.csv")
else:
    ppg_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/ppg.csv")
eda_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/eda.csv")
tmp_features_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{background_block_name}/tmp.csv")
eda_features_bg.drop(columns=["subject"], inplace=True)
tmp_features_bg.drop(columns=["subject"], inplace=True)

x_bg = pd.concat([ppg_features_bg, eda_features_bg, tmp_features_bg], axis=1).reset_index().drop(columns=["index"])
# x_bg = eda_features_bg.reset_index().drop(columns=["index"])
x_bg.fillna(0, inplace=True)  # TODO hack to resolve nans

labels_bg = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{background_block_name}.csv")

# load all features and labels
x_all = None
y_all = None

for block in block_names:
    labels = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/labels/{block}.csv")
    labels["block_estimation"] = labels["block_estimation"] - labels_bg["block_estimation"] + 1
    # if block == "exp_MA":
    #     labels["block_estimation"] = 0
    # elif block == "exp_T":  # exp_TU exp_MA
    #     labels["block_estimation"] = 1
    # elif block == "exp_TU":
    #     labels["block_estimation"] = 0
    # else:
    #     labels["block_estimation"] = 3
    # print(block)

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
    # x.loc[:, x.columns != 'subject'] = (x.loc[:, x.columns != 'subject'] - x_bg.loc[:, x_bg.columns != 'subject']) / x_bg.loc[:, x_bg.columns != 'subject']

    if x_all is None:
        x_all = x
        y_all = labels
    else:
        x_all = pd.concat([x_all, x], axis=0)
        y_all = pd.concat([y_all, labels], axis=0)

# if use_shap or feature_selection is not None:
if neuro_kit:
    all_features_c = constants.ALL_PPG_FEATURES_NEUROKIT_AVAILABLE + constants.ALL_EDA_FEATURES + constants.ALL_TMP_FEATURES
else:
    all_features_c = constants.ALL_PPG_FEATURES_HEARTPY + constants.ALL_EDA_FEATURES + constants.ALL_TMP_FEATURES
    # all_features_c = constants.ALL_EDA_FEATURES

if feature_selection == "MANUAL":
    all_features_c = manual_features
    manual_features.append("subject")
    x_all = x_all.drop(columns=[col for col in x_all.columns if col not in manual_features])

    # apply some additional dimensionality reduction aka smart down scaling
    # dim_reducer = Isomap(n_components=3, n_neighbors=5)
    # dim_reducer = LocallyLinearEmbedding(n_components=2, n_neighbors=2)
    # x_numpy = dim_reducer.fit_transform(x_all.drop(columns=["subject"]))
    # x_df = pd.DataFrame(data=x_numpy, columns=["dim1", "dim2", "dim3"])
    # x_df = pd.DataFrame(data=x_numpy, columns=["dim1", "dim2"])
    # x_all_tmp = x_all["subject"].reset_index().drop(columns=["index"])
    # x_all_tmp["dim1"] = x_df["dim1"]
    # x_all_tmp["dim2"] = x_df["dim2"]
    # # x_all_tmp["dim3"] = x_df["dim3"]
    # x_all = x_all_tmp

########################################################################################################################
# set up training and test set
########################################################################################################################
for p_t in ksubsets(population, 1):  # 3
# settings = list(ksubsets(population, 6))
# for p_counter in range(10):  # 3
#     p_t = settings[np.random.randint(0, len(settings))]
    p_test = list(p_t)
    p_train = list(set(population) - set(p_test))
    # p_train = list(p_t)
    # print(f"p_test: {p_test}, p_train: {p_train}")
    # print(f"p_test: {p_test}")
    print(f"-------------test subject: {p_test[0]}-------------")
    x_train = x_all.loc[x_all["subject"].isin(p_train)].drop(columns=["subject"]).to_numpy()
    y_train = y_all.loc[y_all["subject"].isin(p_train)]["block_estimation"].drop(columns=["subject"]).to_numpy()
    x_test = x_all.loc[x_all["subject"].isin(p_test)].drop(columns=["subject"]).to_numpy()
    y_test = y_all.loc[y_all["subject"].isin(p_test)]["block_estimation"].drop(columns=["subject"]).to_numpy()

    # apply scaling
    min_max = MinMaxScaler()
    x_train = min_max.fit_transform(x_train)
    x_test = min_max.transform(x_test)
    # TODO this is the wrong way of scaling because you include validation information
    # x = min_max.fit_transform(np.concatenate((x_train, x_test), axis=0))
    # x_train = x[:x_train.shape[0], :]
    # x_test = x[x_train.shape[0]:, :]
    # smote it in
    # sm = BorderlineSMOTE(random_state=42, k_neighbors=3, m_neighbors=3)
    # x_train, y_train = sm.fit_resample(x_train, y_train)
    # print(f"labels: {np.unique(y_train, return_counts=True)}")

    if classifier_name == "SVC":
        estimator = SVC(kernel="rbf", class_weight='balanced', probability=True)  # , class_weight='balanced'
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
        estimator = RandomForestClassifier(criterion="gini", n_estimators=100, bootstrap=True, n_jobs=-1)  # , class_weight='balanced'  , n_estimators=2000
    elif classifier_name == "GB":
        estimator = GradientBoostingClassifier(loss='exponential', n_estimators=50, learning_rate=0.2)  # 2000
    elif classifier_name == "AB":
        estimator = AdaBoostClassifier(n_estimators=50)  # n_estimators=3000, learning_rate=0.2
    elif classifier_name == "XGB":
        estimator = XGBClassifier(n_estimators=50, booster='gbtree')  # 1000
    elif classifier_name == "QDA":
        estimator = QuadraticDiscriminantAnalysis()
        estimator = estimator.fit(x_train, y_train)
    elif classifier_name == "NN":
        # print(f"nb classes {np.unique(y_train).shape[0]}")
        pass
        # estimator = FeedForwardNN(x_train.shape[1], 10, np.unique(y_train).shape[0])

    if feature_selection == "RFECV":
        selector = RFECV(estimator, step=1, cv=len(p_train), min_features_to_select=1)  # cv=len(p_train)
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        result_df = pd.DataFrame(zip(all_features_c, selector.support_, selector.ranking_))
        result_df.columns = ["features", "used", "ranking"]
        # remove unused features for shap viz
        all_features_viz = list(compress(all_features_c, selector.support_))
        print(all_features_viz)
        selector = estimator.fit(x_train, y_train)
    elif feature_selection == "SFS":
        selector = SequentialFeatureSelector(estimator, n_features_to_select=2, direction="forward",
                                             tol=0.0001)  # 'auto' , n_jobs=-1
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        result_df = pd.DataFrame(zip(all_features_c, selector.get_support()))
        result_df.columns = ["features", "used"]
        # remove unused features for shap viz
        all_features_viz = list(compress(all_features_c, selector.get_support()))
        selector = estimator.fit(x_train, y_train)
    elif ((feature_selection == "MANUAL") or (feature_selection == "None")) and classifier_name != "NN":
        selector = estimator.fit(x_train, y_train)
        all_features_viz = all_features_c
        if vizualize:
            if x_train.shape[1] == 2:
                plot_physio2D(x_train, y_train, x_test, y_test, ["dim 1", "dim 2"], selector)
            elif x_train.shape[1] == 3:
                plot_physio3D(x_train, y_train, x_test, y_test, all_features_c, selector)
    elif classifier_name == "NN":
        # print(f"x_train original shape: {x_train.shape}")
        if len(block_names) == 2:
            name = f"study{study}_{individual_tag}_block_{block_names[0]}_{block_names[1]}_subject_{p_test[0]}"
        else:
            name = f"study{study}_{individual_tag}_block_all_{feature_selection}_subject_{p_test[0]}"
        # make train + validation
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        y_pred = train_nn_model(x_train, y_train, x_val, y_val, x_test, y_test, model_name=name)
        y_pred = y_pred[0].numpy()
        print(y_pred)
        # estimator.train_model(x_train, y_train, 100)
        # selector = estimator
        # all_features_viz = all_features_c
    else:
        print("feature selection is not implemented, exiting...")
        exit(3)

    if classifier_name != "NN":
        y_pred = selector.predict(x_test)
    else:
        # y_pred = predict_NN(x_test, y_test)
        # x_test_tensor = t.tensor(x_test, dtype=t.float32)
        # y_pred_ohe = selector(x_test_tensor)
        # _, y_pred = t.max(y_pred_ohe, 1)
        pass

    if use_shap:
        number_of_features = len(all_features_viz)
        x_t = pd.DataFrame(data=x_test, columns=all_features_viz)
        x_train_wn = pd.DataFrame(data=x_train, columns=all_features_viz)
        if classifier_name in ["SVC", "DTC", "KNN", "GNB", "QDA", "RF", "AB"]:
            explainer = shap.KernelExplainer(selector.predict_proba, x_train_wn)
            shap_value = explainer(x_t)
            # returns probability for class 0 and 1, but we only need one bc p = 1 - p
            shap_value.values = shap_value.values[:, :, 1]
            shap_value.base_values = shap_value.base_values[:, 1]
        else:
            explainer = shap.Explainer(selector, x_train_wn)
            shap_value = explainer(x_t)

        print_frame = pd.DataFrame(data=np.zeros((1, number_of_features)),
                                   columns=all_features_viz)

        print_frame[shap_value.feature_names] = shap_value.abs.mean(axis=0).values
        for z in print_frame.columns:
            print(f"{print_frame[z].to_numpy().squeeze()}")

        plt.figure()
        shap.plots.bar(shap_value, max_display=number_of_features, show=True)
        plt.show()
        # print(print_frame)
        # shap_values = print_frame  # pd.concat([shap_values, print_frame], axis='index', ignore_index=True)
        # print(shap_values)
    # todo select features to plot
    # features_to_observe = [0, 1]
    # plot_classifier(selector, x_test, y_test, features_to_observe, [all_features_c[i] for i in features_to_observe])

    # if classifier_name != "NN":
    #     print(f"{selector.score(x_test, y_test)}")
    #     tmp_all_acc.append(selector.score(x_test, y_test))
    # else:
    #     print(y_test)
    #     print(y_pred)
    #     print(f"{accuracy_score(y_test, y_pred)}")
    #     tmp_all_acc.append(accuracy_score(y_test, y_pred))
    try:
        print(f"accuracy train: {selector.score(x_train, y_train)}")
        print(f"accuracy test: {selector.score(x_test, y_test)}")
        print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
        print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred, labels=np.unique(y_all['block_estimation']).tolist())}")
        tmp_all_acc.append(accuracy_score(y_test, y_pred))
    except:
        print("probably NN")
        print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred, labels=np.unique(y_all['block_estimation']).tolist())}")
        tmp_all_acc.append(accuracy_score(y_test, y_pred))
        tmp_all_cm.append(confusion_matrix(y_test, y_pred, labels=np.unique(y_all['block_estimation']).tolist()))

if classifier_name == "NN":
    print("start cm ----------------")
    for i, cm in enumerate(tmp_all_cm):
        print(f"confusion matrix of patietent {population[i]}:")
        print(cm)

print(f"mean accuracy: {np.mean(tmp_all_acc)}")
