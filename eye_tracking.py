import os

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import platform
import pandas as pd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import LearningCurveDisplay

from utils.learner_pipeline import get_pipeline_for_features, fit_classifier

from sklearn.feature_selection import VarianceThreshold
from plotting_scripts.roc_curve_plotting import get_mccv_ROC_display

# from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE, ADASYN

from utils.feature_loader import load_eye_tracking_data, load_eye_tracking_data_slice, load_eye_tracking_data_tw

from sklearn.metrics import auc, get_scorer

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

import shap

from plotting_scripts.plot_physio import plot_physio3D, plot_physio2D


# load data
# X, y = load_eye_tracking_data(make_binary=True, load_preprocessed=True)
# X.drop(columns=["participant", "time", "robot"], inplace=True)


# def fit_classifer(learner, x_train, x_test, y_train, y_test, n_classes=2, use_shap=False):
#     learner_c = clone(learner).fit(x_train.to_numpy(), y_train.values.ravel())
#     y_pred = learner_c.predict(x_test.to_numpy())
#
#     shap_values = pd.DataFrame(data=np.zeros((1, x_train.shape[1])), columns=x_train.columns)
#     if use_shap:
#         explainer = shap.KernelExplainer(learner_c.predict_proba, shap.sample(x_train, 100))  # x_test or x_train?
#         shap_value = explainer(X_test)
#         shap_value.values = shap_value.values[:, :, 1]
#         shap_value.base_values = shap_value.base_values[:, 1]
#         shap_values[:] = shap_value.abs.mean(axis=0).values
#
#     if n_classes == 2:
#         scorer = get_scorer("accuracy")  # roc_auc
#     else:
#         scorer = get_scorer("accuracy")  # roc_auc_ovr
#
#     # return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), pl_interpretable
#     return scorer(learner_c, x_test, y_test), confusion_matrix(y_test, y_pred), shap_values  # , pl_interpretable
#

if __name__ == '__main__':
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
    elif platform.system() == "Linux":
        matplotlib.use('TkAgg')
    num_splits = 10
    n_classes = 2
    tw = 20  # time window in seconds
    label = "duration_estimate"  # "ppot" or "duration_estimate"
    m_workers = os.cpu_count()
    save_results = True
    scoring = "accuracy"  # "roc_auc"?
    bls = True  # baseline subtraction
    tag = "_bls" if bls else ""
    use_shap = False

    X_train, y_train = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, tw=tw, label_name=[label], bls=bls)
    X_test, y_test = load_eye_tracking_data_tw(number_of_classes=n_classes, load_preprocessed=True, tw=tw,
                                               label_name=[label], load_test=True, bls=bls)

    # use preprocessing: the best subset
    # X = X[['sub_max_speed_fix', 'sub_mean_dispersion_fix', 'sub_mean_duration_fix', 'sub_min_dispersion_fix', 'sub_min_speed_fix', 'sub_number_clusters_fix']]
    # pca = PCA()  # n_components=7
    # X_pp = pca.fit_transform(X)
    # X = pd.DataFrame(X_pp, columns=[f"PCA_{i}" for i in range(7)])

    # upsampling the data
    # sm = BorderlineSMOTE(random_state=42)  # random_state=42
    # X, y = sm.fit_resample(X, y)

    print(f"X data shape: {X_train.shape}")
    print(f"y data shape: {y_train.shape}\n")
    _, counts = np.unique(y_test, return_counts=True)

    for i, c in enumerate(counts):
        print(f"Class {i} count: {c/np.sum(counts)}")
    print(f"distribution of y: {np.unique(y_train, return_counts=True)}")
    print(f"distribution of y_test: {np.unique(y_test, return_counts=True)}")

    if n_classes == 2 and label == "duration_estimate":
        # learner = ExtraTreesClassifier(n_estimators=1024)
        learner = ExtraTreesClassifier(criterion='entropy', max_features=0.9197700535609098, min_samples_leaf=3,
                                       min_samples_split=14, n_estimators=512, warm_start=True)
        preprocessor = None  # MinMaxScaler()
    elif n_classes == 2 and label == "ppot":
        preprocessor = VarianceThreshold()
        learner = ExtraTreesClassifier()
    elif n_classes == 3 and label == "duration_estimate":
        preprocessor = VarianceThreshold()
        learner = ExtraTreesClassifier(max_features=0.707960088654249, min_samples_split=6, n_estimators=512,
                                       warm_start=True)
    elif n_classes == 3 and label == "ppot":
        preprocessor = None
        learner = ExtraTreesClassifier(max_features=0.5350930770802981, n_estimators=512, warm_start=True)
    else:
        print("no optimized model for that configuration using standard ExtraTreesClassifier")
        learner = ExtraTreesClassifier(n_estimators=1024)

    pl_interpretable = get_pipeline_for_features(learner, preprocessor)

    ## generate ROC curve
    # fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    # get_mccv_ROC_display(pl_interpretable, X, y, repeats=num_splits, ax=axs)  #
    # plt.savefig(f"plots/eye_tracking_analysis/roc_curve_repeats_{num_splits}_extra_tree_{n_classes}_classes_n_features:{X.shape[1]}.pdf")
    # plt.show()

    ## play trough
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=0)

    acc_list = []
    futures = []
    conf_m = np.zeros((n_classes, n_classes))

    # pbar = tqdm(total=num_splits)
    # m_workers = os.cpu_count()

    # score, conf_ma, shap_val = fit_classifer(pl_interpretable, X, X_test, y, y_test, n_classes, False)
    # shap_val = shap_val.T
    # shap_val = shap_val.rename(columns={0: "shap_values"})
    # shap_val = shap_val.sort_values(by="shap_values", ascending=False)
    # shap_val.to_csv(f"results/shap/{label}_{n_classes}_shap_values.csv")
    # sv = shap.Explanation(values=shap_val["shap_values"].to_numpy(), feature_names=X_test.columns)
    # # TODO does not work because shap_values not a explanantion object but a dataframe
    # shap.plots.bar(sv, max_display=26, show=True)
    # plt.show()
    pbar = tqdm(total=num_splits)
    shap_values_list = []
    accuracy_result_frame = pd.DataFrame(columns=["accuracy"] * num_splits)
    with ProcessPoolExecutor(max_workers=m_workers) as executor:
        # for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        for seed in range(num_splits):
            # X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=0.8, random_state=seed)
            # print(f"Fold {i}")
            # x_train, x_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
            # y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)
            # upsampling the data
            # sm = ADASYN()  # random_state=42
            # X_train, y_train = sm.fit_resample(X_train, y_train)

            futures.append(
                executor.submit(
                    fit_classifier, learner, X_train, X_test, y_train, y_test, scoring=scoring, use_shap=use_shap, n_classes=n_classes
                )
            )


        def _cb(future):
            pbar.update(1)


        for future in futures:
            future.add_done_callback(_cb)

        as_completed(futures)

        shap_vals = pd.DataFrame(columns=X_train.columns)
        for future in futures:
            acc, conf_m_tmp, shap_values = future.result()
            acc_list.append(acc)
            conf_m += conf_m_tmp
            shap_values_list.append(shap_values)

        #     # todo get best lerner and do shap analysis
        # conf_m /= num_splits
    pbar.close()
    print("\n")
    print(f"Mean accuracy: {np.mean(acc_list)}")
    print(f"Std accuracy: {np.std(acc_list)}")
    print(f"Max accuracy: {np.max(acc_list)}")
    print(f"Min accuracy: {np.min(acc_list)}")
    print(f"Confusion matrix: \n{conf_m}")
    accuracy_result_frame.loc[0] = acc_list
    if save_results:
        accuracy_result_frame.to_csv(
            f"results/eye_tracking_{n_classes}_classes/all/accuracy_{label}_tw_{tw}{tag}.csv")
    print(shap_values_list[0])

    if use_shap:
        shap_values = shap_values_list[0][0]
        print(type(shap_values))
        print(shap_values.shape)
        plt.figure()
        plt.boxplot(shap_values, labels=X_test.columns)
        plt.xticks(rotation=90)
        plt.show()
        mean = np.mean(shap_values, axis=0)
        print(mean.shape)
        # shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
        # sv = shap.Explanation(values=shap_values["shap_values"].to_numpy(), feature_names=X_test.columns)
        # shap.plots.bar(sv, max_display=26, show=True)
        # shap_values = shap_values.T
        # shap_values = shap_values.rename(columns={0: "shap_values"})
        # shap_values = shap_values.sort_values(by="shap_values", ascending=False)
        # shap_values.to_csv(f"results/shap/{label}_{n_classes}_shap_values{tag}.csv")
        sv = shap.Explanation(values=mean, feature_names=X_test.columns)
        # # TODO does not work because shap_values not a explanantion object but a dataframe
        shap.plots.bar(sv, max_display=26, show=True)
        plt.show()

    # plt.figure()
    # plt.hist(acc_list, label=r'Mean Accuracy (ACC = %0.2f $\pm$ %0.2f)' % (np.mean(acc_list), np.std(acc_list)))
    # plt.xlabel("Accuracy")  # 0.5410447761
    # upper_lim = np.max(np.unique(acc_list, return_counts=True)[1]) * 10
    # # plt.vlines(majority_class, 0, upper_lim, colors="red", label="Majority class", linestyles="--")
    # plt.legend()
    # # plt.savefig(
    # #     f"plots/eye_tracking_analysis/accuracy_hist_repeats_{num_splits}_extra_tree_{n_classes}_classes_n_features:{X.shape[1]}.pdf")
    # #
    # # if n_classes == 2:
    # #     disp = ConfusionMatrixDisplay(confusion_matrix=conf_m,
    # #                                   display_labels=["slow", "fast"])
    # # elif n_classes == 3:
    # #     disp = ConfusionMatrixDisplay(confusion_matrix=conf_m,
    # #                                   display_labels=["slow", "medium", "fast"])
    # #
    # # disp.plot()
    # # plt.savefig(
    # #     f"plots/eye_tracking_analysis/confusion_matrix_repeats_{num_splits}_extra_tree_{n_classes}_classes_n_features:{X.shape[1]}.pdf")
    # plt.show()
