from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, get_scorer
import shap
import pandas as pd
import numpy as np


def get_pipeline_for_features(classifier, data_pre_processor, X=None, y=None, feature_list=None):
    steps = []
    # TODO we need no encoding
    # attributes_that_require_encoding = list(set(feature_list) & set(categorical_attributes))
    # if attributes_that_require_encoding:
    #     steps.append(("Categorical Encoder", make_column_transformer((OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), attributes_that_require_encoding), remainder="passthrough")))
    if data_pre_processor is not None:
        steps.append(('data-pre-processor', data_pre_processor))
    steps.append(("Learner", classifier))
    return Pipeline(steps)


def fit_classifier(learner, x_train, x_test, y_train, y_test, scoring="accuracy", use_shap=False, n_classes=2):
    learner_c = clone(learner).fit(x_train.to_numpy(), y_train.values.ravel())
    y_pred = learner_c.predict(x_test.values)

    shap_values = pd.DataFrame(data=np.zeros((1, x_train.shape[1])), columns=x_train.columns)
    if use_shap:
        explainer = shap.KernelExplainer(learner_c.predict_proba, shap.sample(x_train, 100))  # x_test or x_train?
        shap_value = explainer(x_test)
        shap_value.values = shap_value.values[:, :, 1]
        shap_value.base_values = shap_value.base_values[:, 1]
        shap_values[:] = shap_value.abs.mean(axis=0).values

    scorer = get_scorer(scoring)  # roc_auc
    c_m = confusion_matrix(y_test, y_pred, labels=range(n_classes))
    # return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), pl_interpretable
    return scorer(learner_c, x_test.values, y_test), c_m, shap_values  # , pl_interpretable


def fit_classifier_cf(learner, x_train, x_test, y_train, y_test, scoring="accuracy", use_shap=False):
    learner_c = clone(learner).fit(x_train.to_numpy(), y_train.values.ravel())
    y_pred = learner_c.predict(x_test.values)

    shap_values = pd.DataFrame(data=np.zeros((1, x_train.shape[1])), columns=x_train.columns)
    if use_shap:
        explainer = shap.KernelExplainer(learner_c.predict_proba, shap.sample(x_train, 100))  # x_test or x_train?
        shap_value = explainer(x_test)
        shap_value.values = shap_value.values[:, :, 1]
        shap_value.base_values = shap_value.base_values[:, 1]
        shap_values[:] = shap_value.abs.mean(axis=0).values

    scorer = get_scorer(scoring)  # roc_auc

    # return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), pl_interpretable
    return scorer(learner_c, x_test.values, y_test), confusion_matrix(y_test, y_pred), shap_values, learner_c


