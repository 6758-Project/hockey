""" 
Preprocessing input data based on selected model.
"""

import pandas as pd
import sklearn
from utils import INFREQUENT_STOPPAGE_EVENTS, SHAP_COLS, LASSO_COLS


def Process_Data(X_Test, Y_Test, X_Train, Model_Type):

    if Model_Type == "logreg":
        X_processed, Y_processed = LG_preprocess(X_Train, X_Test, Y_Test)
        return X_processed, Y_processed

    if Model_Type == "logreg_all":
        X_processed, Y_processed = preprocess_lr_all(X_Train, X_Test, Y_Test)
        return X_processed, Y_processed

    if Model_Type == "logreg_SMOTE" or Model_Type == "logreg_non_corr_feats":
        X_processed, Y_processed = preprocess_lr_smote(X_Train, X_Test, Y_Test)
        return X_processed, Y_processed

    if Model_Type == "xgboost_SHAP":
        X_processed = XGB_SHAP_preprocess(X_Test)
        return X_processed, Y_Test

    if Model_Type == "xgboost_lasso":
        X_processed = XGB_Lasso_preprocess(X_Test)
        return X_processed, Y_Test

    if (
        Model_Type == "xgboost_non_corr"
        or Model_Type == "logreg_SMOTE"
        or Model_Type == "xgboost_SMOTE"
    ):
        X_processed = XGB_Non_Corr_preprocess(X_Test)
        return X_processed, Y_Test

    if Model_Type == "NN_MLP":
        X_processed = NN_preprocess(X_Test, X_Train)
        return X_processed, Y_Test


# Logistic Regression
def LG_preprocess(X_train, X_data, y_data):

    na_mask = X_train.isnull().any(axis=1)
    X_train = X_train[~na_mask]

    na_mask = X_data.isnull().any(axis=1)
    X_data = X_data[~na_mask]
    y_data = y_data[~na_mask]

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_data_scaled = pd.DataFrame(
        data=scaler.transform(X_data.values), index=X_data.index, columns=X_data.columns
    )

    return X_data_scaled, y_data


# Logistic Regression
def preprocess_lr_all(X_train, X_data, y_data):

    na_mask = X_train.isnull().any(axis=1)
    X_train = X_train[~na_mask]

    na_mask = X_data.isnull().any(axis=1)
    X_data = X_data[~na_mask]
    y_data = y_data[~na_mask]

    X_train["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)

    X_train["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_train = pd.get_dummies(X_train, ["shot", "prev_event"])
    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_data_scaled = pd.DataFrame(
        data=scaler.transform(X_data.values), index=X_data.index, columns=X_data.columns
    )

    return X_data_scaled, y_data


# Logistic Regression
def preprocess_lr_smote(X_train, X_data, y_data):

    X_train["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_train["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )
    X_train = pd.get_dummies(X_train, ["shot", "prev_event"])

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )
    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])

    na_mask = X_train.isnull().any(axis=1)
    X_train = X_train[~na_mask]

    na_mask = X_data.isnull().any(axis=1)
    X_data = X_data[~na_mask]
    y_data = y_data[~na_mask]

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_data_scaled = pd.DataFrame(
        data=scaler.transform(X_data.values), index=X_data.index, columns=X_data.columns
    )

    # the redundant features after inspecting them in "./notebooks/M2_detect-feat-correlation.ipynb"
    redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]

    # Training and validation data of the selected features
    selected_feats = X_data_scaled.columns.difference(redundant_feats)
    X_data_scaled = X_data_scaled[selected_feats]

    X_data_scaled = X_data_scaled.fillna(0)

    return X_data_scaled, y_data


# XGBoost with SHAP
def XGB_SHAP_preprocess(X_data):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])

    X_data = X_data[SHAP_COLS]

    return X_data


# XGBoost with Lasso
def XGB_Lasso_preprocess(X_data):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])

    X_data = X_data[LASSO_COLS]

    return X_data


def XGB_Non_Corr_preprocess(X_data):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])

    #    the redundant features after inspecting them in "./notebooks/M2_detect-feat-correlation.ipynb"
    redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]

    # Training and validation data of the selected features
    selected_feats = X_data.columns.difference(redundant_feats)
    X_data = X_data[selected_feats]

    return X_data


# Neural Network with Advance
def NN_preprocess(X_data, X_train):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_train["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_train["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])
    X_train = pd.get_dummies(X_train, ["shot", "prev_event"])

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_data_scaled = pd.DataFrame(
        data=scaler.transform(X_data.values), index=X_data.index, columns=X_data.columns
    )

    X_data_scaled = X_data_scaled.fillna(0)

    return X_data_scaled
