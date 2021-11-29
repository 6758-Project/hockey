""" Trains and saves an XGBoost tree model trained on selected features with two different methods
"""
import os
import logging
import pandas as pd
import comet_ml
import xgboost as xgb
from sklearn import feature_selection
from sklearn import linear_model


from utils import (
    INFREQUENT_STOPPAGE_EVENTS,
    TRAIN_COLS_PART_4,
    LABEL_COL,
    clf_performance_metrics,
    log_experiment,
    register_model,
)

logging.basicConfig(level=logging.INFO)

EXP_NAME = "xgboost_lasso"

EXP_PARAMS = {"model_type": "xgboost"}


def load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    X_train, Y_train = train[TRAIN_COLS_PART_4], train[LABEL_COL].astype(int)
    X_val, Y_val = val[TRAIN_COLS_PART_4], val[LABEL_COL].astype(int)

    return X_train, Y_train, X_val, Y_val


def preprocess(X_train, X_val):
    # TODO: try working with categorical data directly on GPU
    # see "1.4.17 Categorical Data" in https://buildmedia.readthedocs.org/media/pdf/xgboost/latest/xgboost.pdf

    X_train["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_train["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_val["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_val["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_train = pd.get_dummies(X_train, ["shot", "prev_event"])
    X_val = pd.get_dummies(X_val, ["shot", "prev_event"])

    # prepare training data to fit lasso model
    X_train = X_train.fillna(0)

    return X_train, X_val


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = load_train_and_validation()
    X_train, X_val = preprocess(X_train, X_val)

    lass_feat_selector = feature_selection.SelectFromModel(
        estimator=linear_model.Lasso(alpha=0.001)
    ).fit(X_train, Y_train)
    selected_feats = list(lass_feat_selector.get_feature_names_out())

    # new training and validation data with the selected features
    X_train_lasso, X_val_lasso = X_train[selected_feats], X_val[selected_feats]

    # chosen via results of hparam optimization search on the selected features using Lasso
    params = {
        "max_depth": 5,
        "n_estimators": 495,
        "learning_rate": 0.062,
        "gamma": 0.184,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train_lasso, Y_train)

    y_pred = clf.predict(X_val_lasso)
    y_proba = clf.predict_proba(X_val_lasso)[:, 1]
    res = pd.DataFrame({"y_true": Y_val, "y_preds": y_pred, "y_proba": y_proba})

    # saving predictions
    preds_path = f"./models/predictions/"
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)
    res.to_csv(os.path.join(preds_path, f"{EXP_NAME}.csv"), index=False)

    perf_metrics, confusion_matrix = clf_performance_metrics(
        Y_val, y_pred, y_proba, verbose=True
    )

    comet_exp = log_experiment(
        {**params, **EXP_PARAMS},
        perf_metrics,
        X_train_lasso,
        confusion_matrix=confusion_matrix,
        exp_name=EXP_NAME,
    )
    register_model(clf, comet_exp, f"./models/{EXP_NAME}.pickle")
