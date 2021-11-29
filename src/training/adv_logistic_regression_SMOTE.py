import os
import logging
import pandas as pd

import comet_ml

from imblearn.over_sampling import SMOTE
import sklearn
from sklearn.linear_model import LogisticRegression

from utils import (
    clf_performance_metrics,
    log_experiment,
    register_model,
    TRAIN_COLS_PART_4,
    LABEL_COL,
    RANDOM_STATE,
    INFREQUENT_STOPPAGE_EVENTS,
)

logging.basicConfig(level=logging.INFO)

EXP_NAME = "lr_SMOTE"

EXP_PARAMS = {
    "random_state": RANDOM_STATE,
    "model_type": "logreg",
    "scaler": "standard",
}


redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]
NON_CORR_FEATS = list(set(TRAIN_COLS_PART_4) - set(redundant_feats))


def load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    na_mask = train[NON_CORR_FEATS + [LABEL_COL]].isnull().any(axis=1)
    logging.info(
        f"dropping {na_mask.sum()} rows (of {len(train)} total) containing nulls from train"
    )
    train = train[~na_mask]

    na_mask = val[NON_CORR_FEATS + [LABEL_COL]].isnull().any(axis=1)
    logging.info(
        f"dropping {na_mask.sum()} rows (of {len(val)} total) containing nulls from val"
    )
    val = val[~na_mask]

    X_train = train[NON_CORR_FEATS]
    Y_train = train[LABEL_COL].astype(int)

    X_val = val[NON_CORR_FEATS]
    Y_val = val[LABEL_COL].astype(int)

    return X_train, Y_train, X_val, Y_val


def preprocess(X_train, X_val):

    if "secondary_type" in X_train.columns:
        X_train["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
        X_val["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)

    if "prev_event_type" in X_train.columns:
        X_train["prev_event_type"].replace(
            to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
        )
        X_val["prev_event_type"].replace(
            to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
        )

        X_train = pd.get_dummies(X_train, ["shot", "prev_event"])
        X_val = pd.get_dummies(X_val, ["shot", "prev_event"])

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_train_scaled = pd.DataFrame(
        data=scaler.transform(X_train.values),
        index=X_train.index,
        columns=X_train.columns,
    )

    X_val_scaled = pd.DataFrame(
        data=scaler.transform(X_val.values), index=X_val.index, columns=X_val.columns
    )

    return X_train_scaled, X_val_scaled


if __name__ == "__main__":

    X_train, Y_train, X_val, Y_val = load_train_and_validation()
    X_train, X_val = preprocess(X_train, X_val)

    # Oversampling with SMOTE
    X_train = X_train.fillna(0)
    X_train, Y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, Y_train)

    # logistic regression with all the features in Q4
    clf = LogisticRegression(random_state=RANDOM_STATE).fit(X_train, Y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    res = pd.DataFrame({"y_true": Y_val, "y_preds": y_pred, "y_proba": y_proba})
    # saving predictions
    preds_path = f"./models/predictions/"
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)
    res.to_csv(os.path.join(preds_path, f"{EXP_NAME}.csv"), index=False)

    perf_metrics, confusion_matrix = clf_performance_metrics(
        Y_val, y_pred, y_proba, verbose=True
    )

    logging.info(f"Logging model information for {EXP_NAME}")
    comet_exp = log_experiment(
        EXP_PARAMS,
        perf_metrics,
        X_train,
        confusion_matrix=confusion_matrix,
        exp_name=EXP_NAME,
    )

    register_model(clf, comet_exp, f"./models/{EXP_NAME}.pickle")
