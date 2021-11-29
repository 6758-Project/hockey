""" Trains and saves a simple XGBoost tree model trained on distance and angle only.
"""
import os
import logging
import pandas as pd
import comet_ml
import xgboost as xgb
from utils import LABEL_COL, clf_performance_metrics, log_experiment, register_model

logging.basicConfig(level=logging.INFO)

TRAIN_COLS_BASELINE = ["distance_from_net", "angle"]

EXP_NAME = "xgboost_baseline"

EXP_PARAMS = {"model_type": "xgboost"}


def load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    X_train, Y_train = train[TRAIN_COLS_BASELINE], train[LABEL_COL].astype(int)
    X_val, Y_val = val[TRAIN_COLS_BASELINE], val[LABEL_COL].astype(int)

    return X_train, Y_train, X_val, Y_val


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = load_train_and_validation()

    params = {
        "max_depth": 2,
        "eta": 1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "num_boost_round": 100,
    }

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, Y_train)

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

    comet_exp = log_experiment(
        {**params, **EXP_PARAMS},
        perf_metrics,
        X_train,
        confusion_matrix=confusion_matrix,
        exp_name=EXP_NAME,
    )
    register_model(clf, comet_exp, f"./models/{EXP_NAME}.pickle")
