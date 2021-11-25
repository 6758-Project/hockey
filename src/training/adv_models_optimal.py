""" Trains and saves a simple XGBoost tree model trained on distance and angle only.
"""
import argparse
import logging
logging.basicConfig(level = logging.INFO)

import pandas as pd

import comet_ml
import xgboost as xgb

from utils import (
    EXP_KWARGS, INFREQUENT_STOPPAGE_EVENTS, TRAIN_COLS_PART_4, LABEL_COL,
    clf_performance_metrics, log_experiment, register_model
)

EXP_PARAMS = {
    "model_type": "xgboost"
}


def load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    X_train, Y_train =  train[TRAIN_COLS_PART_4], train[LABEL_COL].astype(int)
    X_val, Y_val = val[TRAIN_COLS_PART_4], val[LABEL_COL].astype(int)

    return X_train, Y_train, X_val, Y_val


def preprocess(X_train, X_val):
    # TODO: try working with categorical data directly on GPU
    # see "1.4.17 Categorical Data" in https://buildmedia.readthedocs.org/media/pdf/xgboost/latest/xgboost.pdf

    X_train['secondary_type'].replace({'Tip-in': 'Deflection'}, inplace=True)
    X_train['prev_event_type'].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value = 'STOP', inplace=True
    )

    X_val['secondary_type'].replace({'Tip-in': 'Deflection'}, inplace=True)
    X_val['prev_event_type'].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value = 'STOP', inplace=True
    )

    X_train = pd.get_dummies(X_train, ['shot', 'prev_event'])
    X_val = pd.get_dummies(X_val)

    return X_train, X_val




if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = load_train_and_validation()
    X_train, X_val = preprocess(X_train, X_val)

    # chosen via results of hparam optimization search
    params = {
        'max_depth':1, 'n_estimators': 275, 'learning_rate': .1, 'gamma': .1,
        'objective':'binary:logistic', 'eval_metric': 'logloss', 'use_label_encoder': False
    }

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:,1]
    res = pd.DataFrame({
        'y_true': Y_val,
        'y_preds': y_pred,
        'y_proba': y_proba
    })

    exp_name = "xgboost_optimal"
    res.to_csv(f"./models/predictions/{exp_name}.csv", index=False)

    perf_metrics = clf_performance_metrics(Y_val, y_pred, y_proba, verbose=True)

    comet_exp = log_experiment(params, perf_metrics, X_train, 'GBDT_hparam_opt')
    register_model(clf_optimal, comet_exp, "./models/GBDT_hparam_opt.pickle")

