""" Trains and saves an optimal XGBoost tree model

Resources used:
    https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
    https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    https://www.kaggle.com/prashant111/xgboost-k-fold-cv-feature-importance?scriptVersionId=48823316&cellId=69

Decisions to make:
 * use xgb's cv() [prashant link above], or sklearn's CV [stuart above]?
"""
import argparse
import logging
logging.basicConfig(level = logging.INFO)

import pandas as pd
from scipy.stats import uniform, randint

import comet_ml

from sklearn.model_selection import RandomizedSearchCV
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

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False
    )

    hparam_search_ranges = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(1, 6),
        "colsample_bytree": uniform(.33, 1-.33),
        "subsample": uniform(0.5, 1-.5),
        "learning_rate": uniform(0.03, .67-.03),
        "gamma": uniform(0, 0.5)
    }

    search = RandomizedSearchCV(
        xgb_model, param_distributions=hparam_search_ranges,
        random_state=1729, n_iter=50, cv=5, verbose=2, n_jobs=1, return_train_score=True, refit=True
    )

    search.fit(X_train.values, Y_train.values)

    search_results_summary = pd.DataFrame(search.cv_results_)
    search_results_summary.to_csv("./figures/advanced_models/xgb_hparam_search_results.csv")

    clf_optimal = search.best_estimator_
    opt_params={key:getattr(clf_optimal,key) for key in hparam_search_ranges}

    y_pred = clf_optimal.predict(X_val)
    y_proba = clf_optimal.predict_proba(X_val)[:,1]

    perf_metrics = clf_performance_metrics(Y_val, y_pred, y_proba, verbose=True)

    comet_exp = log_experiment(opt_params, perf_metrics, X_train, 'GBDT_hparam_opt')
    register_model(clf_optimal, comet_exp, "./models/GBDT_hparam_opt.pickle")
