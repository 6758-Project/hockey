""" Trains and saves a simple XGBoost tree model trained on distance and angle only.
"""
import argparse
import logging
logging.basicConfig(level = logging.INFO)

import pandas as pd

import xgboost as xgb

from utils import EXP_KWARGS

TRAIN_COLS = ['distance_from_net', 'angle']
LABEL_COL = 'is_goal'

EXP_PARAMS = {
    "model_type": "xgboost"
}  # TODO log results to comet.ml


def load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    xg_train = xgb.DMatrix(
        data = train[TRAIN_COLS],
        label = train[LABEL_COL].astype(int)
    )
    xg_train.save_binary('./data/processed/train_simple.buffer')
    xg_train = xgb.DMatrix('./data/processed/train_simple.buffer')

    xg_val = xgb.DMatrix(
        data = val[TRAIN_COLS],
        label = val[LABEL_COL].astype(int)
    )
    xg_val.save_binary('./data/processed/val_simple.buffer')
    xg_val = xgb.DMatrix('./data/processed/val_simple.buffer')

    return xg_train, xg_val




if __name__ == "__main__":
    xg_train, xg_val = load_train_and_validation()

    param = {
        'max_depth':2, 'eta':1, 'objective':'binary:logistic',
        'eval_metric': 'logloss'
    }

    clf = xgb.train(param,xg_train, num_boost_round=100)

    res = pd.DataFrame({
        'y_true': xg_val.get_label(),
        'y_preds': None,
        'y_proba': clf.predict(xg_val)
    })

    exp_name = "xgboost_distance_angle_only"
    res.to_csv(f"./models/predictions/{exp_name}.csv", index=False)
