import os
import argparse
import logging
import pandas as pd
import numpy as np

import comet_ml

import sklearn
from sklearn.linear_model import LogisticRegression

from utils import (
    clf_performance_metrics,
    log_experiment,
    register_model,
    TRAIN_COLS_BASIC,
    LABEL_COL,
    RANDOM_STATE,
    INFREQUENT_STOPPAGE_EVENTS,
)

logging.basicConfig(level=logging.INFO)

EXP_PARAMS = {
    "random_state": RANDOM_STATE,
    "model_type": "logreg",
    "scaler": "standard",
}


col_subsets = [["distance_from_net"], ["angle"], ["distance_from_net", "angle"]]


BASELINE_EXP_NAMES = [
    "LR_" + sub for sub in ["distance_only", "angle_only", "distance_and_angle"]
]


ddef load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    na_mask = train[TRAIN_COLS_BASIC + [LABEL_COL]].isnull().any(axis=1)
    logging.info(
        f"dropping {na_mask.sum()} rows (of {len(train)} total) containing nulls from train"
    )
    train = train[~na_mask]

    na_mask = val[TRAIN_COLS_BASIC + [LABEL_COL]].isnull().any(axis=1)
    logging.info(
        f"dropping {na_mask.sum()} rows (of {len(val)} total) containing nulls from val"
    )
    val = val[~na_mask]

    X_train = train[TRAIN_COLS_BASIC]
    Y_train = train[LABEL_COL].astype(int)

    X_val = val[TRAIN_COLS_BASIC]
    Y_val = val[LABEL_COL].astype(int)

    return X_train, Y_train, X_val, Y_val


def preprocess(X_train, X_val):
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


def main(args):
    X_train, Y_train, X_val, Y_val = load_train_and_validation()
    X_train, X_val = preprocess(X_train, X_val)

    # The random classifier
    np.random.seed(RANDOM_STATE)
    y_proba = np.random.uniform(low=0, high=1, size=len(Y_val))
    y_pred = (y_proba >= 0.5).astype(int)
    res = pd.DataFrame({"y_true": Y_val, "y_preds": y_pred, "y_proba": y_proba})

    # saving predictions
    preds_path = f"./models/predictions/"
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)
    res.to_csv(os.path.join(preds_path, f"random_classifier.csv"), index=False)

    # the other 3 LR baselines
    for exp_name, subset in zip(BASELINE_EXP_NAMES, col_subsets):
        logging.info(f"Processing {exp_name}...")

        X_train_sub = X_train[subset].values
        X_val_sub = X_val[subset].values

        clf = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_sub, Y_train)
        y_pred = clf.predict(X_val_sub)
        y_proba = clf.predict_proba(X_val_sub)[:, 1]

        res = pd.DataFrame({"y_true": Y_val, "y_preds": y_pred, "y_proba": y_proba})
        # saving predictions
        preds_path = f"./models/predictions/"
        if not os.path.exists(preds_path):
            os.makedirs(preds_path)
        res.to_csv(os.path.join(preds_path, f"{exp_name}.csv"), index=False)

        perf_metrics, confusion_matrix = clf_performance_metrics(
            Y_val, y_pred, y_proba, verbose=True
        )

        if args.log_results:
            logging.info(f"Logging model information for {exp_name}")
            comet_exp = log_experiment(
                EXP_PARAMS,
                perf_metrics,
                X_train_sub,
                confusion_matrix=confusion_matrix,
                exp_name=exp_name,
            )

            if args.register_models:
                pickle_path = f"./models/{exp_name}.pickle"
                register_model(clf, comet_exp, pickle_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression models")

    parser.add_argument(
        "-l",
        "--log-results",
        dest="log_results",
        help="(boolean) if passed, logs model parameters and performance metrics to Comet.ml",
        action="store_true",
    )
    parser.set_defaults(log_results=False)

    parser.add_argument(
        "-s",
        "--register-models",
        dest="register_models",
        help="(boolean) if passed, upload model to registry",
        action="store_true",
    )
    parser.set_defaults(register_models=False)

    args = parser.parse_args()

    if not args.log_results and args.register_models:
        raise ValueError("Cannot register model if results are not logged")

    main(args)
