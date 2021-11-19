import argparse
import logging
import warnings
logging.basicConfig(level = logging.INFO)

import pandas as pd

from comet_ml import Experiment

import sklearn
from sklearn.linear_model import LogisticRegression

from visuals import generate_shot_classifier_charts

TRAIN_COLS_BASIC = [
    'period', 'goals_home', 'goals_away',
    'shooter_id', 'coordinate_x', 'coordinate_y', 'distance_from_net',
    'angle'
]

LABEL_COL = 'is_goal'

RANDOM_STATE = 1729

EXP_KWARGS = {
    'project_name': 'ift6758-hockey',
    'workspace': "tim-k-lee",
    'auto_param_logging': False
}

EXP_PARAMS = {
    "random_state": RANDOM_STATE,
    "model_type": "logreg",
    "scaler": "standard",
}


def load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    na_mask = train[TRAIN_COLS_BASIC+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask.sum()} rows (of {len(train)} total) containing nulls from train")
    train = train[~na_mask]

    na_mask = val[TRAIN_COLS_BASIC+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask.sum()} rows (of {len(val)} total) containing nulls from val")
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
        columns=X_train.columns
    )

    X_val_scaled = pd.DataFrame(
        data=scaler.transform(X_val.values),
        index=X_val.index,
        columns=X_val.columns
    )

    return X_train_scaled, X_val_scaled


def clf_performance_metrics(y_true, y_pred, y_proba, verbose=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', sklearn.exceptions.UndefinedMetricWarning)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)

    if verbose:
        logging.info("Accuracy is {:6.3f}".format(acc))
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
        logging.info(f"Confusion Matrix:\n {cm}")

        logging.info("F1 score is {:6.3f}".format(f1))
        logging.info("Precision score is {:6.3f}".format(precision))
        logging.info("Recall score is {:6.3f}".format(recall))

    res = {
        'accuracy': acc, 'f1_score': f1, 'precision': precision, 'recall': recall
    }

    return res


def log_experiment(params, perf_metrics, X_train, exp_name=None):
    comet_exp = Experiment(**EXP_KWARGS)

    comet_exp.log_parameters(params)
    comet_exp.log_metrics(perf_metrics)
    comet_exp.log_dataset_hash(X_train)

    if exp_name:
        comet_exp.set_name(exp_name)


def main(args):
    X_train, Y_train, X_val, Y_val = load_train_and_validation()
    X_train, X_val = preprocess(X_train, X_val)

    y_trues, y_preds, y_probas = [], [], []
    exp_names = ['baseline_logreg_'+sub for sub in ['distance_only', 'angle_only', 'distance_and_angle']]
    col_subsets = [['distance_from_net'], ['angle'], ['distance_from_net', 'angle']]
    for exp_name, subset in zip(exp_names, col_subsets):
        logging.info(f"Processing {exp_name}...")
        X_train_sub = X_train[subset].values
        X_val_sub = X_val[subset].values

        clf = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_sub, Y_train)
        y_pred = clf.predict(X_val_sub)
        y_proba = clf.predict_proba(X_val_sub)[:,1]

        y_trues.append(Y_val)
        y_preds.append(y_pred)
        y_probas.append(y_proba)

        perf_metrics = clf_performance_metrics(Y_val, y_pred, y_proba, verbose=True)

        if args.log_results:
            log_experiment(EXP_PARAMS, perf_metrics, X_train_sub, exp_name=exp_name)

    if args.generate_charts:
        title = "Visual Summary - Simple Logistic Regressions"
        image_dir = "./src/training/visualizations/simple_log_reg/"

        generate_shot_classifier_charts(
            y_trues, y_preds, y_probas, exp_names,
            title=title, image_dir=image_dir
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Models')

    parser.add_argument('-c', '--generate-charts', dest="generate_charts",
                    help='(boolean) if passed, generate model visuals',
                    action='store_true')
    parser.set_defaults(generate_charts=False)

    parser.add_argument('-l', '--log-results', dest="log_results",
                    help='(boolean) if passed, logs model parameters and performance metrics to Comet.ml',
                    action='store_true')
    parser.set_defaults(log_results=False)

    args = parser.parse_args()

    main(args)
