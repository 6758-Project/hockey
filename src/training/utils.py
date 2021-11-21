""" Utilities for training models """

import logging
import warnings
logging.basicConfig(level = logging.INFO)

from comet_ml import Experiment
import sklearn


# Defined Constants

EXP_KWARGS = {
    'project_name': 'ift6758-hockey',
    'workspace': "tim-k-lee",
    'auto_param_logging': True
}

LABEL_COL = 'is_goal'

KNOWN_NON_TRAIN_COLS = [
    'game_id', 'event_index', 'description',
    'game_sec', 'time', 'time_remaining', 'date', 'prev_event_time_diff',
    'is_goal'
]

TRAIN_COLS_BASIC = [
    'period', 'goals_home', 'goals_away',
    'shooter_id', 'coordinate_x', 'coordinate_y', 'distance_from_net',
    'angle'
]

# TODO: throw out is_empty_net + associated rows?
# see https://piazza.com/class/krgt4sfrgfp278?cid=255
TRAIN_COLS_ADV = [
    'period_type', 'period', 'goals_home', 'goals_away',
    'shooter_team_name', 'shooter_id', 'secondary_type', 'goalie_name',
    'coordinate_x', 'coordinate_y', 'distance_from_net', 'angle',
    'angle_between_prev_event', 'distance_from_prev_event', 'speed',
    'prev_event_type', 'is_rebound', 'rebound_angle', 'is_empty_net'
]




# Functions


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