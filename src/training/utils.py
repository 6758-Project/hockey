""" Utilities for training models """

import logging
import pickle
import warnings
from comet_ml import Experiment
from comet_ml import ConfusionMatrix
import sklearn

logging.basicConfig(level=logging.INFO)


# Defined Constants

EXP_KWARGS = {
    "project_name": "ift6758-hockey",
    "workspace": "tim-k-lee",
    "auto_param_logging": True,
}

LABEL_COL = "is_goal"

INFREQUENT_STOPPAGE_EVENTS = [
    "PERIOD_START",
    "PERIOD_READY",
    "PERIOD_END",
    "SHOOTOUT_COMPLETE",
    "PERIOD_OFFICIAL",
    "GAME_OFFICIAL",
    "PENALTY",
    "GOAL",
    "CHALLENGE",
]

KNOWN_NON_TRAIN_COLS = [
    "game_id",
    "event_index",
    "description",
    "game_sec",
    "time",
    "time_remaining",
    "date",
    "prev_event_time_diff",
    "is_goal",
]

TRAIN_COLS_DISTANCE = ["distance_from_net"]
TRAIN_COLS_ANGLE = ["angle"]
TRAIN_COLS_BASELINE = ["distance_from_net", "angle"]

TRAIN_COLS_BASIC = [
    "period",
    "goals_home",
    "goals_away",
    "shooter_id",
    "coordinate_x",
    "coordinate_y",
    "distance_from_net",
    "angle",
]


SHAP_COLS = [
            "distance_from_net",
            "is_rebound",
            "prev_event_SHOT",
            "prev_event_time_diff",
            "angle",
            "is_empty_net",
            "shot_Snap Shot",
            "shot_Slap Shot",
            "distance_from_prev_event",
            "coordinate_y",
            "prev_event_HIT"
            ]

LASSO_COLS = [
            "coordinate_x",
            "coordinate_y",
            "distance_from_net",
            "angle",
            "angle_between_prev_event",
            "distance_from_prev_event",
            "prev_event_time_diff",
            "speed",
            "is_rebound",
            "rebound_angle",
            "is_empty_net",
            "prev_event_x_coord",
            "shot_Backhand",
            "shot_Tip-In",
            "shot_Wrist Shot",
            "prev_event_FACEOFF",
            "prev_event_GIVEAWAY",
            "prev_event_HIT",
            "prev_event_SHOT",
            ]

NON_CORR_COLS = [
    "period",
    "coordinate_x",
    "coordinate_y",
    "is_rebound",
]

# TODO: throw out is_empty_net + associated rows?
# see https://piazza.com/class/krgt4sfrgfp278?cid=255
TRAIN_COLS_PART_4 = [
    "game_sec",
    "period",
    "secondary_type",
    "coordinate_x",
    "coordinate_y",
    "distance_from_net",
    "angle",
    "prev_event_type",
    "angle_between_prev_event",
    "distance_from_prev_event",
    "prev_event_time_diff",
    "speed",
    "is_rebound",
    "rebound_angle",
    "is_empty_net",
]  # 'period_type', 'shooter_team_name', 'shooter_id', 'goalie_name',

TRAIN_COLS_LASSO = [
    "game_sec",
    "period",
    "secondary_type",
    "coordinate_x",
    "coordinate_y",
    "distance_from_net",
    "angle",
    "prev_event_type",
    "angle_between_prev_event",
    "distance_from_prev_event",
    "prev_event_time_diff",
    "speed",
    "is_rebound",
    "rebound_angle",
    "is_empty_net",
    "prev_event_x_coord",
] 

RANDOM_STATE = 1729


# Functions


def clf_performance_metrics(y_true, y_pred, y_proba, verbose=False):
    """Generate the model performance metrics.

    Args:
        y_true: True label
        y_pred: Predicted label
        y_proba: Probability of label prediction
        verbose: Display information of metrics
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)

        cm = ConfusionMatrix()
        cm.compute_matrix(y_true, y_pred)

    if verbose:
        logging.info("Accuracy is {:6.3f}".format(acc))
        logging.info(f"Confusion Matrix:\n {cm}")
        logging.info("F1 score is {:6.3f}".format(f1))
        logging.info("Precision score is {:6.3f}".format(precision))
        logging.info("Recall score is {:6.3f}".format(recall))

    res = {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}

    return res, cm


def log_experiment(params, perf_metrics, X_train, confusion_matrix, exp_name=None):
    """Log model parameters and performance metrics to Comet.
    Args:
        params: Model parameters
        perf_metrics: Performance information of the model
        X_train: Features used to train the model
        exp_name: Name of the experiment
        pickle_path: file name and folder directry where the pickle file is located
    Returns:
        comet_exp: a Comet.ml Experiment object
    """
    comet_exp = Experiment(**EXP_KWARGS)

    comet_exp.log_parameters(params)
    comet_exp.log_metrics(perf_metrics)
    comet_exp.log_dataset_hash(X_train)
    comet_exp.log_confusion_matrix(matrix=confusion_matrix)

    if exp_name:
        comet_exp.set_name(exp_name)

    return comet_exp


def register_model(clf, comet_exp, pickle_path=None):
    """Uploads model to comet.ml registry

    Args:
        clf: model to be saved
        comet_exp: a Comet.ml experiment
        pickle_path: file name and folder directry where the pickle file will be saved
    """
    if not clf or not pickle_path:
        logging.info("Parameters missing, cannot save files")
    else:
        logging.info("Saving model to pickle file")
        with open(pickle_path, "wb") as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

        comet_exp.log_model(name=comet_exp.get_name(), file_or_folder=pickle_path)
