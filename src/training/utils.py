from collections.abc import Sequence
import os
from typing import List

import argparse
import logging
import warnings
logging.basicConfig(level = logging.INFO)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

import sklearn
from sklearn import metrics
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression

from comet_ml import Experiment





# Calculate and return the metrics using he model prediction
def clf_performance_metrics(y_true, y_pred, y_proba, verbose=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', sklearn.exceptions.UndefinedMetricWarning)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
    
    # Outputting information to the terminal using loging 
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
        
        
# Upload experiment to comet
def log_experiment(params, perf_metrics, X_train, exp_name=None, **EXP_KWARGS):
    comet_exp = Experiment(**EXP_KWARGS)

    comet_exp.log_parameters(params)
    comet_exp.log_metrics(perf_metrics)
    comet_exp.log_dataset_hash(X_train)

    if exp_name:
        comet_exp.set_name(exp_name)
        
        
        
        
        
        
        
        
        
        
        