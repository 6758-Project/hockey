""" A utility for creating model quality charts from predictions

Usage:
Modify the generate_adv_model_figures function directly with:
 1. the location of the prediction filenames, and
 2. the model name to associate with the predictions

"""

import pandas as pd

from visuals import generate_shot_classifier_charts


def generate_adv_model_figures(experiment_prediction_filenames=None , title = "Default", image_dir = "./"):

    exp_preds = {exp: pd.read_csv(fname) for exp, fname in experiment_prediction_filenames.items()}

    exp_names, y_trues, y_preds, y_probas = [], [], [], []
    for exp, preds in exp_preds.items():
        exp_names.append(exp)
        y_trues.append(preds['y_true'].values)
        y_preds.append(preds['y_preds'].values)
        y_probas.append(preds['y_proba'].values)

    title = "Visual Summary - XGBoost Models"
    image_dir = "./figures/advanced_models/"

    generate_shot_classifier_charts(
        y_trues, y_preds, y_probas, exp_names,
        title=title, image_dir=image_dir
    )




if __name__ == '__main__':

    exp_pred_filenames = {
        "xgb_distance_and_angle_only": "./models/predictions/xgboost_distance_angle_only.csv"
    }
    title = "Visual Summary - XGBoost Models"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames = exp_pred_filenames,
        title=title, image_dir=image_dir
    )

    exp_pred_filenames = {
        "NN_distance": "./models/predictions/NN_distance.csv",
        "NN_baseline":"./models/predictions/NN_baseline.csv",
        "NN_basic":"./models/predictions/NN_basic.csv",
        "NN_adv":"./models/predictions/NN_adv.csv"
    }
    title = "Visual Summary - Neural Network Models"
    image_dir = "./figures/exploration/"
    generate_adv_model_figures(
        experiment_prediction_filenames = exp_pred_filenames,
        title=title, image_dir=image_dir
    )
