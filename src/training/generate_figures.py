""" A utility for creating model quality charts from predictions

Usage:
Modify the generate_adv_model_figures function directly with:
 1. the location of the prediction filenames, and
 2. the model name to associate with the predictions

"""

import pandas as pd

from visuals import generate_shot_classifier_charts


def generate_adv_model_figures(
    experiment_prediction_filenames=None, title="Default", image_dir="./"
):

    exp_preds = {
        exp: pd.read_csv(fname)
        for exp, fname in experiment_prediction_filenames.items()
    }

    exp_names, y_trues, y_preds, y_probas = [], [], [], []
    for exp, preds in exp_preds.items():
        exp_names.append(exp)
        y_trues.append(preds["y_true"].values)
        y_preds.append(preds["y_preds"].values)
        y_probas.append(preds["y_proba"].values)

    generate_shot_classifier_charts(
        y_trues, y_preds, y_probas, exp_names, title=title, image_dir=image_dir
    )


if __name__ == "__main__":

    xgb_exp_pred_filenames = {
        "xgboost_baseline": "./models/predictions/xgboost_baseline.csv",
        "xgboost_optimal": "./models/predictions/xgboost_optimal.csv",
        "xgboost_lasso": "./models/predictions/xgboost_lasso.csv",
        "xgboost_shap": "./models/predictions/xgboost_SHAP.csv",
    }
    title = "Visual Summary - XGBoost Models"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=xgb_exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )

    nn_exp_pred_filenames = {
        "NN_distance": "./models/predictions/NN_distance.csv",
        "NN_baseline": "./models/predictions/NN_baseline.csv",
        "NN_basic": "./models/predictions/NN_basic.csv",
        "NN_adv": "./models/predictions/NN_adv.csv",
    }
    title = "Visual Summary - Neural Network Models"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=nn_exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )

    # best models so far from XGBoost and NN models
    best_exp_pred_filenames = {
    	"xgboost_lasso": "./models/predictions/xgboost_lasso.csv",
        "NN_adv": "./models/predictions/NN_adv.csv",
    }
    title = "Visual Summary - Best Performers"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=best_exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )