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

    # Generate figures for the Logistic Regression models
    lr_exp_pred_filenames = {
        "random_classifier": "./models/predictions/random_classifier.csv",
        "LR_distance_only": "./models/predictions/LR_distance_only.csv",
        "LR_angle_only": "./models/predictions/LR_angle_only.csv",
        "LR_distance_and_angle": "./models/predictions/LR_distance_and_angle.csv",

    }
    title = "Visual Summary - Logistic Regression Baselines"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=lr_exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )


    # Generate Logistic Regression Vs XGBoost
    exp_pred_filenames = {
        "LR_distance_and_angle": "./models/predictions/LR_distance_and_angle.csv",
        "xgboost_baseline": "./models/predictions/xgboost_baseline.csv",
    }
    title = "Visual Summary - LR vs XGBoost Baselines"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )


    # Generate figures for the XGBoost models (baseline, optimal and with feature selection methods)
    xgb_exp_pred_filenames = {
        "xgboost_baseline": "./models/predictions/xgboost_baseline.csv",
        "xgboost_optimal": "./models/predictions/xgboost_optimal.csv",
        "xgboost_lasso": "./models/predictions/xgboost_lasso.csv",
        "xgboost_shap": "./models/predictions/xgboost_SHAP.csv",
        "xgboost_feats_non_corr": "./models/predictions/xgboost_feats_non_corr.csv",

    }
    title = "Visual Summary - XGBoost Models"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=xgb_exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )


    # Generate figures for the XGBoost hyperparam optimization
    xgb_exp_pred_filenames = {
        "xgboost_baseline": "./models/predictions/xgboost_baseline.csv",
        "xgboost_optimal": "./models/predictions/xgboost_optimal.csv",
    }
    title = "Visual Summary - XGBoost Hyperparams Tuning"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=xgb_exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )


    # Generate figures for the NN models
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

    # Generate figures for the best models so far
    best_exp_pred_filenames = {
    	"xgboost_lasso": "./models/predictions/xgboost_lasso.csv",
        "NN_adv": "./models/predictions/NN_adv.csv",
        "xgboost_feats_non_corr": "./models/predictions/xgboost_feats_non_corr.csv",
        "lr_all_feats": "./models/predictions/lr_all_feats.csv",
        "lr_non_corr_feats": "./models/predictions/lr_non_corr_feats.csv",
    }
    title = "Visual Summary - Best Performers"
    image_dir = "./figures/advanced_models/"
    generate_adv_model_figures(
        experiment_prediction_filenames=best_exp_pred_filenames,
        title=title,
        image_dir=image_dir,
    )

    
    ########################
    # Figures for Test set #
    ########################
    # testset_pred_filenames = {
    #     "logistic_regression_distance_only": "./models/predictions/Testset_Eval_Reg_logistic_regression_distance_only.csv",
    #     "logistic_regression_angle_only": "./models/predictions/Testset_Eval_Reg_logistic_regression_angle_only.csv",
    #     "logistic_regression_distance_and_angle": "./models/predictions/Testset_Eval_Reg_logistic_regression_distance_and_angle.csv",
    #     "xgboost_SHAP": "./models/predictions/Test_Reg_xgboost_SHAP.csv",
    #     "NN_adv": "./models/predictions/Test_Reg_NN_adv.csv",
    # }
    # title = "Visual Summary - Models Performance Comparison_Reg"
    # image_dir = "./figures/Test_Evaluation/"
    # generate_adv_model_figures(
    #     experiment_prediction_filenames=testset_pred_filenames,
    #     title=title,
    #     image_dir=image_dir,
    # )
    
    # testset_pred_filenames = {
    #     "logistic_regression_distance_only": "./models/predictions/Testset_Eval_Post_logistic_regression_distance_only.csv",
    #     "logistic_regression_angle_only": "./models/predictions/Testset_Eval_Post_logistic_regression_angle_only.csv",
    #     "logistic_regression_distance_and_angle": "./models/predictions/Testset_Eval_Post_logistic_regression_distance_and_angle.csv",
    #     "xgboost_SHAP": "./models/predictions/Test_Post_xgboost_SHAP.csv",
    #     "NN_adv": "./models/predictions/Test_Post_NN_adv.csv",
    # }
    # title = "Visual Summary - Models Performance Comparison_Post"
    # image_dir = "./figures/Test_Evaluation/"
    # generate_adv_model_figures(
    #     experiment_prediction_filenames=testset_pred_filenames,
    #     title=title,
    #     image_dir=image_dir,
    # )   
