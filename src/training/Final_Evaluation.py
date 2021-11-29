""" Evalute the 5 models using the test set (2019/20 regular and playoff games data).
The 5 models are:
    - A: Logistic regression model trained on distance
    - B: Logistic regression model trained on angle
    - C: Logistic regression model trained on distance and angle
    - D: Best performing XGBoost model
    - E: Best performing model
"""
import argparse
import logging
import os
import pickle
from collections import ChainMap
import pandas as pd
from comet_ml import API
import sklearn

logging.basicConfig(level=logging.INFO)

from generate_figures import generate_adv_model_figures
from preprocessing import Process_Data

from utils import (
    EXP_KWARGS,
    TRAIN_COLS_DISTANCE,
    TRAIN_COLS_ANGLE,
    TRAIN_COLS_BASELINE,
    TRAIN_COLS_PART_4,
    LABEL_COL,
    log_experiment,
    clf_performance_metrics,
)


# Model Information
MODELINFO_A = {
    "model_type": "logreg",
    "Name": "Logistic Regression Distance Only",
    "CometModelName": "logistic-regression-distance-only",
    "Version": "1.0.2",
    "FileName": "LR_distance_only",
    "Col": TRAIN_COLS_DISTANCE,
}
MODELINFO_B = {
    "model_type": "logreg",
    "Name": "Logistic Regression Angle Only",
    "CometModelName": "logistic-regression-angle-only",
    "Version": "1.0.3",
    "FileName": "LR_angle_only",
    "Col": TRAIN_COLS_ANGLE,
}
MODELINFO_C = {
    "model_type": "logreg",
    "Name": "Logistic Regression Distance and Angle",
    "CometModelName": "logistic-regression-distance-and-angle",
    "Version": "1.0.2",
    "FileName": "LR_distance_and_angle",
    "Col": TRAIN_COLS_BASELINE,
}
MODELINFO_D = {
    "model_type": "xgboost_lasso",
    "Name": "XGBoost Model with Lasso",
    "CometModelName": "xgboost-lasso",
    "Version": "1.0.1",
    "FileName": "xgboost_lasso",
    "Col": TRAIN_COLS_PART_4,
}

MODELINFO_E = {
    "model_type": "xgboost_SHAP",
    "Name": "XGBoost Model with SHAP",
    "CometModelName": "xgboost-shap",
    "Version": "1.0.1",
    "FileName": "xgboost_SHAP",
    "Col": TRAIN_COLS_PART_4,
}


MODELINFO_F = {
    "model_type": "xgboost_non_corr",
    "Name": "XGBoost with Non Correlated Features",
    "CometModelName": "xgboost-feats-non-corr",
    "Version": "1.0.1",
    "FileName": "xgboost_feats_non_corr",
    "Col": TRAIN_COLS_PART_4,
}

MODELINFO_G = {
    "model_type": "NN_MLP",
    "Name": "Neural Network - Advance Features",
    "CometModelName": "nn-adv",
    "Version": "1.0.1",
    "FileName": "NN_adv",
    "Col": TRAIN_COLS_PART_4,
}

MODELINFO_LR_ALL = {
    "model_type": "logreg_all",
    "Name": "logistic Regression with all Features in (Q4)",
    "CometModelName": "lr-all-feats",
    "Version": "1.0.0",
    "FileName": "lr_all_feats",
    "Col": TRAIN_COLS_PART_4,
}

MODELINFO_LOGREG_NON_CORR = {
    "model_type": "logreg_non_corr_feats",
    "Name": "Logistic Regression without Correlated Features",
    "CometModelName": "lr-non-corr-feats",
    "Version": "1.0.0",
    "FileName": "lr_non_corr_feats",
    "Col": TRAIN_COLS_PART_4,
}

MODELINFO_XGBOOST_SMOTE = {
    "model_type": "xgboost_SMOTE",
    "Name": "XGBoost with SMOTE Oversampling",
    "CometModelName": "xgboost-SMOTE",
    "Version": "1.0.0",
    "FileName": "xgboost_SMOTE",
    "Col": TRAIN_COLS_PART_4,
}

MODELINFO_LOGREG_SMOTE = {
    "model_type": "logreg_SMOTE",
    "Name": "Logistic Regression with SMOTE Oversampling",
    "CometModelName": "lr-SMOTE",
    "Version": "1.0.0",
    "FileName": "lr_SMOTE",
    "Col": TRAIN_COLS_PART_4,
}

MODELINFO = [
    MODELINFO_A,
    MODELINFO_B,
    MODELINFO_C,
    MODELINFO_D,
    MODELINFO_E,
    MODELINFO_F,
    MODELINFO_G,
    MODELINFO_XGBOOST_SMOTE,
    MODELINFO_LOGREG_SMOTE,
    MODELINFO_LR_ALL,
    MODELINFO_LOGREG_NON_CORR,
]


# Download registered models from Comet and load the models
def Retrieve_Comet(ModelComet):
    api = API()
    api.download_registry_model(
        EXP_KWARGS["workspace"],
        ModelComet["CometModelName"],
        ModelComet["Version"],
        output_path="./models",
        expand=True,
    )
    with open(
        os.path.join("./models/", ModelComet["FileName"] + ".pickle"), "rb"
    ) as fid:
        Model = pickle.load(fid)
    return Model


# Load data from the specific file path, select all, regular season, or post season data.
def load_dataset(Col, FilePath, Season=None):

    df = pd.read_csv(FilePath)

    if Season == "Reg":
        df = df[
            df["game_id"].astype(int).astype(str).str.contains("201902")
        ].reset_index()
    elif Season == "Post":
        df = df[
            df["game_id"].astype(int).astype(str).str.contains("201903")
        ].reset_index()
    else:
        df["game_id"] = df["game_id"].astype(int)

    X_data = df[Col]
    Y_data = df[LABEL_COL].astype(int)

    return X_data, Y_data


def main(args):

    # Retrieve and process data and model
    TestPath = "./data/processed/test_processed.csv"
    TrainPath = "./data/processed/train_processed.csv"

    for Season in ["Reg", "Post"]:

        experiment_prediction_filenames = []

        for Model_Param in MODELINFO:
            logging.info(f"{Season} season using " + Model_Param["Name"])

            clf = Retrieve_Comet(Model_Param)

            X_Test, Y_Test = load_dataset(Model_Param["Col"], TestPath, Season=Season)
            X_Train, _ = load_dataset(Model_Param["Col"], TrainPath)

            X_Processed, Y_Processed = Process_Data(
                X_Test, Y_Test, X_Train, Model_Param["model_type"]
            )

            y_pred = clf.predict(X_Processed)
            y_proba = clf.predict_proba(X_Processed)[:, 1]

            csv_path = (
                f"./models/predictions/Testset_Eval_{Season}_"
                + Model_Param["FileName"]
                + ".csv"
            )
            res = pd.DataFrame(
                {"y_true": Y_Processed, "y_preds": y_pred, "y_proba": y_proba}
            )
            res.to_csv(csv_path, index=False)

            acc = sklearn.metrics.accuracy_score(Y_Processed, y_pred)
            logging.info("Accuracy is {:6.3f}".format(acc))

            exp_pred_filename = {Model_Param["FileName"]: csv_path}
            experiment_prediction_filenames.append(exp_pred_filename)

            if args.log_results:
                perf_metrics, confusion_matrix = clf_performance_metrics(
                    Y_Processed, y_pred, y_proba, verbose=True
                )

                log_experiment(
                    {**Model_Param},
                    perf_metrics,
                    X_Processed,
                    confusion_matrix=confusion_matrix,
                    exp_name=Model_Param["FileName"]+f"testset_eval_{Season}",
                )

        experiment_prediction_filenames = dict(
            ChainMap(*experiment_prediction_filenames)
        )
        title = "Visual Summary - Models Performance Comparison" + "_" + Season
        image_dir = "./figures/Test_Evaluation/"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        generate_adv_model_figures(experiment_prediction_filenames, title, image_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Evaluation")

    parser.add_argument(
        "-c",
        "--generate-charts",
        dest="generate_charts",
        help="(boolean) if passed, generate model visuals",
        action="store_true",
    )
    parser.set_defaults(generate_charts=True)

    parser.add_argument(
        "-l",
        "--log-results",
        dest="log_results",
        help="(boolean) if passed, logs model parameters and performance metrics to Comet.ml",
        action="store_true",
    )
    parser.set_defaults(log_results=False)

    args = parser.parse_args()

    main(args)
