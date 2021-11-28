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
import pandas as pd
import numpy as np
from comet_ml import API
import sklearn
from sklearn.preprocessing import StandardScaler
logging.basicConfig(level = logging.INFO)

from generate_figures import generate_adv_model_figures 
from visuals import generate_shot_classifier_charts
from preprocessing import Process_Data
from collections import ChainMap

from utils import (
    EXP_KWARGS,
    INFREQUENT_STOPPAGE_EVENTS,
    TRAIN_COLS_PART_4,
    LABEL_COL,
    clf_performance_metrics,
    log_experiment,
    register_model,
    RANDOM_STATE,
)

TRAIN_COLS_DISTANCE = ["distance_from_net"]
TRAIN_COLS_ANGLE = ["angle"]
TRAIN_COLS_BASELINE = ["distance_from_net", "angle"]



# Model Information
MODELINFO_A = {
    "model_type": "logreg",
    "Name": "Logistic Regression Distance Only", 
    "CometModelName": "logistic-regression-distance-only",
    "Version": "1.0.1",
    "FileName": "logistic_regression_distance_only",
    "Col": TRAIN_COLS_DISTANCE
}
MODELINFO_B = {
    "model_type": "logreg",
    "Name": "Logistic Regression Angle Only", 
    "CometModelName": "logistic-regression-angle-only",
    "Version": "1.0.1",
    "FileName": "logistic_regression_angle_only",
    "Col": TRAIN_COLS_ANGLE
}
MODELINFO_C = {
    "model_type": "logreg",
    "Name": "Logistic Regression Distance and Angle", 
    "CometModelName": "logistic-regression-distance-and-angle",
    "Version": "1.0.1",
    "FileName": "logistic_regression_distance_and_angle",
    "Col": TRAIN_COLS_BASELINE
}
MODELINFO_D = {
    "model_type": "xgboost",
    "Name": "XGBoost Model with SHAP", 
    "CometModelName": "xgboost-shap", 
    "Version": "1.0.0",
    "FileName": "xgboost_SHAP" ,
    "Col": TRAIN_COLS_PART_4
}
MODELINFO_E = {
    "model_type": "NN_MLP",
    "Name": "Neural Network - Advance Features", 
    "CometModelName": "nn-adv",
    "Version": "1.0.0", 
    "FileName": "NN_adv",
    "Col": TRAIN_COLS_PART_4
}

MODELINFO = [MODELINFO_A,MODELINFO_B,MODELINFO_C,MODELINFO_D,MODELINFO_E] 


# Download registered models from Comet and load the models
def Retrieve_Comet(ModelComet):
    api = API()
    api.download_registry_model(EXP_KWARGS["workspace"], ModelComet["CometModelName"], ModelComet["Version"],
                                output_path="./models", expand=True)    
    with open(os.path.join("./models/",ModelComet["FileName"]+".pickle"), 'rb') as fid:
        Model = pickle.load(fid)
    return Model

    
# Load data from the specific file path, select all, regular season, or post season data.
def load_dataset(Col,FilePath,Season = None):

    df = pd.read_csv(FilePath)
    
    if Season == "Reg":
        df = df[df["game_id"].astype(int).astype(str).str.contains("201902")]
    elif Season == "Post":
        df = df[df["game_id"].astype(int).astype(str).str.contains("201903")]
    else:
        df["game_id"] = df["game_id"].astype(int)    
    
    X_data = df[Col]
    Y_data = df[LABEL_COL].astype(int)

    return X_data, Y_data
     

    
def main(args):
    
    # Retrieve and process data and model
    TestPath = "./data/processed/test_processed.csv"
    TrainPath = "./data/processed/train_processed.csv"

    for Season in ["Reg","Post"]:
        
        experiment_prediction_filenames = []
        
        for Model_Param in MODELINFO:
            logging.info(f"{Season} season using " + Model_Param["Name"])
            
            clf = Retrieve_Comet(Model_Param)

            X_Test, Y_Test = load_dataset(Model_Param["Col"],TestPath,Season = Season)
            X_Train, Y_Train = load_dataset(Model_Param["Col"],TrainPath)
            
            X_Processed, Y_Processed = Process_Data(X_Test,Y_Test,X_Train,Model_Param["model_type"])

            y_pred = clf.predict(X_Processed)
            y_proba = clf.predict_proba(X_Processed)[:, 1]
            
            csv_path = f"./models/predictions/Testset_Eval_{Season}_"+Model_Param["FileName"]+".csv"
            res = pd.DataFrame({"y_true": Y_Processed, "y_preds": y_pred, "y_proba": y_proba})
            res.to_csv(csv_path, index=False)

            perf_metrics = clf_performance_metrics(Y_Processed, y_pred, y_proba, verbose=True)

            exp_pred_filename = {Model_Param["FileName"]: csv_path}
            experiment_prediction_filenames.append(exp_pred_filename)

            if args.generate_charts:
                image_dir = "./figures/Test_Evaluation/"
                generate_adv_model_figures(exp_pred_filename, Model_Param["Name"]+ "_" + Season , image_dir)

        experiment_prediction_filenames = dict(ChainMap(*experiment_prediction_filenames))
        title = "Visual Summary - Models Performance Comparison" + "_" + Season 
        image_dir = "./figures/Test_Evaluation/"
        generate_adv_model_figures(experiment_prediction_filenames, title, image_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Final Evaluation')

    parser.add_argument('-c', '--generate-charts', dest="generate_charts",
                    help='(boolean) if passed, generate model visuals',
                    action='store_true')
    parser.set_defaults(generate_charts=False)   
    
    args = parser.parse_args()

    main(args)    
    
    
    
    
