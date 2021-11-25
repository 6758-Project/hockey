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


from visuals import generate_shot_classifier_charts
from utils import (
    clf_performance_metrics, EXP_KWARGS,
    TRAIN_COLS_BASIC, TRAIN_COLS_ADV, LABEL_COL, RANDOM_STATE
)


# Model Information
MODELINFO_A = {
    "Name": "Logistic Regression Distance Only", 
    "CometModelName": "logistic-regression-distance-only",
    "Version": "1.0.1",
    "FileName": "logistic_regression_distance_only.pickle"
}
MODELINFO_B = {
    "Name": "Logistic Regression Angle Only", 
    "CometModelName": "logistic-regression-angle-only",
    "Version": "1.0.1",
    "FileName": "logistic_regression_angle_only.pickle"
}
MODELINFO_C = {
    "Name": "Logistic Regression Distance and Angle", 
    "CometModelName": "logistic-regression-distance-and-angle",
    "Version": "1.0.1",
    "FileName": "logistic_regression_distance_and_angle.pickle"
}
MODELINFO_D = {
    "Name": "XGBoost Model", 
    "CometModelName": "logistic-regression-distance-only", #Need Update
    "Version": "1.0.1", #Need Update
    "FileName": "logistic_regression_distance_only.pickle" #Need Update
}
MODELINFO_E = {
    "Name": "Best Performing Model", 
    "CometModelName": "logistic-regression-distance-only", #Need Update
    "Version": "1.0.1", #Need Update
    "FileName": "logistic_regression_distance_only.pickle" #Need Update
}








# Download registered models from Comet and load the models
def Retrieve_Comet():
    api = API()
   
    api.download_registry_model(EXP_KWARGS["workspace"], MODELINFO_A["CometModelName"], MODELINFO_A["Version"],
                                output_path="./models", expand=True)
    api.download_registry_model(EXP_KWARGS["workspace"], MODELINFO_B["CometModelName"], MODELINFO_B["Version"],
                                output_path="./models", expand=True)
    api.download_registry_model(EXP_KWARGS["workspace"], MODELINFO_C["CometModelName"], MODELINFO_C["Version"],
                                output_path="./models", expand=True)
    api.download_registry_model(EXP_KWARGS["workspace"], MODELINFO_D["CometModelName"], MODELINFO_D["Version"],
                                output_path="./models", expand=True)
    api.download_registry_model(EXP_KWARGS["workspace"], MODELINFO_E["CometModelName"], MODELINFO_E["Version"],
                                output_path="./models", expand=True)

    with open(os.path.join("./models/",MODELINFO_A["FileName"]), 'rb') as fid:
        Model_A = pickle.load(fid)
    with open(os.path.join("./models/",MODELINFO_B["FileName"]), 'rb') as fid:
        Model_B = pickle.load(fid)    
    with open(os.path.join("./models/",MODELINFO_C["FileName"]), 'rb') as fid:
        Model_C = pickle.load(fid)  
    with open(os.path.join("./models/",MODELINFO_D["FileName"]), 'rb') as fid:
        Model_D = pickle.load(fid)  
    with open(os.path.join("./models/",MODELINFO_E["FileName"]), 'rb') as fid:
        Model_E = pickle.load(fid)  
    
    return Model_A, Model_B, Model_C, Model_D, Model_E
 

    
# Load data from the specific file path, select all, regular season, or post season data.
def load_dataset(FilePath,Season = None):
    df = pd.read_csv(FilePath)
    
    if Season == "Reg":
        df = df[df["game_id"].astype(int).astype(str).str.contains("201902")]
    elif Season == "Post":
        df = df[df["game_id"].astype(int).astype(str).str.contains("201903")]
    df["game_id"] = df["game_id"].astype(int)
    
    na_mask_basic = df[TRAIN_COLS_BASIC+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask_basic.sum()} rows (of {len(df)} total) containing nulls from train")
    df_basic = df[~na_mask_basic]
        
    X_basic = df_basic[TRAIN_COLS_BASIC]
    Y_basic = df_basic[LABEL_COL].astype(int)
    
    na_mask_adv = df[TRAIN_COLS_ADV+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask_adv.sum()} rows (of {len(df)} total) containing nulls from train")
    df_adv = df[~na_mask_adv]
    
    X_adv = df_adv[TRAIN_COLS_ADV]
    Y_adv = df_adv[LABEL_COL].astype(int)   
    
    return X_basic, Y_basic, X_adv, Y_adv       
    

def preprocess(X_train, X_data):
    scaler = StandardScaler().fit(X_train.values)

    X_data_scaled = pd.DataFrame(
        data=scaler.transform(X_data.values),
        index=X_data.index,
        columns=X_data.columns
    )

    return X_data_scaled    
    

def Pred_Plot_Gen(Models,X_Test_b,Y_Test_b,X_Test_a,Y_Test_a,title):
    y_preds_A = Models[0].predict(X_Test_b['distance_from_net'].values.reshape(-1, 1))
    y_preds_B = Models[1].predict(X_Test_b['angle'].values.reshape(-1, 1))
    y_preds_C = Models[2].predict(X_Test_b[['distance_from_net', 'angle']].values.reshape(-1, 2))
    y_preds_D = y_preds_A
    y_preds_E = y_preds_A
    
    y_proba_A = Models[0].predict_proba(X_Test_b['distance_from_net'].values.reshape(-1, 1))[:,1]
    y_proba_B = Models[1].predict_proba(X_Test_b['angle'].values.reshape(-1, 1))[:,1]
    y_proba_C = Models[2].predict_proba(X_Test_b[['distance_from_net', 'angle']].values.reshape(-1, 2))[:,1]
    y_proba_D = y_proba_A
    y_proba_E = y_proba_A
    
    y_trues = [Y_Test_b,Y_Test_b,Y_Test_b,Y_Test_b,Y_Test_b]
    y_preds = [y_preds_A,y_preds_B,y_preds_C,y_preds_D,y_preds_E]
    y_probas = [y_proba_A,y_proba_B,y_proba_C,y_proba_D,y_proba_E]
    exp_names = [MODELINFO_A["Name"],MODELINFO_B["Name"],MODELINFO_C["Name"],MODELINFO_D["Name"],MODELINFO_E["Name"]]
    
    for i in range(5):
        perf_metrics = clf_performance_metrics(y_trues[i], y_preds[i], y_probas[i], verbose=True)
    
    if args.generate_charts:
        image_dir = "./figures/final_eval/"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        generate_shot_classifier_charts(
            y_trues, y_preds, y_probas, exp_names,
            title=title, image_dir=image_dir
        )       
   
    
    
    
    
    
def main(args):
    
    
    # Retrieve and process data and model
    TestPath = "./data/processed/test_processed.csv"
    TrainPath = "./data/processed/train_processed.csv"
    
    X_Test_basic_r, Y_Test_basic_r, X_Test_adv_r, Y_Test_adv_r = load_dataset(TestPath, Season="Reg")
    X_Test_basic_p, Y_Test_basic_p, X_Test_adv_p, Y_Test_adv_p = load_dataset(TestPath, Season="Post")
    X_Train_basic, Y_Train_basic, X_Train_adv, Y_Train_adv = load_dataset(TrainPath)
    
    X_Test_basic_r_scaled = preprocess(X_Train_basic, X_Test_basic_r)
    X_Test_basic_p_scaled = preprocess(X_Train_basic, X_Test_basic_p)
    # Uncertain whether scaling is needed
    #X_Test_adv_r_scaled = preprocess(X_Train_adv, X_Test_adv_r)
    #X_Test_adv_p_scaled = preprocess(X_Train_adv, X_Test_adv_p)
    
    Model_A, Model_B, Model_C, Model_D, Model_E = Retrieve_Comet()
    Models = [Model_A, Model_B, Model_C, Model_D, Model_E]

    # Generate prediction and plot for regular season
    title = "Visual Summary - Final Evaluate (Regular Season)"
    Pred_Plot_Gen(Models,X_Test_basic_r,Y_Test_basic_r,X_Test_adv_r,Y_Test_adv_r,title)
           
    # Generate prediction and plot for post season
    title = "Visual Summary - Final Evaluate (Post Season)"
    Pred_Plot_Gen(Models,X_Test_basic_p,Y_Test_basic_p,X_Test_adv_p,Y_Test_adv_p,title)    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Final Evaluation')

    parser.add_argument('-c', '--generate-charts', dest="generate_charts",
                    help='(boolean) if passed, generate model visuals',
                    action='store_true')
    parser.set_defaults(generate_charts=False)

    args = parser.parse_args()

    main(args)    
    
    
    
    
