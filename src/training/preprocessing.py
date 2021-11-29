""" 
Preprocessing input data based on selected model.
"""

import pandas as pd
import sklearn
from utils import (
    INFREQUENT_STOPPAGE_EVENTS,
    LABEL_COL,
)

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


def Process_Data(X_Test,Y_Test,X_Train,Model_Type):     
        
    if Model_Type == "logreg":
        X_processed, Y_processed = LG_preprocess(X_Train,X_Test,Y_Test)
        return X_processed, Y_processed
        
    if Model_Type == "xgboost":
        X_processed = XGB_preprocess(X_Test)
        return X_processed, Y_Test
        
    if Model_Type == "NN_MLP":
        X_processed = NN_preprocess(X_Test,X_Train)
        return X_processed, Y_Test
    

#Logistic Regression
def LG_preprocess(X_train, X_data, y_data):
    
    na_mask = X_train.isnull().any(axis=1)
    X_train = X_train[~na_mask]

    na_mask = X_data.isnull().any(axis=1)
    X_data = X_data[~na_mask]
    y_data = y_data[~na_mask]
    
    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_data_scaled = pd.DataFrame(
        data=scaler.transform(X_data.values),
        index=X_data.index,
        columns=X_data.columns
    )

    return X_data_scaled, y_data 


#XGBoost with SHAP
def XGB_preprocess(X_data):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])
    
    X_data = X_data[SHAP_COLS]

    return X_data


#Neural Network with Advance
def NN_preprocess(X_data,X_train):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_train["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_train["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )
    
    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])
    X_train = pd.get_dummies(X_train, ["shot", "prev_event"])
    
    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_data_scaled = pd.DataFrame(
        data=scaler.transform(X_data.values),
        index=X_data.index,
        columns=X_data.columns,
    )

    X_data_scaled = X_data_scaled.fillna(0)
    
    return X_data_scaled






















