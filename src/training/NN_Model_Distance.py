""" Trains and saves a multi-layer perceptron

Resources used:
    https://scikit-learn.org/stable/modules/neural_networks_supervised.html

"""

import comet_ml
from sklearn.neural_network import MLPClassifier
import sklearn 
from sklearn import metrics
import logging
import pandas as pd
import matplotlib.pyplot as plt
from generate_figures_mod import generate_adv_model_figures 


from utils import (
    EXP_KWARGS,
    clf_performance_metrics, log_experiment, register_model,
)

TRAIN_COLS_BASELINE = [ 'distance_from_net']

EXP_PARAMS = {
    "model_type": "Multi-layer Perceptron",
}

TRAIN_COLS_BASIC = [
    'period', 'goals_home', 'goals_away',
    'shooter_id', 'coordinate_x', 'coordinate_y', 'distance_from_net',
    'angle'
]

LABEL_COL = 'is_goal'

RANDOM_STATE = 1729


def load_train_and_validation():
    
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    na_mask = train[TRAIN_COLS_BASELINE+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask.sum()} rows (of {len(train)} total) containing nulls from train")
    train = train[~na_mask]

    na_mask = val[TRAIN_COLS_BASELINE+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask.sum()} rows (of {len(val)} total) containing nulls from val")
    val = val[~na_mask]

    X_train = train[TRAIN_COLS_BASELINE]
    Y_train = train[LABEL_COL].astype(int)

    X_val = val[TRAIN_COLS_BASELINE]
    Y_val = val[LABEL_COL].astype(int) 
        
    return X_train, Y_train, X_val, Y_val


def preprocess(X_train, X_val):
    scaler = sklearn.preprocessing.StandardScaler().fit(X_train.values)

    X_train_scaled = pd.DataFrame(
        data=scaler.transform(X_train.values),
        index=X_train.index,
        columns=X_train.columns
    )

    X_val_scaled = pd.DataFrame(
        data=scaler.transform(X_val.values),
        index=X_val.index,
        columns=X_val.columns
    )

    return X_train_scaled, X_val_scaled




if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = load_train_and_validation()
    X_train_scaled, X_val_scaled = preprocess(X_train, X_val)

    input_layer = X_train_scaled.shape[1]+1
    output_layer = 2
    hidden_layer = (input_layer+output_layer)//2
    
    params = {
        'solver':'adam', 'hidden_layer_sizes': (input_layer,2*hidden_layer, output_layer), 'activation': "logistic", 'learning_rate': "adaptive",
        'random_state':RANDOM_STATE, 'verbose': True, 'validation_fraction': 0.1
    }

    clf = MLPClassifier(**params)
    clf.fit(X_train_scaled,Y_train)

    y_pred = clf.predict(X_val_scaled)
    y_proba = clf.predict_proba(X_val_scaled)[:,1]


    res = pd.DataFrame({
        'y_true': Y_val,
        'y_preds': y_pred,
        'y_proba': y_proba
    })
    exp_name = "NN_distance"
    res.to_csv(f"./models/predictions/{exp_name}.csv", index=False)
    
    title = "Visual Summary - NN Distance"
    image_dir = "./figures/exploration/"
    experiment_prediction_filenames = {
        "NN_distance": "./models/predictions/NN_distance.csv",
    }
    generate_adv_model_figures(experiment_prediction_filenames,title,image_dir)
    
    
    perf_metrics = clf_performance_metrics(Y_val, y_pred, y_proba, verbose=True)
    
    #comet_exp = log_experiment(EXP_PARAMS, perf_metrics, X_train_scaled, 'NN_distance')
    #register_model(clf, comet_exp, "./models/NN_distance.pickle")






