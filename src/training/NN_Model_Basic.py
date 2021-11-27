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
    EXP_KWARGS,TRAIN_COLS_BASIC,TRAIN_COLS_BASIC,LABEL_COL,RANDOM_STATE,
    clf_performance_metrics, log_experiment, register_model
)

EXP_PARAMS = {
    "model_type": "Multi-layer Perceptron",
}



def load_train_and_validation():
    
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    X_train = train[TRAIN_COLS_BASIC]
    Y_train = train[LABEL_COL].astype(int)

    X_val = val[TRAIN_COLS_BASIC]
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

    X_train_scaled=X_train_scaled.fillna(0)
    X_val_scaled=X_val_scaled.fillna(0)
    
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
    exp_name = "NN_basic"
    res.to_csv(f"./models/predictions/{exp_name}.csv", index=False)
    
    title = "Visual Summary - NN Basic"
    image_dir = "./figures/exploration/"
    experiment_prediction_filenames = {
        "NN_basic": "./models/predictions/NN_basic.csv",
    }
    generate_adv_model_figures(experiment_prediction_filenames,title,image_dir)
    
    
    perf_metrics = clf_performance_metrics(Y_val, y_pred, y_proba, verbose=True)
    
    #comet_exp = log_experiment(EXP_PARAMS, perf_metrics, X_train_scaled, 'NN_basic')
    #register_model(clf, comet_exp, "./models/NN_basic.pickle")






