""" 
Trains and saves 4 multi-layer perceptron models
- Trained on distance
- Trained on distance and angle
- Trained on basic features
- Trained on advance features

Resources used:
    https://scikit-learn.org/stable/modules/neural_networks_supervised.html
"""

import comet_ml
import argparse
from sklearn.neural_network import MLPClassifier
import sklearn 
from sklearn import metrics
import logging
import pandas as pd
import matplotlib.pyplot as plt
from generate_figures_mod import generate_adv_model_figures 
from collections import ChainMap

from utils import (
    EXP_KWARGS, INFREQUENT_STOPPAGE_EVENTS, TRAIN_COLS_PART_4, LABEL_COL,TRAIN_COLS_BASIC,
    clf_performance_metrics, log_experiment, register_model, RANDOM_STATE
)

TRAIN_COLS_DISTANCE = [ 'distance_from_net']
TRAIN_COLS_BASELINE = [ 'distance_from_net','angle']

EXP_PARAMS = {
    "model_type": "Multi-layer Perceptron",
}

MODEL_TYPE = None



def load_train_and_validation(Col):
    
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    X_train = train[Col]
    Y_train = train[LABEL_COL].astype(int)

    X_val = val[Col]
    Y_val = val[LABEL_COL].astype(int) 
    
    return X_train, Y_train, X_val, Y_val


def preprocess(X_train, X_val, Col):
    
    if Col == TRAIN_COLS_PART_4:
        X_train['secondary_type'].replace({'Tip-in': 'Deflection'}, inplace=True)
        X_train['prev_event_type'].replace(
            to_replace=INFREQUENT_STOPPAGE_EVENTS, value = 'STOP', inplace=True
        )

        X_val['secondary_type'].replace({'Tip-in': 'Deflection'}, inplace=True)
        X_val['prev_event_type'].replace(
            to_replace=INFREQUENT_STOPPAGE_EVENTS, value = 'STOP', inplace=True
        )

        X_train = pd.get_dummies(X_train, ['shot', 'prev_event'])
        X_val = pd.get_dummies(X_val,['shot', 'prev_event'])    
    

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







def main(args):
   
    experiment_prediction_filenames = []
    
    for Model in args.Model_Type:
    
        print(Model)
        
        if (Model != "Distance") and (Model != "Baseline") and (Model != "Basic") and (Model != "Advance"):
            logging.info("Input column selection incorrect")
            return None
            
        
        if Model == "Distance":
            exp_name = "NN_distance"
            title = "Visual Summary - NN Distance"
            exp_pred_filename = {"NN_distance": "./models/predictions/NN_distance.csv"}            
            Col = ['distance_from_net']
        
        if Model == "Baseline":
            exp_name = "NN_baseline"
            title = "Visual Summary - NN Baseline"
            exp_pred_filename = {"NN_baseline": "./models/predictions/NN_baseline.csv"}
            Col = ['distance_from_net','angle']
            
        if Model == "Basic":
            exp_name = "NN_basic"
            title = "Visual Summary - NN Basic"
            exp_pred_filename = {"NN_basic": "./models/predictions/NN_basic.csv"}
            Col = TRAIN_COLS_BASIC
            
        if Model == "Advance":
            exp_name = "NN_adv"
            title = "Visual Summary - NN Advance"
            exp_pred_filename = {"NN_adv": "./models/predictions/NN_adv.csv"}                 
            Col = TRAIN_COLS_PART_4
            
    
        X_train, Y_train, X_val, Y_val = load_train_and_validation(Col)
        X_train_scaled, X_val_scaled = preprocess(X_train, X_val, Col)

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
        
        res.to_csv(f"./models/predictions/{exp_name}.csv", index=False)


        perf_metrics = clf_performance_metrics(Y_val, y_pred, y_proba, verbose=True)  

        experiment_prediction_filenames.append(exp_pred_filename)
        
        if args.generate_charts:    
            image_dir = "./figures/exploration/"
            generate_adv_model_figures(exp_pred_filename,title,image_dir)

        if args.log_results:  
            comet_exp = log_experiment(EXP_PARAMS, perf_metrics, X_train_scaled, 'NN_distance')
            if args.register_models:  
                pickle_path = f"./models/{exp_name}.pickle"
                register_model(clf, comet_exp, pickle_path)
    
    
    experiment_prediction_filenames = dict(ChainMap(*experiment_prediction_filenames))
    title = "Visual Summary - Neural Network Models Comparison"
    image_dir = "./figures/exploration/"
    generate_adv_model_figures(experiment_prediction_filenames,title,image_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network Models')

    parser.add_argument('-c', '--generate-charts', dest="generate_charts",
                    help='(boolean) if passed, generate model visuals',
                    action='store_true')
    parser.set_defaults(generate_charts=False)

    parser.add_argument('-l', '--log-results', dest="log_results",
                    help='(boolean) if passed, logs model parameters and performance metrics to Comet.ml',
                    action='store_true')
    parser.set_defaults(log_results=False)

    parser.add_argument('-s', '--register-models', dest="register_models",
                    help="(boolean) if passed, upload model to registry",
                    action='store_true')
    parser.set_defaults(register_models=False)

    parser.add_argument('-m', '--column-keywords', dest="Model_Type" ,type=str, nargs='+')
    
    args = parser.parse_args()

    if not args.log_results and args.register_models:
        raise ValueError("Cannot register model if results are not logged")

    main(args)







