import argparse
import logging
import warnings
import sklearn
import pandas as pd
import numpy as np
import copy
from comet_ml import Experiment
from sklearn.linear_model import LogisticRegression
from visuals import generate_shot_classifier_charts
from utils import clf_performance_metrics, log_experiment

logging.basicConfig(level = logging.INFO)

TRAIN_COLS_BASIC = [
    'period', 'goals_home', 'goals_away',
    'shooter_id', 'coordinate_x', 'coordinate_y', 'distance_from_net',
    'angle'
]

LABEL_COL = 'is_goal'

RANDOM_STATE = 1729

EXP_PARAMS = {
    "random_state": RANDOM_STATE,
    "model_type": "logreg",
    "scaler": "standard",
}


# Load training and validation data from the pre-determined location.
def load_train_and_validation():
    train = pd.read_csv("./data/processed/train_processed.csv")
    val = pd.read_csv("./data/processed/validation_processed.csv")

    na_mask = train[TRAIN_COLS_BASIC+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask.sum()} rows (of {len(train)} total) containing nulls from train")
    train = train[~na_mask]

    na_mask = val[TRAIN_COLS_BASIC+[LABEL_COL]].isnull().any(axis=1)
    logging.info(f"dropping {na_mask.sum()} rows (of {len(val)} total) containing nulls from val")
    val = val[~na_mask]

    X_train = train[TRAIN_COLS_BASIC]
    Y_train = train[LABEL_COL].astype(int)

    X_val = val[TRAIN_COLS_BASIC]
    Y_val = val[LABEL_COL].astype(int)

    return X_train, Y_train, X_val, Y_val


# Preprocess the features using z-score scaling
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


def main(args):
    X_train, Y_train, X_val, Y_val = load_train_and_validation()
    X_train, X_val = preprocess(X_train, X_val)

    y_trues, y_preds, y_probas = [], [], []
    exp_names = ['baseline_logreg_'+sub for sub in ['distance_only', 'angle_only', 'distance_and_angle']]
    col_subsets = [['distance_from_net'], ['angle'], ['distance_from_net', 'angle']]
    
    # Exploring the 3 different sets of features as instructed
    for exp_name, subset in zip(exp_names, col_subsets):
        logging.info(f"Processing {exp_name}...")
        X_train_sub = X_train[subset].values
        X_val_sub = X_val[subset].values

        clf = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_sub, Y_train)
        y_pred = clf.predict(X_val_sub)
        y_proba = clf.predict_proba(X_val_sub)[:,1]

        y_trues.append(Y_val)
        y_preds.append(y_pred)
        y_probas.append(y_proba)

        # Generate the performance matrix for this model feature combination
        perf_metrics = clf_performance_metrics(Y_val, y_pred, y_proba, verbose=True)

        # Log results to comet if commanded
        if args.log_results:
            logging.info(f"Logging model information for {exp_name}")
            log_experiment(EXP_PARAMS, perf_metrics, X_train_sub, exp_name=exp_name)
            
            
    # Adding the random baseline        
    exp_names.append('baseline_logreg_Random_Baseline')        
    y_trues.append(Y_val)
    np.random.seed(RANDOM_STATE)
    y_proba = np.random.uniform(low=0, high=1, size=len(Y_val))
    y_probas.append(y_proba)
    y_pred = (y_proba >=0.5).astype(int)
    y_preds.append(y_pred)    
    
        
    # Generate the images if commanded           
    if args.generate_charts:
        title = "Visual Summary - Simple Logistic Regressions"
        image_dir = "./src/training/visualizations/simple_log_reg/"

        generate_shot_classifier_charts(
            y_trues, y_preds, y_probas, exp_names,
            title=title, image_dir=image_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Models')

    parser.add_argument('-c', '--generate-charts', dest="generate_charts",
                    help='(boolean) if passed, generate model visuals',
                    action='store_true')
    parser.set_defaults(generate_charts=False)

    parser.add_argument('-l', '--log-results', dest="log_results",
                    help='(boolean) if passed, logs model parameters and performance metrics to Comet.ml',
                    action='store_true')
    parser.set_defaults(log_results=False)

    args = parser.parse_args()

    main(args)
