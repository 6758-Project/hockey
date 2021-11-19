import argparse

import pandas as pd

from comet_ml import Experiment

import sklearn
from sklearn.linear_model import LogisticRegression

from visuals import generate_shot_classifier_charts

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

    na_mask = train[TRAIN_COLS_BASIC+[LABEL_COL]].isnull().any(axis=1)
    print(f"dropping {na_mask.sum()} rows (of {len(train)} total) containing nulls from train")
    train = train[~na_mask]

    na_mask = val[TRAIN_COLS_BASIC+[LABEL_COL]].isnull().any(axis=1)
    print(f"dropping {na_mask.sum()} rows (of {len(val)} total) containing nulls from val")
    val = val[~na_mask]

    X_train = train[TRAIN_COLS_BASIC].values
    Y_train = train[LABEL_COL].astype(int)

    X_val = val[TRAIN_COLS_BASIC].values
    Y_val = val[LABEL_COL].astype(int)

    return X_train, Y_train, X_val, Y_val


def analyse_model_performance(y_true, y_pred, y_proba, X_train, experiment=None, generate_charts=True):
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)

    print("Accuracy is {:6.3f}".format(acc))
    print(f"Confusion Matrix:\n {cm}")

    print("F1 score is {:6.3f}".format(f1))
    print("Precision score is {:6.3f}".format(precision))
    print("Recall score is {:6.3f}".format(recall))

    if experiment:
        params={
            "random_state": RANDOM_STATE,
            "model_type": "logreg",
            "scaler": "standard",
        }

        metrics = {
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

        experiment.log_dataset_hash(X_train)
        experiment.log_parameters(params)
        experiment.log_metrics(metrics)

    if generate_charts:
        model_id = experiment.get_name() if experiment else "baseline_log_reg"
        image_dir = "./src/training/visualizations/" + model_id+"/"
        generate_shot_classifier_charts(
            y_true, y_pred, y_proba,
            model_id=model_id, image_dir=image_dir
        )


def main(args):
    comet_exp = Experiment(
        project_name='ift6758-hockey', workspace="tim-k-lee", auto_param_logging=False
    ) if args.log_results else None

    X_train, Y_train, X_val, Y_val = load_train_and_validation()

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    Y_val_pred = clf.predict(X_val)
    Y_val_proba = clf.predict_proba(X_val)[:,1]

    analyse_model_performance(Y_val, Y_val_pred, Y_val_proba, X_train, comet_exp, args.generate_charts)




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
