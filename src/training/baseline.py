import pandas as pd

from comet_ml import Experiment

import sklearn
from sklearn.linear_model import LogisticRegression

TRAIN_COLS_BASIC = [
    'period', 'goals_home', 'goals_away',
    'shooter_id', 'coordinate_x', 'coordinate_y', 'distance_from_net',
    'angle'
]

LABEL_COL = 'is_goal'

RANDOM_STATE = 1729


if __name__ == "__main__":
    exp = Experiment(project_name='ift-6758-milestone-2', auto_param_logging=False)

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

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    val_preds = clf.predict(X_val)
    acc = sklearn.metrics.accuracy_score(Y_val, val_preds)
    cm = sklearn.metrics.confusion_matrix(Y_val, val_preds)
    f1 = sklearn.metrics.f1_score(Y_val, val_preds)
    precision = sklearn.metrics.precision_score(Y_val, val_preds)
    recall = sklearn.metrics.recall_score(Y_val, val_preds)

    print("Accuracy is {:6.3f}".format(acc))
    print(f"Confusion Matrix:\n {cm}")

    print("F1 score is {:6.3f}".format(f1))
    print("Precision score is {:6.3f}".format(precision))
    print("Recall score is {:6.3f}".format(recall))

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

    exp.log_dataset_hash(X_train)
    exp.log_parameters(params)
    exp.log_metrics(metrics)



