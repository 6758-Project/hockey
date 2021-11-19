from collections.abc import Sequence
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics


def generate_shot_classifier_charts(
    y_true: Sequence[int], y_pred: Sequence[float], y_proba: Sequence[float],
    model_id: str, image_dir: os.PathLike ="./"):
    """Generates four core classifier performance visualizations given labels, predictions, and probabilities.

    Submethods assume classifier application is NHL goal prediction.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    fig, ax = roc_auc_curve(y_true, y_proba, label=model_id)
    fig.savefig(os.path.join(image_dir, f'{model_id}_roc_curve.png'), bbox_inches='tight')
    plt.close()

    fig, ax = true_positive_rate_curve(y_true, y_proba, label=model_id)
    fig.savefig(os.path.join(image_dir, f'{model_id}_positive_rate.png'), bbox_inches='tight')
    plt.close()



def roc_auc_curve(
    y_true: Sequence[int], y_proba: Sequence[float],
    label: str = "provided", include_baseline: bool = True):
    """ Generates a ROC Curve for a pair of label and predicted probability vectors

    Args:
        y_true: a vector of labels
        y_proba: a vector of probabilities
        label: the identifier associated with the predictions
        include_baseline: If True, plots metrics for a baseline classifier
            which predicts 50% probability for each observation

    Returns:
        fig: current Matplotlib figure
        ax: current Matplotlib axes
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba)
    auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=label)

    if include_baseline:
        baseline_proba = np.ones_like(y_proba) * .5
        fpr, tpr, thresholds = metrics.roc_curve(y_true, baseline_proba)

        plt.plot(tpr, tpr, lw=2, linestyle="--", label="baseline")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve: AUC={round(auc, 4)}")
    plt.legend(loc="lower right")

    return plt.gcf(), plt.gca()


def true_positive_rate_curve(
    y_true: Sequence[int], y_proba: Sequence[float], label: str = "provided"):
    """ Generates a positive rate curve for a pair of label and estimated probability vectors.

    A true positive rate curve plots the proportion of positives as a function of estimated
    probability percentiles.

    In more detail, each plotted point (p, f) is the fraction F of positives in the subset of
    Y_TRUE corresponding to the Pth percentile and below of Y_PROBA.

    Args:
        y_true: a vector of labels
        y_proba: a vector of probabilities
        label: the identifier associated with the predictions
    Returns:
        fig: current Matplotlib figure
        ax: current Matplotlib axes
    """

    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=True)
    df['percentile'] = df['y_proba'].rank(pct=True)


    run_sum = df['y_true'].cumsum()
    row_num = np.arange(1, len(df)+1)
    df['positive_rate'] = run_sum / row_num

    ax = df.plot.line(x='percentile', y='positive_rate', label=label)
    fig = plt.gcf()

    plt.xlim([0.0, 1.0])
    plt.xlabel("Estimated Probability Percentile")
    plt.ylabel("True Positive Rate")
    plt.title(f"TPR by Estimated Probability Percentile")
    plt.legend(loc="lower right")

    return fig, ax


def positive_proportion_curve(y_true, y_proba):
    pass


def reliability_curve(y_true, y_proba):
    pass
