from collections.abc import Sequence
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.calibration import CalibrationDisplay

plt.style.use('ggplot')


def generate_shot_classifier_charts(
    y_trues: List[Sequence[int]], y_preds: List[Sequence[int]], y_probas: List[Sequence[float]],
    exp_names: List[str], title='', image_dir: os.PathLike ="./"):
    """Generates four core classifier performance visualizations.

    Assumes given lists of labels, predictions, and probabilities for a set of models.

    Args:
        y_trues: a list of vectors of labels
        y_preds: a list of vectors of predictions
        y_probas: a list of vectors of model-estimated probabilities
        exp_names: a list of experiment names to be used as labels
        title: the title of the chart set (optional)
        image_dir: where the chart image should be written

    Submethods assume classifier application is NHL goal prediction.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

    M = len(exp_names)
    for i in range(M):
        
        color=next(axes[0,0]._get_lines.prop_cycler)['color']
        
        ax = roc_auc_curve(
            y_trues[i], y_probas[i], label=exp_names[i],
            include_baseline=(i==0), ax=axes[0,0], color=color
        )
        ax.set_title("ROC")

        ax = reliability_curve(
            y_trues[i], y_probas[i], label=exp_names[i],
            ax=axes[0,1], n_bins=(len(y_trues[i])//300), strategy='quantile', lw=1, color=color
        )
        ax.set_title("Reliability")

        ax = true_positive_rate_curve(
            y_trues[i], y_probas[i], label=exp_names[i], ax=axes[1,0], color=color
        )
        ax.set_title("True Positive Rate by Percentile")

        ax = positive_proportion_curve(
            y_trues[i], y_probas[i], label=exp_names[i], ax=axes[1,1], color=color
        )
        ax.set_title("Positive Proportion by Percentile")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(image_dir, f'{title}.png'), bbox_inches='tight')


def roc_auc_curve(
    y_true: Sequence[int], y_proba: Sequence[float],
    label: str = "provided", include_baseline: bool = True, ax=None, color='b'):
    """ Generates a ROC Curve for a pair of label and predicted probability vectors

    Args:
        y_true: a vector of labels
        y_proba: a vector of probabilities
        label: the identifier associated with the predictions
        include_baseline: If True, plots metrics for a baseline classifier
            which predicts 50% probability for each observation
        ax: a Matplotlib Axes object (optional)

    Returns:
        ax: current Matplotlib axes
    """
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba)
    auc = metrics.auc(fpr, tpr)

    ax = plt.gca() if ax is None else ax
    
    if include_baseline:
        baseline_proba = np.ones_like(y_proba) * .5
        fpr, tpr, thresholds = metrics.roc_curve(y_true, baseline_proba)

        ax.plot(tpr, tpr, lw=2, linestyle="--", label="baseline", color='k')     
    
    ax.plot(fpr, tpr, lw=2, label=label + f" (AUC={round(auc, 4)})", color=color)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="upper left")

    return ax


def true_positive_rate_curve(
    y_true: Sequence[int], y_proba: Sequence[float], label: str = "provided",
    ax=None, color='b'):
    """ Generates a positive rate curve for a pair of label and estimated probability vectors.

    A true positive rate curve plots the proportion of positives as a function of estimated
    probability percentiles.

    In more detail, each plotted point (p, f) is the fraction F of positives in the subset of
    Y_TRUE corresponding to the Pth percentile and below of Y_PROBA.

    f = (# positives in percentile <= p) / (# observations in percentile <= p)

    Args:
        y_true: a vector of labels
        y_proba: a vector of probabilities
        label: the identifier associated with the predictions
        ax: a Matplotlib Axes object (optional)

    Returns:
        ax: current Matplotlib axes
    """

    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df['percentile'] = df['y_proba'].rank(pct=True) * 100
    df = df.sort_values("y_proba", ascending=False)

    run_sum = df['y_true'].cumsum()
    row_num = np.arange(1, len(df)+1)
    df['positive_rate'] = run_sum / row_num

    df = df[df['percentile'] <= 97.5]  # trim extremes before plotting

    ax = plt.gca() if ax is None else ax
    ax = df.plot.line(x='percentile', y='positive_rate', label=label, ax=ax, color=color)

    ax.set_xlim([100.05, -0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Estimated Probability Percentile")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="upper left")

    return ax


def positive_proportion_curve(
    y_true: Sequence[int], y_proba: Sequence[float], label: str = "provided",
    ax=None, color='b'):
    """ Generates a positive proportion curve for a pair of label and estimated probability vectors.

    A positive distribution curve plots the proportion of population positives as a function of estimated probability percentiles.

    In more detail, each plotted point (p, f) is the fraction F of positives in the subset of
    Y_TRUE corresponding to the Pth percentile and below of Y_PROBA over the total number of positives.

    f = (# positives in percentile <= p) / all positives

    Args:
        y_true: a vector of labels
        y_proba: a vector of probabilities
        label: the identifier associated with the predictions
        ax: a Matplotlib Axes object (optional)

    Returns:
        ax: current Matplotlib axes
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df['percentile'] = df['y_proba'].rank(pct=True) * 100
    df = df.sort_values("y_proba", ascending=False)

    run_sum = df['y_true'].cumsum()
    df['positive_proportion'] = run_sum / df['y_true'].sum()

    ax = plt.gca() if ax is None else ax
    ax = df.plot.line(x='percentile', y='positive_proportion', label=label, ax=ax, color=color)

    ax.set_xlim([100.05,-0.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Estimated Probability Percentile")
    ax.set_ylabel("Cumulative Positive Proportion")
    ax.legend(loc="upper left")

    return ax


def reliability_curve(
    y_true: Sequence[int], y_proba: Sequence[float], label: str = "provided",
    ax=None, color='b', **kwargs: dict):
    """ Generates a reliability curve for a pair of label and estimated probability vectors.

    See scikit learn's CalibrationDisplay for more:
    https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html

    Args:
        y_true: a vector of labels
        y_proba: a vector of probabilities
        label: the identifier associated with the predictions
        ax: a Matplotlib Axes object (optional)
        kwargs: passed to sklearn's CalibrationDisplay

    Returns:
        ax: current Matplotlib axes
    """
    ax = plt.gca() if ax is None else ax
    cd = CalibrationDisplay.from_predictions(y_true, y_proba, name=label, ax=ax, **kwargs, color=color)
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="upper left")
    
    return cd.ax_
