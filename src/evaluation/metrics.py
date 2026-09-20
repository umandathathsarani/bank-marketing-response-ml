"""Model evaluation utilities for the Bank Marketing Response project."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    model_name: str = "Model",
) -> dict:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    model_name : str
        Label used in printed output.

    Returns
    -------
    dict
        Dictionary of metric name → value.
    """
    metrics = {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC":    round(average_precision_score(y_true, y_prob), 4),
    }
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:12s}: {v}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['no (0)', 'yes (1)'])}")
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    ax: plt.Axes | None = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Plot a ROC curve for a single model.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    model_name : str
        Legend label.
    ax : matplotlib Axes or None
        Axes to draw on. Creates a new figure if None.
    color : str
        Line color.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    ax: plt.Axes | None = None,
    color: str = "darkorange",
) -> plt.Axes:
    """Plot a Precision-Recall curve for a single model.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    model_name : str
        Legend label.
    ax : matplotlib Axes or None
        Axes to draw on. Creates a new figure if None.
    color : str
        Line color.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, label=f"{model_name} (AP={ap:.3f})", color=color)
    ax.axhline(np.mean(y_true), color="red", linestyle="--",
               linewidth=0.8, label=f"Random baseline ({np.mean(y_true):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return ax


def threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute Precision, Recall, and F1 across a range of decision thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    thresholds : array-like or None
        Thresholds to evaluate. Defaults to np.arange(0.05, 0.75, 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by threshold with Precision, Recall, F1,
        and Predicted Positive (%) columns.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.75, 0.05)

    rows = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rows.append({
            "Threshold":              round(t, 2),
            "Precision":              precision_score(y_true, preds, zero_division=0),
            "Recall":                 recall_score(y_true, preds, zero_division=0),
            "F1":                     f1_score(y_true, preds, zero_division=0),
            "Predicted Positive (%)": preds.mean() * 100,
        })

    return pd.DataFrame(rows).set_index("Threshold").round(4)
