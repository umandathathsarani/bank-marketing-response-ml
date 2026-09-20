"""Prediction utilities for the Bank Marketing Response project."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.train import DEFAULT_THRESHOLD, load_pipeline


def predict_proba(
    X: pd.DataFrame,
    pipeline: Pipeline | None = None,
) -> np.ndarray:
    """Return the predicted probability of subscription (positive class).

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (same schema as training data, after
        ``basic_cleaning`` and ``make_features``).
    pipeline : Pipeline or None
        Fitted sklearn pipeline. If None, loads the default saved model.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Predicted probabilities for the positive class (y = 1).
    """
    if pipeline is None:
        pipeline = load_pipeline()
    return pipeline.predict_proba(X)[:, 1]


def predict(
    X: pd.DataFrame,
    pipeline: Pipeline | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Return binary predictions using a tuned decision threshold.

    The default threshold (0.30) was selected in notebook 05 to maximise
    F1 on the validation set. It provides a better Recall / Precision
    trade-off than the standard 0.50 threshold for this imbalanced dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (same schema as training data).
    pipeline : Pipeline or None
        Fitted sklearn pipeline. If None, loads the default saved model.
    threshold : float
        Decision threshold (default 0.30).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Binary predictions (0 = likely no subscription, 1 = likely subscriber).
    """
    proba = predict_proba(X, pipeline)
    return (proba >= threshold).astype(int)


def score_customers(
    X: pd.DataFrame,
    pipeline: Pipeline | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Score a customer DataFrame and return results sorted by priority.

    Returns a DataFrame with the original index, predicted probability,
    predicted class, and a human-readable priority label.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame indexed by customer / record identifier.
    pipeline : Pipeline or None
        Fitted sklearn pipeline. If None, loads the default saved model.
    threshold : float
        Decision threshold for the binary class (default 0.30).

    Returns
    -------
    pd.DataFrame
        Scored DataFrame sorted by ``subscription_probability`` descending.
        Columns: ``subscription_probability``, ``predicted_class``,
        ``priority``.
    """
    proba  = predict_proba(X, pipeline)
    preds  = (proba >= threshold).astype(int)
    result = pd.DataFrame(
        {
            "subscription_probability": proba,
            "predicted_class": preds,
            "priority": ["High" if p == 1 else "Low" for p in preds],
        },
        index=X.index,
    )
    return result.sort_values("subscription_probability", ascending=False)
