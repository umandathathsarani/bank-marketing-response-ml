"""Feature engineering for the Bank Marketing Response project."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Feature group definitions used by the preprocessing pipeline ──────────

NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day_of_week",
    "campaign",
    "campaign_capped",
    "pdays",
    "previous",
    "was_previously_contacted",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "age_group",
]

CAMPAIGN_CONTACT_PERCENTILE = 0.95  # cap level for campaign_capped


def add_was_previously_contacted(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary flag indicating prior campaign contact.

    ``pdays = -1`` is the sentinel value meaning the client was never
    contacted before the current campaign. This flag makes that implicit
    encoding explicit and easier for linear models to use.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ``pdays``.

    Returns
    -------
    pd.DataFrame
        Dataset with added ``was_previously_contacted`` (0 or 1).
    """
    df = df.copy()
    df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin age into interpretable ordinal groups.

    Groups: ``<30``, ``30-40``, ``40-50``, ``50-60``, ``60+``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ``age``.

    Returns
    -------
    pd.DataFrame
        Dataset with added ``age_group`` (string category).
    """
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"],
    ).astype(str)
    return df


def add_campaign_capped(
    df: pd.DataFrame,
    cap: int | None = None,
    percentile: float = CAMPAIGN_CONTACT_PERCENTILE,
) -> pd.DataFrame:
    """Cap extreme campaign contact counts to reduce outlier influence.

    The ``campaign`` variable contains heavy outliers (some clients were
    contacted 60+ times). This feature caps those values at the 95th
    percentile of the training distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing ``campaign``.
    cap : int or None
        Upper cap value. If None, computed from ``percentile``.
    percentile : float
        Percentile of ``campaign`` to use as the cap (default 0.95).

    Returns
    -------
    pd.DataFrame
        Dataset with added ``campaign_capped``.
    """
    df = df.copy()
    if cap is None:
        cap = int(df["campaign"].quantile(percentile))
    df["campaign_capped"] = df["campaign"].clip(upper=cap)
    return df


def make_features(df: pd.DataFrame, campaign_cap: int | None = None) -> pd.DataFrame:
    """Apply all feature engineering steps in a single call.

    Steps applied:
    1. ``was_previously_contacted`` — binary flag from ``pdays``.
    2. ``age_group`` — ordinal age bins.
    3. ``campaign_capped`` — campaign contacts capped at 95th percentile.

    All engineered features are:
    - Available at the intended prediction point (before campaign contact).
    - Non-leaky — derived only from pre-contact information.
    - Reproducible — deterministic transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (after ``basic_cleaning``).
    campaign_cap : int or None
        Explicit cap for campaign contacts. Computed from data if None.

    Returns
    -------
    pd.DataFrame
        Dataset with all engineered features added.
    """
    df = add_was_previously_contacted(df)
    df = add_age_group(df)
    df = add_campaign_capped(df, cap=campaign_cap)
    return df
