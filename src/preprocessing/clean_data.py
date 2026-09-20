"""Data cleaning utilities for the Bank Marketing Response project."""
from __future__ import annotations

import pandas as pd


# Features excluded due to target leakage
LEAKAGE_FEATURES = ["duration"]


def drop_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features that would not be available at prediction time.

    ``duration`` is the last contact duration in seconds. It is only
    measurable *after* a call has ended, making it unavailable when the
    bank decides which customers to contact. Including it would create
    target leakage and produce a model that cannot be deployed.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with leakage features removed.
    """
    cols_to_drop = [c for c in LEAKAGE_FEATURES if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_target(df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
    """Encode the binary target variable as integers.

    Maps ``'yes'`` → 1 and ``'no'`` → 0 in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the target column.
    target_col : str
        Name of the target column (default ``'y'``).

    Returns
    -------
    pd.DataFrame
        Dataset with integer-encoded target.
    """
    df = df.copy()
    df[target_col] = (df[target_col] == "yes").astype(int)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all standard cleaning steps in a single call.

    Steps applied:
    1. Drop target-leakage features (``duration``).
    2. Encode the target variable (``y``) as 0 / 1.

    ``'unknown'`` string values in categorical columns are intentionally
    *retained* as an explicit category. Analysis showed they carry
    predictive signal and mass-imputation is not appropriate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.
    """
    df = drop_leakage_features(df)
    df = encode_target(df)
    return df
