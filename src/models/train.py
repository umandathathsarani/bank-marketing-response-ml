"""Model training utilities for the Bank Marketing Response project."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "gradient_boosting_pipeline.pkl"

# Best hyperparameters from notebook 04 evaluation
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}

# Decision threshold tuned in notebook 05 (maximises F1 on validation set)
DEFAULT_THRESHOLD = 0.30


def build_preprocessor() -> ColumnTransformer:
    """Build the shared ColumnTransformer preprocessing pipeline.

    Numerical features: median imputation + StandardScaler.
    Categorical features: OneHotEncoder (retains 'unknown' as category).

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline,  NUMERICAL_FEATURES),
            ("cat", cat_pipeline,  CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(**gb_params) -> Pipeline:
    """Build a full preprocessing + Gradient Boosting pipeline.

    Parameters
    ----------
    **gb_params
        Keyword arguments forwarded to ``GradientBoostingClassifier``.
        Defaults to ``DEFAULT_PARAMS`` if none provided.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline.
    """
    params = {**DEFAULT_PARAMS, **gb_params}
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model",        GradientBoostingClassifier(**params)),
    ])


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_path: Path | str | None = DEFAULT_MODEL_PATH,
    **gb_params,
) -> Pipeline:
    """Train the Gradient Boosting pipeline and optionally save to disk.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (pre-engineered, including all columns needed
        by ``NUMERICAL_FEATURES`` and ``CATEGORICAL_FEATURES``).
    y_train : pd.Series
        Binary target (0 = no subscription, 1 = subscription).
    save_path : Path, str, or None
        Where to save the fitted pipeline as a .pkl file.
        Pass ``None`` to skip saving.
    **gb_params
        Optional overrides for GradientBoostingClassifier parameters.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for prediction.
    """
    pipeline = build_pipeline(**gb_params)
    pipeline.fit(X_train, y_train)
    print(f"Model trained on {len(X_train):,} samples.")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        print(f"Pipeline saved to {save_path}")

    return pipeline


def load_pipeline(path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline from disk.

    Parameters
    ----------
    path : Path or str
        Path to the .pkl file.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model found at {path}. "
            "Run train() first or provide a valid path."
        )
    return joblib.load(path)
