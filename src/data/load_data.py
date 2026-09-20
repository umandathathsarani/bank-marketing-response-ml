"""Data loading utilities for the Bank Marketing Response project."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "bank_marketing.csv"


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Bank Marketing CSV into a DataFrame.

    The dataset must first be downloaded by running:
        python -c "
        from ucimlrepo import fetch_ucirepo; import pandas as pd
        bm = fetch_ucirepo(id=222)
        df = pd.concat([bm.data.features, bm.data.targets], axis=1)
        df.to_csv('data/raw/bank_marketing.csv', index=False)
        "

    Parameters
    ----------
    path : Path or str
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with 45,211 rows and 17 columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Run the download script in data/raw/ first."
        )
    df = pd.read_csv(path)
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a concise dataset summary."""
    print(f"Shape:      {df.shape}")
    print(f"Target distribution:\n{df['y'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicates: {df.duplicated().sum()}")
