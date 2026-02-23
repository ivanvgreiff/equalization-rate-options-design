"""I/O helpers for reading/writing processed funding data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ddx.data.schema import validate


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load a canonical processed parquet/csv and validate."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return validate(df)


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Validate and save to parquet."""
    validate(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
