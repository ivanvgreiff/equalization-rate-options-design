"""Canonical schema for processed funding-rate data.

The processed dataset has one row per funding interval with these columns:
    timestamp       : UTC datetime of the funding settlement
    funding_rate    : raw per-interval funding rate as published by the venue
    funding_cf      : normalized per-interval cashflow per $1 notional,
                      from the *buyer (short-perp)* perspective.
                      Positive = buyer receives; negative = buyer pays.
    dt_hours        : interval length in hours (8 for standard BitMEX)
    is_regular      : boolean — True if dt_hours is within tolerance of 8h
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "funding_rate", "funding_cf", "dt_hours"]
INTERVAL_HOURS = 8.0
INTERVAL_TOLERANCE_HOURS = 0.5


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if *df* doesn't conform to the canonical schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("'timestamp' must be datetime64")

    if df["funding_cf"].isna().any():
        raise ValueError("'funding_cf' contains NaN — handle gaps before validation")

    return df
