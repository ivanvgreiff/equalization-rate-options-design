"""Preprocessing utilities: raw venue dump -> canonical funding series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ddx.data.schema import validate


def normalize_bitmex(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw BitMEX funding-rate CSV into canonical format.

    Expected raw columns (at minimum):
        timestamp       : ISO-8601 string or datetime
        fundingRate     : per-interval funding fraction (e.g. +0.0001 = 1 bp)

    BitMEX sign convention: positive fundingRate means longs pay shorts.
    For a short-perp buyer, funding_cf = fundingRate (no sign flip).
    """
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["funding_rate"] = df_raw["fundingRate"].values
    out["funding_cf"] = out["funding_rate"]  # short-perp perspective

    # Infer interval lengths from consecutive timestamps
    dt = out["timestamp"].diff().dt.total_seconds() / 3600
    dt.iloc[0] = 8.0  # assume 8h for first row
    out["dt_hours"] = dt

    return validate(out)


def funding_cf_to_apr(cf: np.ndarray, dt_years: float) -> np.ndarray:
    """Convert per-interval CF to annualized APR."""
    return cf / dt_years


def apr_to_funding_cf(apr: np.ndarray, dt_years: float) -> np.ndarray:
    """Convert annualized APR to per-interval CF."""
    return apr * dt_years
