"""Preprocessing utilities: raw venue dump -> canonical funding series.

Each venue has a normalize_<venue>() function that converts raw data into
the canonical schema (timestamp, funding_rate, funding_cf, dt_hours, is_regular).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ddx.data.schema import (
    INTERVAL_HOURS,
    INTERVAL_TOLERANCE_HOURS,
    validate,
)

VENUE_NORMALIZERS: dict[str, str] = {
    "bitmex": "normalize_bitmex",
    "deribit": "normalize_deribit",
}


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

    dt = out["timestamp"].diff().dt.total_seconds() / 3600
    dt.iloc[0] = INTERVAL_HOURS
    out["dt_hours"] = dt

    out["is_regular"] = (
        (out["dt_hours"] - INTERVAL_HOURS).abs() < INTERVAL_TOLERANCE_HOURS
    )

    return validate(out)


def enforce_regular_grid(
    df: pd.DataFrame,
    tolerance: float = INTERVAL_TOLERANCE_HOURS,
) -> pd.DataFrame:
    """Drop rows whose dt_hours deviates from 8h by more than *tolerance*.

    Because dropping a row changes the dt of the *next* row, this function
    iteratively recomputes dt and drops newly-irregular rows until the grid
    stabilises.  Returns a copy.
    """
    out = df.copy().reset_index(drop=True)
    while True:
        dt = out["timestamp"].diff().dt.total_seconds() / 3600
        dt.iloc[0] = INTERVAL_HOURS
        out["dt_hours"] = dt
        out["is_regular"] = (out["dt_hours"] - INTERVAL_HOURS).abs() < tolerance

        if out["is_regular"].all():
            break

        out = out.loc[out["is_regular"]].reset_index(drop=True)
        if len(out) == 0:
            break

    return validate(out)


def normalize_deribit(
    df_raw: pd.DataFrame,
    subsample_hours: int = 8,
) -> pd.DataFrame:
    """Convert raw Deribit BTC-PERPETUAL funding history into canonical format.

    Deribit provides hourly records with ``interest_8h`` (rolling 8-hour
    funding rate) and ``interest_1h``.  To align with our 8-hour framework,
    this function subsamples at ``subsample_hours`` boundaries and uses
    ``interest_8h`` as the per-interval cashflow.

    Expected raw columns:
        timestamp    : Unix epoch in *milliseconds*
        interest_8h  : rolling 8-hour funding rate (per-interval fraction)

    Sign convention: positive interest_8h = longs pay shorts (same as BitMEX).
    For the short-perp buyer, funding_cf = interest_8h (no sign flip).
    """
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    out["interest_8h"] = df_raw["interest_8h"].values
    out["interest_1h"] = df_raw["interest_1h"].values

    out = out.sort_values("timestamp").reset_index(drop=True)

    # Subsample at 8h boundaries (00:00, 08:00, 16:00 UTC by default)
    hour = out["timestamp"].dt.hour
    mask = hour % subsample_hours == 0
    out = out.loc[mask].reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("No rows remain after subsampling at 8h boundaries")

    result = pd.DataFrame()
    result["timestamp"] = out["timestamp"]
    result["funding_rate"] = out["interest_8h"]
    result["funding_cf"] = out["interest_8h"]

    dt = result["timestamp"].diff().dt.total_seconds() / 3600
    dt.iloc[0] = INTERVAL_HOURS
    result["dt_hours"] = dt

    result["is_regular"] = (
        (result["dt_hours"] - INTERVAL_HOURS).abs() < INTERVAL_TOLERANCE_HOURS
    )

    return validate(result)


def normalize_bybit(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Bybit BTCUSD inverse perp funding history into canonical format.

    Bybit provides native 8h records at 00:00, 08:00, 16:00 UTC with
    ``fundingRate`` as the per-interval fraction.

    Expected raw columns:
        fundingRateTimestamp : Unix epoch in *milliseconds* (int or str)
        fundingRate          : per-interval funding fraction (str)

    Sign convention: positive fundingRate = longs pay shorts (same as BitMEX).
    For the short-perp buyer, funding_cf = fundingRate (no sign flip).
    """
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(
        pd.to_numeric(df_raw["fundingRateTimestamp"]), unit="ms", utc=True,
    )
    out["funding_rate"] = pd.to_numeric(df_raw["fundingRate"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["funding_cf"] = out["funding_rate"]

    dt = out["timestamp"].diff().dt.total_seconds() / 3600
    dt.iloc[0] = INTERVAL_HOURS
    out["dt_hours"] = dt

    out["is_regular"] = (
        (out["dt_hours"] - INTERVAL_HOURS).abs() < INTERVAL_TOLERANCE_HOURS
    )

    return validate(out)


def normalize_binance(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Binance COIN-M BTCUSD_PERP funding history into canonical format.

    Binance provides native 8h records at 00:00, 08:00, 16:00 UTC with
    ``fundingRate`` as the per-interval fraction.

    Expected raw columns:
        fundingTime  : Unix epoch in *milliseconds* (int, may have small ms offsets)
        fundingRate  : per-interval funding fraction (str)

    Sign convention: positive fundingRate = longs pay shorts (same as BitMEX).
    For the short-perp buyer, funding_cf = fundingRate (no sign flip).
    """
    out = pd.DataFrame()
    # fundingTime sometimes has small ms offsets (e.g., 12ms past the hour);
    # round to nearest second for clean timestamps
    ts_ms = pd.to_numeric(df_raw["fundingTime"])
    ts_rounded = (ts_ms / 1000).round() * 1000
    out["timestamp"] = pd.to_datetime(ts_rounded, unit="ms", utc=True)
    out["funding_rate"] = pd.to_numeric(df_raw["fundingRate"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["funding_cf"] = out["funding_rate"]

    dt = out["timestamp"].diff().dt.total_seconds() / 3600
    dt.iloc[0] = INTERVAL_HOURS
    out["dt_hours"] = dt

    out["is_regular"] = (
        (out["dt_hours"] - INTERVAL_HOURS).abs() < INTERVAL_TOLERANCE_HOURS
    )

    return validate(out)


def normalize_venue(venue: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Dispatch to the appropriate venue normalizer."""
    normalizers = {
        "bitmex": normalize_bitmex,
        "deribit": normalize_deribit,
        "bybit": normalize_bybit,
        "binance": normalize_binance,
    }
    if venue not in normalizers:
        raise ValueError(
            f"Unknown venue {venue!r}. Available: {list(normalizers)}"
        )
    return normalizers[venue](df_raw)


def funding_cf_to_apr(cf: np.ndarray, dt_years: float) -> np.ndarray:
    """Convert per-interval CF to annualized APR."""
    return cf / dt_years


def apr_to_funding_cf(apr: np.ndarray, dt_years: float) -> np.ndarray:
    """Convert annualized APR to per-interval CF."""
    return apr * dt_years
