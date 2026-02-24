"""Fetch full historical Bybit BTCUSD inverse perpetual funding-rate data.

Downloads the complete funding-rate history from Bybit's public REST API
by paginating backward from the present to the earliest available record.

API details:
    Endpoint  : GET https://api.bybit.com/v5/market/funding/history
    Max rows  : 200 per request
    Params    : category, symbol, endTime, limit
    Pagination: set endTime = oldest_timestamp - 1 from previous page
    Response  : newest-first list of {symbol, fundingRate, fundingRateTimestamp}

Funding data begins 2018-11-15 (launchTime 2018-11-14T16:00 UTC).
Native 8h intervals at 00:00, 08:00, 16:00 UTC.

Usage:
    python scripts/fetch_bybit.py
    python scripts/fetch_bybit.py --output data/raw/bybit_btcusd_raw.csv
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.bybit.com/v5/market/funding/history"
CATEGORY = "inverse"
SYMBOL = "BTCUSD"
PAGE_SIZE = 200
RATE_LIMIT_PAUSE = 0.5
MAX_RETRIES = 5
RETRY_BACKOFF = 3.0


def fetch_page(end_time: int | None = None) -> list[dict]:
    """Fetch one page of funding data ending at end_time (newest-first)."""
    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "limit": PAGE_SIZE,
    }
    if end_time is not None:
        params["endTime"] = end_time

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)

            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"  Rate limited (429). Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if data.get("retCode") != 0:
                print(f"  API error: retCode={data.get('retCode')}, msg={data.get('retMsg')}")
                return []

            return data.get("result", {}).get("list", [])

        except requests.exceptions.RequestException as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"  Request error: {e}. Retrying in {wait:.0f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def fetch_all() -> pd.DataFrame:
    """Paginate backward through the full funding history.

    Bybit returns records newest-first. After each page, move endTime
    to 1ms before the oldest timestamp received. Stop when a page
    returns fewer than PAGE_SIZE rows (we've reached the beginning).
    """
    all_rows: list[dict] = []
    cursor: int | None = None
    page_num = 0

    while True:
        page_num += 1
        rows = fetch_page(end_time=cursor)

        if not rows:
            break

        all_rows.extend(rows)
        oldest_ts = min(int(rec["fundingRateTimestamp"]) for rec in rows)
        oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)

        print(
            f"  Page {page_num}: {len(rows)} rows, "
            f"oldest={oldest_dt.strftime('%Y-%m-%d')}, "
            f"total={len(all_rows)}"
        )

        if len(rows) < PAGE_SIZE:
            break

        cursor = oldest_ts - 1
        time.sleep(RATE_LIMIT_PAUSE)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Bybit BTCUSD inverse perp funding-rate history"
    )
    parser.add_argument(
        "--output",
        default="data/raw/bybit_btcusd_raw.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Bybit {SYMBOL} funding rates...")

    df = fetch_all()

    if df.empty:
        print("ERROR: No data fetched.")
        return

    df["fundingRateTimestamp"] = df["fundingRateTimestamp"].astype(int)
    df = df.drop_duplicates(subset=["fundingRateTimestamp"]).sort_values("fundingRateTimestamp")
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    first_dt = datetime.fromtimestamp(df["fundingRateTimestamp"].iloc[0] / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(df["fundingRateTimestamp"].iloc[-1] / 1000, tz=timezone.utc)
    print(f"\nDone. {len(df)} rows saved to {output_path}")
    print(f"  Date range: {first_dt.isoformat()} -> {last_dt.isoformat()}")


if __name__ == "__main__":
    main()
