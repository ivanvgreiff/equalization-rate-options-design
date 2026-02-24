"""Fetch full historical Binance COIN-M BTCUSD_PERP funding-rate data.

Downloads the complete funding-rate history from Binance's public REST API
by paginating forward from the onboard date.

API details:
    Endpoint  : GET https://dapi.binance.com/dapi/v1/fundingRate
    Max rows  : 1000 per request
    Params    : symbol, startTime (ms), endTime (ms), limit
    Pagination: set startTime = max(fundingTime) + 1 from previous page
    Response  : list of {symbol, fundingTime, fundingRate, markPrice}

Contract onboarded 2020-08-10 (fundingInterval = 8h).

Usage:
    python scripts/fetch_binance.py
    python scripts/fetch_binance.py --output data/raw/binance_btcusd_raw.csv
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://dapi.binance.com/dapi/v1/fundingRate"
EXCHANGE_INFO_URL = "https://dapi.binance.com/dapi/v1/exchangeInfo"
SYMBOL = "BTCUSD_PERP"
PAGE_SIZE = 1000
RATE_LIMIT_PAUSE = 0.5
MAX_RETRIES = 5
RETRY_BACKOFF = 3.0


def get_onboard_date() -> int:
    """Fetch the contract's onboardDate (ms) from exchangeInfo."""
    resp = requests.get(EXCHANGE_INFO_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for sym in data["symbols"]:
        if sym["symbol"] == SYMBOL:
            return int(sym["onboardDate"])
    raise ValueError(f"{SYMBOL} not found in exchangeInfo")


def fetch_page(start_time: int, end_time: int) -> list[dict]:
    """Fetch one page of funding data from start_time forward."""
    params = {
        "symbol": SYMBOL,
        "startTime": start_time,
        "endTime": end_time,
        "limit": PAGE_SIZE,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)

            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"  Rate limited (429). Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 451:
                raise RuntimeError(
                    "Binance returned 451 (geo-restricted). "
                    "Cannot access from this location."
                )

            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "code" in data:
                print(f"  API error: {data}")
                return []

            return data

        except requests.exceptions.RequestException as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"  Request error: {e}. Retrying in {wait:.0f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def fetch_all(start_date: str | None = None) -> pd.DataFrame:
    """Paginate forward through the full funding history.

    Binance's COIN-M funding endpoint returns data in ascending order.
    After each page, advance startTime to max(fundingTime) + 1.
    Stop when a page returns fewer than PAGE_SIZE rows.
    """
    if start_date is not None:
        dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cursor = int(dt.timestamp() * 1000)
    else:
        cursor = get_onboard_date()
        onboard_dt = datetime.fromtimestamp(cursor / 1000, tz=timezone.utc)
        print(f"  onboardDate: {onboard_dt.isoformat()}")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_rows: list[dict] = []
    page_num = 0

    while cursor < now_ms:
        page_num += 1
        rows = fetch_page(cursor, now_ms)

        if not rows:
            break

        all_rows.extend(rows)
        max_ts = max(int(rec["fundingTime"]) for rec in rows)
        max_dt = datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc)

        print(
            f"  Page {page_num}: {len(rows)} rows, "
            f"latest={max_dt.strftime('%Y-%m-%d')}, "
            f"total={len(all_rows)}"
        )

        if len(rows) < PAGE_SIZE:
            break

        next_cursor = max_ts + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        time.sleep(RATE_LIMIT_PAUSE)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Binance COIN-M BTCUSD_PERP funding-rate history"
    )
    parser.add_argument(
        "--output",
        default="data/raw/binance_btcusd_raw.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date (YYYY-MM-DD). Default: contract onboard date.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Binance COIN-M {SYMBOL} funding rates...")

    df = fetch_all(start_date=args.start_date)

    if df.empty:
        print("ERROR: No data fetched.")
        return

    df["fundingTime"] = df["fundingTime"].astype(int)
    df = df.drop_duplicates(subset=["fundingTime"]).sort_values("fundingTime")
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    first_dt = datetime.fromtimestamp(df["fundingTime"].iloc[0] / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(df["fundingTime"].iloc[-1] / 1000, tz=timezone.utc)
    print(f"\nDone. {len(df)} rows saved to {output_path}")
    print(f"  Date range: {first_dt.isoformat()} -> {last_dt.isoformat()}")


if __name__ == "__main__":
    main()
