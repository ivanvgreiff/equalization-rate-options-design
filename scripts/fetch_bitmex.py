"""Fetch full historical BitMEX XBTUSD funding-rate data.

Downloads the complete funding-rate history from BitMEX's public REST API
and saves it as a CSV file suitable for ingestion by build_dataset.py.

API details:
    Endpoint : GET https://www.bitmex.com/api/v1/funding
    Max rows : 500 per request
    Params   : symbol, startTime, endTime, count, reverse, start (offset)
    Rate limit: ~30 requests/min for unauthenticated access

Usage:
    python scripts/fetch_bitmex.py
    python scripts/fetch_bitmex.py --output data/raw/bitmex_xbtusd_raw.csv
    python scripts/fetch_bitmex.py --start-date 2020-01-01
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://www.bitmex.com/api/v1/funding"
SYMBOL = "XBTUSD"
PAGE_SIZE = 500
RATE_LIMIT_PAUSE = 2.0  # seconds between requests (conservative)
MAX_RETRIES = 5
RETRY_BACKOFF = 5.0  # seconds base for exponential backoff


def fetch_page(
    start_time: str,
    count: int = PAGE_SIZE,
) -> list[dict]:
    """Fetch a single page of funding data starting at start_time."""
    params = {
        "symbol": SYMBOL,
        "startTime": start_time,
        "count": count,
        "reverse": "false",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)

            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"  Rate limited (429). Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"  Request error: {e}. Retrying in {wait:.0f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def fetch_all(start_date: str | None = None) -> pd.DataFrame:
    """Paginate through the full funding history.

    The strategy: fetch PAGE_SIZE rows at a time using startTime.
    After each page, advance startTime to 1 second past the last
    timestamp received. Stop when a page returns fewer rows than
    PAGE_SIZE (we have reached the end).
    """
    if start_date is None:
        cursor = "2016-05-01T00:00:00.000Z"
    else:
        cursor = f"{start_date}T00:00:00.000Z"

    all_rows: list[dict] = []
    page_num = 0

    while True:
        page_num += 1
        rows = fetch_page(cursor)

        if not rows:
            break

        all_rows.extend(rows)
        last_ts = rows[-1]["timestamp"]

        n_total = len(all_rows)
        print(
            f"  Page {page_num}: got {len(rows)} rows "
            f"({rows[0]['timestamp'][:10]} -> {last_ts[:10]}), "
            f"total so far: {n_total}"
        )

        if len(rows) < PAGE_SIZE:
            break

        # Advance cursor past the last timestamp to avoid duplicates.
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        next_dt = last_dt + pd.Timedelta(seconds=1)
        cursor = next_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        time.sleep(RATE_LIMIT_PAUSE)

    df = pd.DataFrame(all_rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch BitMEX XBTUSD funding-rate history"
    )
    parser.add_argument(
        "--output",
        default="data/raw/bitmex_xbtusd_raw.csv",
        help="Output CSV path (default: data/raw/bitmex_xbtusd_raw.csv)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date (YYYY-MM-DD). Default: fetch from earliest available.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching BitMEX {SYMBOL} funding rates...")
    if args.start_date:
        print(f"  Starting from: {args.start_date}")
    else:
        print("  Starting from: earliest available (2016-05-14)")

    df = fetch_all(start_date=args.start_date)

    if df.empty:
        print("ERROR: No data fetched.")
        return

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    ts_min = df["timestamp"].iloc[0]
    ts_max = df["timestamp"].iloc[-1]
    print(f"\nDone. {len(df)} rows saved to {output_path}")
    print(f"  Date range: {ts_min} -> {ts_max}")


if __name__ == "__main__":
    main()
