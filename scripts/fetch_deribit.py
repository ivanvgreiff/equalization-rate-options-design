"""Fetch full historical Deribit BTC-PERPETUAL funding-rate data.

Downloads hourly funding-rate history from Deribit's public REST API
and saves it as a CSV file suitable for ingestion by build_dataset.py.

API details:
    Endpoint : GET https://www.deribit.com/api/v2/public/get_funding_rate_history
    Params   : instrument_name, start_timestamp (ms), end_timestamp (ms)
    Response : hourly records with interest_8h, interest_1h, index_price
    Limit    : ~744 rows per request (~31 days of hourly data)

Funding data begins 2019-04-30 (despite creation_timestamp being 2018-08-14).

Usage:
    python scripts/fetch_deribit.py
    python scripts/fetch_deribit.py --output data/raw/deribit_btcperp_raw.csv
    python scripts/fetch_deribit.py --start-date 2020-01-01
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
INSTRUMENT = "BTC-PERPETUAL"
CHUNK_DAYS = 30
RATE_LIMIT_PAUSE = 1.0
MAX_RETRIES = 5
RETRY_BACKOFF = 3.0

# Earliest available funding data (empirically determined)
EARLIEST_MS = 1556582400000  # 2019-04-30 00:00 UTC


def fetch_chunk(start_ms: int, end_ms: int) -> list[dict]:
    """Fetch one time-chunk of hourly funding data."""
    params = {
        "instrument_name": INSTRUMENT,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
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
            data = resp.json()

            if "error" in data:
                err = data["error"]
                print(f"  API error: {err}")
                return []

            return data.get("result", [])

        except requests.exceptions.RequestException as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"  Request error: {e}. Retrying in {wait:.0f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def fetch_all(start_date: str | None = None) -> pd.DataFrame:
    """Fetch the full funding history in 30-day chunks.

    Deribit's API returns at most ~744 rows per request (~31 days hourly).
    We chunk the full date range into 30-day windows and concatenate.
    """
    if start_date is not None:
        dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cursor_ms = int(dt.timestamp() * 1000)
    else:
        cursor_ms = EARLIEST_MS

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    chunk_ms = CHUNK_DAYS * 24 * 3600 * 1000

    all_rows: list[dict] = []
    chunk_num = 0

    while cursor_ms < now_ms:
        chunk_num += 1
        end_ms = min(cursor_ms + chunk_ms, now_ms)

        rows = fetch_chunk(cursor_ms, end_ms)

        if rows:
            all_rows.extend(rows)
            first_dt = datetime.fromtimestamp(rows[0]["timestamp"] / 1000, tz=timezone.utc)
            last_dt = datetime.fromtimestamp(rows[-1]["timestamp"] / 1000, tz=timezone.utc)
            print(
                f"  Chunk {chunk_num}: {len(rows)} rows "
                f"({first_dt.strftime('%Y-%m-%d')} -> {last_dt.strftime('%Y-%m-%d')}), "
                f"total: {len(all_rows)}"
            )
        else:
            start_dt = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc)
            print(f"  Chunk {chunk_num}: 0 rows (from {start_dt.strftime('%Y-%m-%d')})")

        cursor_ms = end_ms
        time.sleep(RATE_LIMIT_PAUSE)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Deribit BTC-PERPETUAL funding-rate history"
    )
    parser.add_argument(
        "--output",
        default="data/raw/deribit_btcperp_raw.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date (YYYY-MM-DD). Default: earliest available (~2019-04-30).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Deribit {INSTRUMENT} funding rates...")
    if args.start_date:
        print(f"  Starting from: {args.start_date}")
    else:
        print("  Starting from: earliest available (~2019-04-30)")

    df = fetch_all(start_date=args.start_date)

    if df.empty:
        print("ERROR: No data fetched.")
        return

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    first_dt = datetime.fromtimestamp(df["timestamp"].iloc[0] / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(df["timestamp"].iloc[-1] / 1000, tz=timezone.utc)
    print(f"\nDone. {len(df)} rows saved to {output_path}")
    print(f"  Date range: {first_dt.isoformat()} -> {last_dt.isoformat()}")


if __name__ == "__main__":
    main()
