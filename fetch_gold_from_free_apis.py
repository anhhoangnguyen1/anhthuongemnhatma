from __future__ import annotations

"""
Fetch gold prices from completely free public APIs (no API key required).

Current implementation uses FreeGoldAPI:
  - https://freegoldapi.com/data/latest.json
which provides a very long history of annual / monthly / daily gold prices
normalized to USD.

We down-sample this into a simple daily CSV, compatible with the existing
pipeline (can be used as a world-gold reference, or for research / charts).

Usage examples (from project root):
    python fetch_gold_from_free_apis.py
    python fetch_gold_from_free_apis.py --start 2009-07-21 --end 2026-03-09 \\
        --output FREE_GOLDAPI_HISTORY.csv
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import requests


FREEGOLD_JSON_URL = "https://freegoldapi.com/data/latest.json"
DEFAULT_OUTPUT_FILE = "FREE_GOLDAPI_HISTORY.csv"


def fetch_freegold_history() -> pd.DataFrame:
    """Download full history from FreeGoldAPI as a DataFrame."""
    resp = requests.get(FREEGOLD_JSON_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected FreeGoldAPI response format – expected a JSON list.")

    df = pd.DataFrame(data)
    if "date" not in df.columns or "price" not in df.columns:
        raise ValueError("FreeGoldAPI response missing required columns 'date' or 'price'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    return df


def filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        start_ts = pd.to_datetime(start, errors="coerce")
        df = df[df["date"] >= start_ts]
    if end:
        end_ts = pd.to_datetime(end, errors="coerce")
        df = df[df["date"] <= end_ts]
    return df.reset_index(drop=True)


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the (possibly annual / monthly / daily) series into a
    forward-filled daily series.
    """
    if df.empty:
        return df

    daily = (
        df.set_index("date")[["price"]]
        .sort_index()
        .resample("D")
        .last()
        .ffill()
        .reset_index()
    )
    daily.rename(columns={"date": "timestamp", "price": "world_price_usd_ounce"}, inplace=True)
    return daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch gold price history from FreeGoldAPI (100% free, no key)."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2009-07-21",
        help="Start date (inclusive) in YYYY-MM-DD, default: 2009-07-21.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=dt.date.today().isoformat(),
        help="End date (inclusive) in YYYY-MM-DD, default: today.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_FILE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_raw = fetch_freegold_history()
    df_raw = filter_date_range(df_raw, start=args.start, end=args.end)
    df_daily = resample_to_daily(df_raw)

    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Downloaded {len(df_raw)} raw rows, {len(df_daily)} daily rows.")
    print(f"Saved daily gold history to: {output_path}")


if __name__ == "__main__":
    main()

