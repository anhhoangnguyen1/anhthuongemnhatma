from __future__ import annotations

"""
Fetch financial news + sentiment from Marketaux (free tier) and aggregate to
daily sentiment features for gold-related assets.

Marketaux docs: https://marketaux.com/documentation

You must register for a free API key and either:
  - set environment variable MARKETAUX_API_KEY, or
  - pass --api-key YOUR_KEY on the command line.

Output: a CSV (default: NEWS_SENTIMENT.csv) with columns:
  - date                (UTC date, YYYY-MM-DD)
  - gold_sentiment      (mean sentiment score for gold-related news)
  - gold_news_count     (number of articles)

This file can be joined into the DSS master dataset by
`prepare_gold_dss_pipeline.py` (the pipeline will automatically merge it
when NEWS_SENTIMENT.csv exists in the input directory).

Usage examples (from project root):
    python fetch_news_sentiment_marketaux.py --api-key YOUR_KEY
    python fetch_news_sentiment_marketaux.py --start 2024-01-01 --end 2026-03-09
"""

import argparse
import datetime as dt
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


MARKETAUX_BASE_URL = "https://api.marketaux.com/v1/news/all"
DEFAULT_OUTPUT_FILE = "NEWS_SENTIMENT.csv"


def _get_api_key(cli_key: str | None) -> str:
    key = cli_key or os.getenv("MARKETAUX_API_KEY")
    if not key:
        raise SystemExit(
            "Missing Marketaux API key. Set MARKETAUX_API_KEY environment variable "
            "or pass --api-key on the command line."
        )
    return key


def _daterange(start: dt.date, end: dt.date, step_days: int = 7) -> Iterable[tuple[dt.datetime, dt.datetime]]:
    """Yield (start_dt, end_dt) windows in UTC for pagination."""
    cur = start
    step = dt.timedelta(days=step_days)
    while cur <= end:
        win_start = cur
        win_end = min(end, cur + step - dt.timedelta(days=1))
        yield dt.datetime.combine(win_start, dt.time.min), dt.datetime.combine(win_end, dt.time.max)
        cur = win_end + dt.timedelta(days=1)


def fetch_marketaux_window(
    api_key: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    symbols: str | None = None,
    countries: str | None = None,
    search: str | None = None,
    languages: str = "en",
    max_pages: int = 5,
) -> list[dict[str, Any]]:
    """
    Fetch articles from Marketaux for a given time window.

    We keep usage conservative (max_pages) to respect free-tier limits.
    """
    all_items: list[dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        params = {
            "api_token": api_key,
            "language": languages,
            "filter_entities": "true",
            # Theo ví dụ docs: published_after=2026-03-10T09:56
            "published_after": start_dt.isoformat(timespec="minutes"),
            "published_before": end_dt.isoformat(timespec="minutes"),
            "limit": 100,
            "page": page,
        }
        if symbols:
            params["symbols"] = symbols
        if countries:
            params["countries"] = countries
        if search:
            params["search"] = search

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.get(MARKETAUX_BASE_URL, params=params, timeout=60)
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, ConnectionError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise last_err from e
        else:
            if last_err:
                raise last_err

        if resp.status_code >= 400:
            # In nội dung lỗi rõ ràng rồi dừng, để user thấy nguyên nhân thật từ Marketaux
            try:
                print("[ERROR] Marketaux response:", resp.status_code, resp.text[:500])
            except Exception:
                print("[ERROR] Marketaux HTTP", resp.status_code)
            resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        if not data:
            break
        all_items.extend(data)

        meta = payload.get("meta") or {}
        has_more = bool(meta.get("has_more"))
        if not has_more:
            break
        page += 1

    return all_items


def aggregate_daily_sentiment(articles: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn raw article list into per-day sentiment features."""
    records: list[dict[str, Any]] = []
    for art in articles:
        published_at = art.get("published_at") or art.get("published")
        if not published_at:
            continue
        try:
            ts = pd.to_datetime(published_at, utc=True, errors="coerce")
        except Exception:
            continue
        if pd.isna(ts):
            continue

        # Marketaux usually provides overall_sentiment_score in [-1, 1]
        sent = art.get("overall_sentiment_score")
        if sent is None:
            # fall back to 0 if missing
            sent = 0.0

        records.append(
            {
                "date": ts.date(),
                "sentiment": float(sent),
            }
        )

    if not records:
        return pd.DataFrame(columns=["date", "gold_sentiment", "gold_news_count"])

    df = pd.DataFrame(records)
    grouped = df.groupby("date", as_index=False).agg(
        gold_sentiment=("sentiment", "mean"),
        gold_news_count=("sentiment", "count"),
    )
    return grouped.sort_values("date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch gold-related financial news & sentiment from Marketaux (free tier)."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Marketaux API key (if omitted, read MARKETAUX_API_KEY env var).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(dt.date.today() - dt.timedelta(days=365)).isoformat(),
        help="Start date (YYYY-MM-DD, UTC). Default: today-365d.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=dt.date.today().isoformat(),
        help="End date (YYYY-MM-DD, UTC). Default: today.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to track (optional, e.g. GLD,XAUUSD).",
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="us",
        help="Comma-separated country codes (ISO 3166-1 alpha-2) to filter news, e.g. us,gb,cz.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="gold",
        help="Free-text search query, e.g. 'gold OR \"precious metals\"'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f"Output CSV (default: {DEFAULT_OUTPUT_FILE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = _get_api_key(args.api_key)

    start_date = dt.datetime.fromisoformat(args.start).date()
    end_date = dt.datetime.fromisoformat(args.end).date()
    if end_date < start_date:
        raise SystemExit("end date must be >= start date")

    all_articles: list[dict[str, Any]] = []
    try:
        for win_start, win_end in _daterange(start_date, end_date, step_days=7):
            print(f"[INFO] Fetching {win_start.date()} -> {win_end.date()} ...")
            chunk = fetch_marketaux_window(
                api_key=api_key,
                start_dt=win_start,
                end_dt=win_end,
                symbols=args.symbols,
                countries=args.countries,
                search=args.search,
            )
            print(f"       {len(chunk)} articles")
            all_articles.extend(chunk)
    except (requests.exceptions.RequestException, ConnectionError) as e:
        print(f"[WARN] Request failed ({e}). Saving {len(all_articles)} articles collected so far.")

    daily = aggregate_daily_sentiment(all_articles)
    if daily.empty:
        print("[WARN] No articles returned – output will be empty.")

    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved daily sentiment to: {output_path}  (rows={len(daily)})")


if __name__ == "__main__":
    main()

