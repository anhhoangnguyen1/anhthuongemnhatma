#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


DEFAULT_START = "2025-01-25"
DEFAULT_END = "2026-01-25"
LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"
OUNCE_TO_TAEL_FACTOR = 1.20565


def normalize_col(text: object) -> str:
    s = str(text).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s


def read_csv_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = normalize_col(cand)
        for norm_col, real_col in norm_map.items():
            if key and key in norm_col:
                return real_col
    return None


def parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(r"[^\d,.\-]", "", regex=True)
    )
    values: list[float | None] = []
    for text in cleaned:
        t = str(text).strip(".,")
        if not t:
            values.append(None)
            continue
        if "," in t and "." in t:
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "").replace(",", ".")
            else:
                t = t.replace(",", "")
        elif "," in t:
            if t.count(",") == 1:
                left, right = t.split(",", 1)
                t = f"{left}.{right}" if len(right) <= 2 else left + right
            else:
                t = t.replace(",", "")
        elif t.count(".") > 1:
            t = t.replace(".", "")
        try:
            values.append(float(t))
        except ValueError:
            values.append(None)
    return pd.Series(values, index=series.index, dtype="float64")


def fetch_yahoo_close_series(
    session: requests.Session,
    symbol: str,
    value_col: str,
    start_date: str,
    end_date: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo chart returned empty result for {symbol}.")

    first = result[0]
    timestamps = first.get("timestamp") or []
    closes = (((first.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
    if not timestamps or not closes:
        raise ValueError(f"Yahoo chart missing timestamps/close for {symbol}.")

    size = min(len(timestamps), len(closes))
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps[:size], unit="s", utc=True)
            .tz_convert(LOCAL_TIMEZONE)
            .tz_localize(None)
            .normalize(),
            value_col: pd.to_numeric(closes[:size], errors="coerce"),
        }
    )
    df = df.dropna(subset=[value_col]).drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    print(
        f"[OK] Yahoo {symbol}: {len(df)} rows "
        f"({df['date'].min().date()} -> {df['date'].max().date()})"
    )
    time.sleep(sleep_seconds)
    return df


def fetch_fred_series(
    session: requests.Session,
    series_id: str,
    value_col: str,
    start_date: str,
    end_date: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    resp = session.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id},
        timeout=30,
    )
    resp.raise_for_status()

    raw = pd.read_csv(io.StringIO(resp.text))
    date_col = resolve_col(raw, ["observation_date", "date"])
    src_col = resolve_col(raw, [series_id])
    if date_col is None or src_col is None:
        raise KeyError(f"Cannot parse FRED columns for {series_id}. Found: {list(raw.columns)}")

    df = raw.rename(columns={date_col: "date", src_col: value_col}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date", kind="mergesort")
    df = df.reset_index(drop=True)

    print(
        f"[OK] FRED {series_id}: {len(df)} rows "
        f"({df['date'].min().date()} -> {df['date'].max().date()})"
    )
    time.sleep(sleep_seconds)
    return df


def load_interest_baseline(
    interest_path: Path,
    default_state: float,
    default_market: float,
) -> tuple[float, float, str]:
    state_val = float(default_state)
    market_val = float(default_market)
    source_note = "fallback_defaults"

    if not interest_path.exists():
        return state_val, market_val, source_note

    df = read_csv_fallback(interest_path)
    state_col = resolve_col(df, ["interest_rate_state", "state_rate"])
    market_col = resolve_col(df, ["interest_rate_market", "market_rate"])
    spread_col = resolve_col(df, ["interest_rate_spread", "spread"])

    if state_col is not None:
        parsed = parse_numeric(df[state_col]).dropna()
        if not parsed.empty:
            state_val = float(parsed.iloc[-1])
            source_note = f"{interest_path.name}:{state_col}"

    if market_col is not None:
        parsed = parse_numeric(df[market_col]).dropna()
        if not parsed.empty:
            market_val = float(parsed.iloc[-1])
            source_note = f"{interest_path.name}:{market_col}"
    elif spread_col is not None:
        parsed = parse_numeric(df[spread_col]).dropna()
        if not parsed.empty:
            market_val = state_val + float(parsed.iloc[-1])
            source_note = f"{interest_path.name}:{spread_col}"

    return state_val, market_val, source_note


def load_daily_sell_benchmark(gold_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if not gold_path.exists():
        return pd.DataFrame(columns=["date", "benchmark_sell_price"])

    df = read_csv_fallback(gold_path)
    date_col = resolve_col(df, ["ngay", "date", "timestamp"])
    sell_col = resolve_col(df, ["gia_ban", "sell_price", "sell"])
    code_col = resolve_col(df, ["ma_vang", "gold_code"])

    if date_col is None or sell_col is None:
        return pd.DataFrame(columns=["date", "benchmark_sell_price"])

    work = df.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work["sell"] = parse_numeric(work[sell_col])
    work = work.dropna(subset=["date", "sell"]).copy()
    work = work[(work["date"] >= pd.Timestamp(start_date)) & (work["date"] <= pd.Timestamp(end_date))]
    if work.empty:
        return pd.DataFrame(columns=["date", "benchmark_sell_price"])

    if code_col is not None:
        codes = work[code_col].astype(str).str.upper()
        sjc_mask = codes.str.contains("SJC", na=False)
        if sjc_mask.any():
            work = work[sjc_mask].copy()

    daily = work.groupby("date", as_index=False)["sell"].median().rename(
        columns={"sell": "benchmark_sell_price"}
    )
    return daily


def build_macro_daily(
    start_date: str,
    end_date: str,
    usd_vnd_df: pd.DataFrame,
    dxy_df: pd.DataFrame,
    fed_df: pd.DataFrame,
    world_gold_df: pd.DataFrame,
    interest_state: float,
    interest_market: float,
    benchmark_sell_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D").normalize()})
    for table in (usd_vnd_df, dxy_df, fed_df, world_gold_df):
        base = base.merge(table, on="date", how="left")

    fill_cols = ["usd_vnd_rate", "dxy_index", "fed_rate", "World_Price_USD_Ounce"]
    base[fill_cols] = base[fill_cols].ffill().bfill()

    base["interest_rate_state"] = float(interest_state)
    base["interest_rate_market"] = float(interest_market)
    base["interest_rate_spread"] = base["interest_rate_market"] - base["interest_rate_state"]
    base["World_Price_VND"] = (
        base["World_Price_USD_Ounce"] * base["usd_vnd_rate"] * OUNCE_TO_TAEL_FACTOR
    )

    if benchmark_sell_df is not None and not benchmark_sell_df.empty:
        base = base.merge(benchmark_sell_df, on="date", how="left")
        base["Domestic_Premium"] = base["benchmark_sell_price"] - base["World_Price_VND"]
        base = base.drop(columns=["benchmark_sell_price"])
    else:
        base["Domestic_Premium"] = pd.NA

    return base[
        [
            "date",
            "usd_vnd_rate",
            "interest_rate_state",
            "interest_rate_market",
            "interest_rate_spread",
            "dxy_index",
            "fed_rate",
            "World_Price_USD_Ounce",
            "World_Price_VND",
            "Domestic_Premium",
        ]
    ].copy()


def backfill_master_dataset(
    master_input: Path,
    master_output: Path,
    macro_daily: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    if not master_input.exists():
        print(f"[WARN] Master file not found: {master_input}. Skip backfill.")
        return

    master = read_csv_fallback(master_input)
    ts_col = resolve_col(master, ["timestamp", "thoidiemcapnhatdulieu"])
    if ts_col is None:
        print(f"[WARN] No timestamp column found in {master_input}. Skip backfill.")
        return

    work = master.copy()
    original_cols = set(work.columns)
    work["__timestamp__"] = pd.to_datetime(work[ts_col], errors="coerce")
    work["__date__"] = work["__timestamp__"].dt.normalize()
    work = work.merge(macro_daily, left_on="__date__", right_on="date", how="left", suffixes=("", "_new"))

    mask = (work["__date__"] >= pd.Timestamp(start_date)) & (work["__date__"] <= pd.Timestamp(end_date))
    macro_cols = [
        "usd_vnd_rate",
        "interest_rate_state",
        "interest_rate_market",
        "interest_rate_spread",
        "dxy_index",
        "fed_rate",
        "World_Price_USD_Ounce",
        "World_Price_VND",
    ]
    for col in macro_cols:
        src_col = f"{col}_new" if col in original_cols else col
        if src_col not in work.columns:
            continue
        if col not in work.columns:
            work[col] = pd.NA
        assign_mask = mask & work[src_col].notna()
        work.loc[assign_mask, col] = work.loc[assign_mask, src_col]

    sell_col = resolve_col(work, ["sell_price", "gia_ban"])
    if sell_col is not None and "World_Price_VND" in work.columns:
        sell_num = pd.to_numeric(work[sell_col], errors="coerce")
        world_num = pd.to_numeric(work["World_Price_VND"], errors="coerce")
        premium = sell_num - world_num
        if "Domestic_Premium" not in work.columns:
            work["Domestic_Premium"] = pd.NA
        premium_mask = mask & premium.notna()
        work.loc[premium_mask, "Domestic_Premium"] = premium.loc[premium_mask]

    drop_cols = [c for c in work.columns if c.endswith("_new")] + ["date", "__timestamp__", "__date__"]
    work = work.drop(columns=[c for c in drop_cols if c in work.columns])
    work.to_csv(master_output, index=False, encoding="utf-8-sig")

    updated_rows = int(mask.sum())
    print(f"[DONE] Backfilled master rows in range: {updated_rows}")
    print(f"[DONE] Saved: {master_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch 1-year macro history and backfill DSS columns."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date, format YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date, format YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=1.2, help="Sleep seconds between HTTP requests")

    parser.add_argument(
        "--interest-file",
        default="interest_rate.csv",
        help="Local CSV to derive interest_rate_state / interest_rate_market baseline",
    )
    parser.add_argument(
        "--gold-benchmark-file",
        default="GOLD_PRICE_1_YEAR_SCRAPED.csv",
        help="Optional gold CSV to estimate daily Domestic_Premium benchmark",
    )
    parser.add_argument("--default-interest-state", type=float, default=4.92)
    parser.add_argument("--default-interest-market", type=float, default=5.32)

    parser.add_argument(
        "--macro-output",
        default="MACRO_FEATURES_1_YEAR.csv",
        help="Daily macro output CSV",
    )
    parser.add_argument(
        "--master-input",
        default="master_dss_dataset.csv",
        help="Master dataset CSV to backfill",
    )
    parser.add_argument(
        "--master-output",
        default="master_dss_dataset_1Y_macro_backfilled.csv",
        help="Backfilled master output CSV",
    )
    parser.add_argument(
        "--skip-master-backfill",
        action="store_true",
        help="Only create macro daily CSV, do not backfill master dataset",
    )
    args = parser.parse_args()

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    if end_ts < start_ts:
        raise ValueError("--end must be >= --start")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )

    usd_vnd = fetch_yahoo_close_series(
        session=session,
        symbol="VND=X",
        value_col="usd_vnd_rate",
        start_date=args.start,
        end_date=args.end,
        sleep_seconds=args.sleep,
    )
    dxy = fetch_yahoo_close_series(
        session=session,
        symbol="DX-Y.NYB",
        value_col="dxy_index",
        start_date=args.start,
        end_date=args.end,
        sleep_seconds=args.sleep,
    )
    world = fetch_yahoo_close_series(
        session=session,
        symbol="GC=F",
        value_col="World_Price_USD_Ounce",
        start_date=args.start,
        end_date=args.end,
        sleep_seconds=args.sleep,
    )
    try:
        fed = fetch_fred_series(
            session=session,
            series_id="DFF",
            value_col="fed_rate",
            start_date=args.start,
            end_date=args.end,
            sleep_seconds=args.sleep,
        )
    except Exception as e:
        print(f"[WARN] FRED DFF failed ({e}). Using constant fed_rate=4.5 for date range.")
        fed = pd.DataFrame({
            "date": pd.date_range(args.start, args.end, freq="D").normalize(),
            "fed_rate": 4.5,
        })

    interest_state, interest_market, interest_source = load_interest_baseline(
        interest_path=Path(args.interest_file),
        default_state=args.default_interest_state,
        default_market=args.default_interest_market,
    )
    print(
        "[INFO] Interest baseline: "
        f"state={interest_state:.4f}, market={interest_market:.4f}, source={interest_source}"
    )

    benchmark_sell = load_daily_sell_benchmark(
        gold_path=Path(args.gold_benchmark_file),
        start_date=args.start,
        end_date=args.end,
    )
    if benchmark_sell.empty:
        print("[WARN] No daily sell benchmark found. Domestic_Premium in macro file will be empty.")
    else:
        print(
            "[INFO] Loaded daily sell benchmark: "
            f"{len(benchmark_sell)} rows ({benchmark_sell['date'].min().date()} -> "
            f"{benchmark_sell['date'].max().date()})"
        )

    macro_daily = build_macro_daily(
        start_date=args.start,
        end_date=args.end,
        usd_vnd_df=usd_vnd,
        dxy_df=dxy,
        fed_df=fed,
        world_gold_df=world,
        interest_state=interest_state,
        interest_market=interest_market,
        benchmark_sell_df=benchmark_sell,
    )
    macro_daily.to_csv(args.macro_output, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved macro daily: {args.macro_output} ({len(macro_daily)} rows)")

    if not args.skip_master_backfill:
        backfill_master_dataset(
            master_input=Path(args.master_input),
            master_output=Path(args.master_output),
            macro_daily=macro_daily,
            start_date=args.start,
            end_date=args.end,
        )


if __name__ == "__main__":
    main()
