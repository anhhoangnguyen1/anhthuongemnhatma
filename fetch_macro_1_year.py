#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch historical SJC gold prices from giavang.org and macro data from
Yahoo Finance / FRED, then produce a dataset matching master_dss_dataset.csv.

Usage examples:
    python fetch_macro_1_year.py
    python fetch_macro_1_year.py --start 2009-07-21 --end 2026-03-09
    python fetch_macro_1_year.py --skip-scrape   # use cached giavang data only
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit(
        "Missing dependency 'beautifulsoup4'. Install via:\n"
        "  pip install beautifulsoup4 lxml"
    )

try:
    from ta.momentum import RSIIndicator
except ImportError:
    RSIIndicator = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_START = "2009-07-21"
DEFAULT_END = dt.date.today().isoformat()
LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"
OUNCE_TO_TAEL_FACTOR = 1.20565
GIAVANG_URL_TEMPLATE = "https://giavang.org/trong-nuoc/sjc/lich-su/{}.html"

ALL_GOLD_CODES = [
    "BT9999NTT", "BTSJC", "DOHCML", "DOHNL", "DOJINHTV",
    "PQHN24NTT", "PQHNVM", "SJ9999", "SJL1L10", "VIETTINMSJC", "VNGSJC",
]

# World Bank Vietnam deposit interest rate (yearly %)
# Source: IMF International Financial Statistics via World Bank API
# Indicator: FR.INR.DPST  |  URL: data.worldbank.org/indicator/FR.INR.DPST?locations=VN
# Used as fallback when the live API is unreachable.
WB_VN_DEPOSIT_RATE_FALLBACK: dict[int, float] = {
    2009:  7.910, 2010: 11.194, 2011: 13.994, 2012: 10.504,
    2013:  7.140, 2014:  5.758, 2015:  4.748, 2016:  5.035,
    2017:  4.809, 2018:  4.738, 2019:  4.975, 2020:  4.120,
    2021:  3.375, 2022:  3.818, 2023:  4.781,
}
WB_DEPOSIT_RATE_API = (
    "https://api.worldbank.org/v2/country/VN/indicator/FR.INR.DPST"
    "?format=json&date={start}:{end}&per_page=100"
)


# ---------------------------------------------------------------------------
# Utility helpers (kept from original)
# ---------------------------------------------------------------------------
def normalize_col(text: object) -> str:
    s = str(text).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s


def _strip_diacritics(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


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


# ---------------------------------------------------------------------------
# Yahoo Finance / FRED fetchers (kept from original)
# ---------------------------------------------------------------------------
def fetch_yahoo_close_series(
    session: requests.Session,
    symbol: str,
    value_col: str,
    start_date: str,
    end_date: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    period2 = int(
        (pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp()
    )

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(symbol, safe='')}"
    )
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
    closes = (
        ((first.get("indicators") or {}).get("quote") or [{}])[0]
    ).get("close") or []
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
        raise KeyError(
            f"Cannot parse FRED columns for {series_id}. "
            f"Found: {list(raw.columns)}"
        )

    df = raw.rename(columns={date_col: "date", src_col: value_col}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df[
        (df["date"] >= pd.Timestamp(start_date))
        & (df["date"] <= pd.Timestamp(end_date))
    ]
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values(
        "date", kind="mergesort"
    )
    df = df.reset_index(drop=True)

    print(
        f"[OK] FRED {series_id}: {len(df)} rows "
        f"({df['date'].min().date()} -> {df['date'].max().date()})"
    )
    time.sleep(sleep_seconds)
    return df


def fetch_worldbank_vn_deposit_rate(
    session: requests.Session,
    start_year: int = 2009,
    end_year: int | None = None,
) -> dict[int, float]:
    """Fetch Vietnam deposit interest rate (yearly) from World Bank API.

    Source: IMF International Financial Statistics via World Bank
    Indicator: FR.INR.DPST
    URL: https://data.worldbank.org/indicator/FR.INR.DPST?locations=VN
    """
    if end_year is None:
        end_year = dt.date.today().year
    url = WB_DEPOSIT_RATE_API.format(start=start_year, end=end_year)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError("Empty response from World Bank API")
        rates: dict[int, float] = {}
        for rec in data[1]:
            year = int(rec["date"])
            value = rec.get("value")
            if value is not None:
                rates[year] = round(float(value), 4)
        if rates:
            print(
                f"[OK] World Bank VN deposit rate (FR.INR.DPST): "
                f"{len(rates)} years "
                f"({min(rates)}: {rates[min(rates)]:.2f}% -> "
                f"{max(rates)}: {rates[max(rates)]:.2f}%)"
            )
            return rates
        raise ValueError("No non-null values in API response")
    except Exception as exc:
        print(f"[WARN] World Bank API failed ({exc}). Using fallback data.")
        return WB_VN_DEPOSIT_RATE_FALLBACK.copy()


def get_historical_interest_rates(
    dates: pd.Series,
    wb_rates: dict[int, float],
    interest_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Map each date to Vietnam deposit interest rate.

    Historical (2009-2023): World Bank yearly deposit rate (FR.INR.DPST from
    IMF IFS).  Applied as a step function per year.  Both state and market
    receive the same value because no free source separates Big-4 vs market
    bank rates for historical periods.

    Recent (from interest_rate.csv): Actual separate state/market rates from
    webgia.com.  Used to override from the year after the last World Bank
    observation onward.

    interest_rate_spread = interest_rate_market - interest_rate_state
    """
    dates_clean = pd.to_datetime(dates).dt.normalize()
    lookup = (
        pd.DataFrame({"date": dates_clean})
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )

    wb_records = [
        {
            "date": pd.Timestamp(f"{year}-01-01"),
            "interest_rate_state": rate,
            "interest_rate_market": rate,
        }
        for year, rate in sorted(wb_rates.items())
    ]
    wb_df = pd.DataFrame(wb_records).sort_values("date")

    if interest_csv_path and interest_csv_path.exists():
        ir_raw = read_csv_fallback(interest_csv_path)
        state_col = resolve_col(ir_raw, ["interest_rate_state", "state_rate"])
        market_col = resolve_col(
            ir_raw, ["interest_rate_market", "market_rate"]
        )
        if state_col and market_col:
            s_vals = parse_numeric(ir_raw[state_col]).dropna()
            m_vals = parse_numeric(ir_raw[market_col]).dropna()
            if not s_vals.empty and not m_vals.empty:
                ir_state = float(s_vals.iloc[-1])
                ir_market = float(m_vals.iloc[-1])
                last_wb_year = max(wb_rates.keys()) if wb_rates else 2023
                override_date = pd.Timestamp(f"{last_wb_year + 1}-01-01")
                wb_df = pd.concat(
                    [
                        wb_df,
                        pd.DataFrame(
                            [
                                {
                                    "date": override_date,
                                    "interest_rate_state": ir_state,
                                    "interest_rate_market": ir_market,
                                }
                            ]
                        ),
                    ]
                ).sort_values("date").drop_duplicates(
                    subset=["date"], keep="last"
                ).reset_index(drop=True)
                print(
                    f"[INFO] interest_rate.csv override from "
                    f"{override_date.date()}: state={ir_state:.2f}%, "
                    f"market={ir_market:.2f}%"
                )

    result = pd.merge_asof(
        lookup, wb_df, on="date", direction="backward"
    )
    result["interest_rate_state"] = (
        result["interest_rate_state"].ffill().bfill()
    )
    result["interest_rate_market"] = (
        result["interest_rate_market"].ffill().bfill()
    )
    result["interest_rate_spread"] = (
        result["interest_rate_market"] - result["interest_rate_state"]
    )
    return result[
        [
            "date",
            "interest_rate_state",
            "interest_rate_market",
            "interest_rate_spread",
        ]
    ]


def extract_xauusd_from_gold_price(gold_price_path: Path) -> pd.DataFrame:
    """Extract XAUUSD (world spot gold) rows from GOLD_PRICE.csv.

    Returns a DataFrame with columns: date, World_Price_USD_Ounce
    (one row per date, using the last update's buy_price).
    """
    if not gold_price_path.exists():
        return pd.DataFrame(columns=["date", "World_Price_USD_Ounce"])

    raw = read_csv_fallback(gold_price_path)
    code_col = resolve_col(raw, ["Mã vàng", "ma_vang", "gold_code"])
    buy_col = resolve_col(raw, ["Giá mua", "gia_mua", "buy_price"])
    ts_col = resolve_col(
        raw,
        ["Thời điểm cập nhật dữ liệu", "timestamp", "ngay", "date"],
    )
    if not (code_col and buy_col and ts_col):
        return pd.DataFrame(columns=["date", "World_Price_USD_Ounce"])

    mask = raw[code_col].astype(str).str.strip().str.upper() == "XAUUSD"
    xau = raw.loc[mask].copy()
    if xau.empty:
        return pd.DataFrame(columns=["date", "World_Price_USD_Ounce"])

    xau["date"] = pd.to_datetime(xau[ts_col], errors="coerce").dt.normalize()
    xau["World_Price_USD_Ounce"] = parse_numeric(xau[buy_col])
    xau = xau.dropna(subset=["date", "World_Price_USD_Ounce"])
    xau = xau[xau["World_Price_USD_Ounce"] > 0]
    xau = (
        xau.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    print(
        f"[OK] XAUUSD from GOLD_PRICE.csv: {len(xau)} rows "
        f"({xau['date'].min().date()} -> {xau['date'].max().date()})"
    )
    return xau[["date", "World_Price_USD_Ounce"]]


# ---------------------------------------------------------------------------
# giavang.org scraper
# ---------------------------------------------------------------------------
def _parse_vnd_price(text: str) -> float | None:
    """Parse giavang.org price text (x1000d/luong) to full VND.

    '82.800' -> 82_800_000 ; dots are thousands separators.
    """
    s = text.strip()
    if ":" in s or "/" in s:
        return None
    s = s.replace(".", "").replace(",", "")
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return None
    val = int(s) * 1000
    if val < 1_000_000 or val > 500_000_000:
        return None
    return float(val)


def _scrape_giavang_single_day(
    session: requests.Session,
    date_str: str,
    sleep_seconds: float,
) -> tuple[float | None, float | None]:
    """Scrape one day from giavang.org and return (buy, sell) in VND."""
    url = GIAVANG_URL_TEMPLATE.format(date_str)
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception:
        time.sleep(sleep_seconds)
        return None, None

    html = resp.text
    if "Không tìm thấy dữ liệu" in html:
        time.sleep(sleep_seconds)
        return None, None

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        buy_idx: int | None = None
        sell_idx: int | None = None
        for row in rows:
            cells = row.find_all(["th", "td"])
            for i, cell in enumerate(cells):
                ct = _strip_diacritics(cell.get_text(strip=True))
                if "mua" in ct and buy_idx is None:
                    buy_idx = i
                elif "ban" in ct and "chenh" not in ct and sell_idx is None:
                    sell_idx = i
            if buy_idx is not None and sell_idx is not None:
                break

        if buy_idx is None or sell_idx is None:
            continue

        for row in rows:
            cells = row.find_all(["th", "td"])
            if not cells or len(cells) <= max(buy_idx, sell_idx):
                continue
            if all(c.name == "th" for c in cells):
                continue

            row_text = " ".join(c.get_text(strip=True) for c in cells)
            row_upper = row_text.upper()
            if "SJC" not in row_upper:
                continue
            row_lower = _strip_diacritics(row_text)
            if any(skip in row_lower for skip in ["nhan", "nu trang"]):
                continue

            buy_price = _parse_vnd_price(
                cells[buy_idx].get_text(strip=True)
            )
            sell_price = _parse_vnd_price(
                cells[sell_idx].get_text(strip=True)
            )

            if buy_price is not None and sell_price is not None:
                time.sleep(sleep_seconds)
                return buy_price, sell_price

    time.sleep(sleep_seconds)
    return None, None


def fetch_sjc_from_giavang(
    session: requests.Session,
    start_date: str,
    end_date: str,
    sleep_seconds: float,
    cache_path: Path | None,
) -> pd.DataFrame:
    """Scrape daily SJC prices from giavang.org with CSV caching."""
    cached = pd.DataFrame(columns=["date", "buy_price", "sell_price"])
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"[INFO] Loaded cache: {cache_path} ({len(cached)} rows)")

    all_dates = pd.date_range(start_date, end_date, freq="D")
    cached_dates = set(cached["date"].dt.date) if not cached.empty else set()
    to_scrape = [d for d in all_dates if d.date() not in cached_dates]

    print(
        f"[INFO] giavang.org: {len(to_scrape)} dates to scrape "
        f"({len(all_dates)} total, {len(cached_dates)} cached)"
    )

    new_rows: list[dict] = []
    for i, date_ts in enumerate(to_scrape):
        date_str = date_ts.strftime("%Y-%m-%d")
        buy, sell = _scrape_giavang_single_day(session, date_str, sleep_seconds)
        if buy is not None and sell is not None:
            new_rows.append(
                {"date": date_ts, "buy_price": buy, "sell_price": sell}
            )

        if (i + 1) % 100 == 0:
            print(
                f"  [{i + 1}/{len(to_scrape)}] Up to {date_str}, "
                f"{len(new_rows)} prices found"
            )

        if cache_path and new_rows and (i + 1) % 500 == 0:
            _save_cache(cached, new_rows, cache_path)

    if new_rows:
        combined = pd.concat(
            [cached, pd.DataFrame(new_rows)], ignore_index=True
        )
    else:
        combined = cached.copy()

    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if cache_path:
        combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved cache: {cache_path} ({len(combined)} rows)")

    return combined


def _save_cache(
    cached: pd.DataFrame,
    new_rows: list[dict],
    cache_path: Path,
) -> None:
    interim = pd.concat(
        [cached, pd.DataFrame(new_rows)], ignore_index=True
    )
    interim["date"] = pd.to_datetime(interim["date"])
    interim = interim.drop_duplicates(subset=["date"], keep="last").sort_values(
        "date"
    )
    interim.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"  [CACHE] Saved interim: {len(interim)} rows -> {cache_path}")


# ---------------------------------------------------------------------------
# Merge gold data sources
# ---------------------------------------------------------------------------
def merge_all_gold_sources(
    giavang_df: pd.DataFrame,
    scraped_path: Path | None,
    gold_price_path: Path | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Combine SJC gold data from giavang.org + local CSV files.

    For historical dates where only a single SJC price exists, the price is
    replicated across all standard gold codes so the output matches the
    multi-code layout of master_dss_dataset.csv.
    """
    parts: list[pd.DataFrame] = []

    # Source 1 -- giavang.org scraped data
    if not giavang_df.empty:
        gv = giavang_df.copy()
        gv = gv.rename(columns={"date": "timestamp"})
        gv["gold_code"] = "SJL1L10"
        parts.append(gv[["timestamp", "gold_code", "buy_price", "sell_price"]])
        print(f"[INFO] giavang.org gold rows: {len(gv)}")

    # Source 2 -- GOLD_PRICE_1_YEAR_SCRAPED.csv
    if scraped_path and scraped_path.exists():
        raw = read_csv_fallback(scraped_path)
        date_col = resolve_col(raw, ["Ngày", "ngay", "date", "timestamp"])
        buy_col = resolve_col(raw, ["Giá mua", "gia_mua", "buy_price"])
        sell_col = resolve_col(raw, ["Giá bán", "gia_ban", "sell_price"])
        if date_col and buy_col and sell_col:
            scraped = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        raw[date_col], errors="coerce"
                    ),
                    "gold_code": "SJL1L10",
                    "buy_price": parse_numeric(raw[buy_col]),
                    "sell_price": parse_numeric(raw[sell_col]),
                }
            )
            scraped = scraped.dropna(
                subset=["timestamp", "buy_price", "sell_price"]
            )
            parts.append(scraped)
            print(f"[INFO] Scraped gold rows: {len(scraped)}")

    # Source 3 -- GOLD_PRICE.csv (multi-brand, hourly data)
    if gold_price_path and gold_price_path.exists():
        raw = read_csv_fallback(gold_price_path)
        ts_col = resolve_col(
            raw,
            [
                "Thời điểm cập nhật dữ liệu",
                "timestamp",
                "ngay",
                "date",
            ],
        )
        code_col = resolve_col(raw, ["Mã vàng", "ma_vang", "gold_code"])
        buy_col = resolve_col(raw, ["Giá mua", "gia_mua", "buy_price"])
        sell_col = resolve_col(raw, ["Giá bán", "gia_ban", "sell_price"])
        if ts_col and code_col and buy_col and sell_col:
            gp = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        raw[ts_col], errors="coerce"
                    ),
                    "gold_code": raw[code_col]
                    .astype(str)
                    .str.strip()
                    .str.upper(),
                    "buy_price": parse_numeric(raw[buy_col]),
                    "sell_price": parse_numeric(raw[sell_col]),
                }
            )
            gp = gp.dropna(subset=["timestamp", "gold_code", "sell_price"])
            gp = gp[gp["gold_code"] != "XAUUSD"]
            gp["_date"] = gp["timestamp"].dt.normalize()
            gp = gp.sort_values("timestamp").drop_duplicates(
                subset=["_date", "gold_code"], keep="last"
            )
            gp["timestamp"] = gp["_date"]
            gp = gp.drop(columns=["_date"])
            parts.append(
                gp[["timestamp", "gold_code", "buy_price", "sell_price"]]
            )
            print(f"[INFO] GOLD_PRICE.csv rows: {len(gp)}")

    if not parts:
        return pd.DataFrame(
            columns=["timestamp", "gold_code", "buy_price", "sell_price"]
        )

    combined = pd.concat(parts, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"]).dt.normalize()
    combined = combined[
        (combined["timestamp"] >= pd.Timestamp(start_date))
        & (combined["timestamp"] <= pd.Timestamp(end_date))
    ]

    # Prefer multi-brand data over single-SJC data on the same date
    combined = combined.sort_values(
        ["timestamp", "gold_code"]
    ).drop_duplicates(subset=["timestamp", "gold_code"], keep="last")

    # For dates with only one gold_code, replicate across all standard codes
    code_counts = combined.groupby("timestamp")["gold_code"].nunique()
    single_dates = set(code_counts[code_counts == 1].index)

    if single_dates:
        single_rows = combined[combined["timestamp"].isin(single_dates)]
        replicated: list[dict] = []
        for _, row in single_rows.drop_duplicates(
            subset=["timestamp"]
        ).iterrows():
            for code in ALL_GOLD_CODES:
                replicated.append(
                    {
                        "timestamp": row["timestamp"],
                        "gold_code": code,
                        "buy_price": row["buy_price"],
                        "sell_price": row["sell_price"],
                    }
                )
        combined = combined[~combined["timestamp"].isin(single_dates)]
        combined = pd.concat(
            [combined, pd.DataFrame(replicated)], ignore_index=True
        )

    combined = (
        combined.sort_values(["timestamp", "gold_code"])
        .drop_duplicates(subset=["timestamp", "gold_code"], keep="last")
        .reset_index(drop=True)
    )

    print(
        f"[OK] Merged gold: {len(combined)} rows, "
        f"{combined['timestamp'].min().date()} -> "
        f"{combined['timestamp'].max().date()}"
    )
    return combined


# ---------------------------------------------------------------------------
# Technical features and supervised targets
# ---------------------------------------------------------------------------
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        ["gold_code", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = df.groupby("gold_code", dropna=False)["sell_price"]

    df["MA_7"] = grouped.transform(
        lambda s: s.rolling(window=7, min_periods=7).mean()
    )
    df["MA_30"] = grouped.transform(
        lambda s: s.rolling(window=30, min_periods=30).mean()
    )

    if RSIIndicator is not None:
        df["RSI_14"] = grouped.transform(
            lambda s: RSIIndicator(
                close=s.astype("float64"), window=14, fillna=False
            ).rsi()
        )
    else:
        print("[WARN] 'ta' library not installed. RSI_14 will be NaN.")
        df["RSI_14"] = np.nan

    df["Momentum_1D_Pct"] = grouped.transform(
        lambda s: s.pct_change(periods=1) * 100.0
    )
    df["Momentum_3D_Pct"] = grouped.transform(
        lambda s: s.pct_change(periods=3) * 100.0
    )
    df["Momentum_7D_Pct"] = grouped.transform(
        lambda s: s.pct_change(periods=7) * 100.0
    )
    return df


def add_supervised_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(
        ["gold_code", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = df.groupby("gold_code", dropna=False)["sell_price"]

    df["Target_Price_Next_Shift"] = grouped.shift(-1)
    df["Target_Trend"] = (
        df["Target_Price_Next_Shift"] > df["sell_price"]
    ).astype("int8")

    df = df.dropna(subset=["Target_Price_Next_Shift"]).reset_index(drop=True)
    df["Target_Trend"] = df["Target_Trend"].astype("int8")
    return df


# ---------------------------------------------------------------------------
# Build full master dataset
# ---------------------------------------------------------------------------

MASTER_COLUMNS = [
    "timestamp",
    "gold_code",
    "buy_price",
    "sell_price",
    "usd_vnd_rate",
    "interest_rate_state",
    "interest_rate_market",
    "interest_rate_spread",
    "dxy_index",
    "fed_rate",
    "World_Price_USD_Ounce",
    "World_Price_VND",
    "Domestic_Premium",
    "MA_7",
    "MA_30",
    "RSI_14",
    "Momentum_1D_Pct",
    "Momentum_3D_Pct",
    "Momentum_7D_Pct",
    "Target_Price_Next_Shift",
    "Target_Trend",
]


def build_full_master(
    gold_df: pd.DataFrame,
    usd_vnd_df: pd.DataFrame,
    dxy_df: pd.DataFrame,
    fed_df: pd.DataFrame,
    world_gold_df: pd.DataFrame,
    wb_rates: dict[int, float],
    interest_csv_path: Path | None = None,
    xauusd_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge gold prices with macro data and compute all derived features."""
    df = gold_df.copy()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Combine Yahoo GC=F with XAUUSD from GOLD_PRICE.csv (prefer XAUUSD)
    if xauusd_df is not None and not xauusd_df.empty:
        yahoo_world = world_gold_df.copy()
        xau_renamed = xauusd_df.rename(
            columns={"World_Price_USD_Ounce": "xau_price"}
        )
        merged_world = pd.merge(
            yahoo_world, xau_renamed, on="date", how="outer"
        ).sort_values("date")
        merged_world["World_Price_USD_Ounce"] = merged_world[
            "xau_price"
        ].combine_first(merged_world["World_Price_USD_Ounce"])
        merged_world = merged_world.drop(columns=["xau_price"]).dropna(
            subset=["World_Price_USD_Ounce"]
        )
        merged_world = merged_world.drop_duplicates(
            subset=["date"], keep="last"
        ).reset_index(drop=True)
        print(
            f"[INFO] World gold after XAUUSD merge: {len(merged_world)} rows "
            f"({merged_world['date'].min().date()} -> "
            f"{merged_world['date'].max().date()})"
        )
        world_gold_df = merged_world

    macro_sources = {
        "usd_vnd_rate": usd_vnd_df,
        "dxy_index": dxy_df,
        "fed_rate": fed_df,
        "World_Price_USD_Ounce": world_gold_df,
    }
    for col, macro_df in macro_sources.items():
        right = (
            macro_df.rename(columns={"date": "timestamp"})[["timestamp", col]]
            .dropna()
            .sort_values("timestamp", kind="mergesort")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            right,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    macro_cols = list(macro_sources.keys())
    df = df.sort_values(
        ["gold_code", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    df[macro_cols] = df.groupby("gold_code", dropna=False)[macro_cols].ffill()
    df[macro_cols] = df.groupby("gold_code", dropna=False)[macro_cols].bfill()

    # Per-date interest rates: World Bank yearly + interest_rate.csv recent
    ir = get_historical_interest_rates(
        df["timestamp"], wb_rates, interest_csv_path
    )
    ir_map = ir.set_index("date")
    df["interest_rate_state"] = df["timestamp"].dt.normalize().map(
        ir_map["interest_rate_state"]
    )
    df["interest_rate_market"] = df["timestamp"].dt.normalize().map(
        ir_map["interest_rate_market"]
    )
    df["interest_rate_spread"] = df["timestamp"].dt.normalize().map(
        ir_map["interest_rate_spread"]
    )

    df["World_Price_VND"] = (
        df["World_Price_USD_Ounce"]
        * df["usd_vnd_rate"]
        * OUNCE_TO_TAEL_FACTOR
    )
    df["Domestic_Premium"] = df["sell_price"] - df["World_Price_VND"]

    print("[INFO] Adding technical features ...")
    df = add_technical_features(df)
    print("[INFO] Adding supervised targets ...")
    df = add_supervised_targets(df)

    df = df.sort_values(
        ["timestamp", "gold_code"], kind="mergesort"
    ).reset_index(drop=True)

    return df[MASTER_COLUMNS]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical SJC gold prices from giavang.org and macro "
            "data from Yahoo/FRED to build a dataset matching "
            "master_dss_dataset.csv."
        )
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Start date YYYY-MM-DD (default: %(default)s)",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help="End date YYYY-MM-DD (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between HTTP requests (default: %(default)s)",
    )

    parser.add_argument(
        "--sjc-cache",
        default="sjc_giavang_cache.csv",
        help="CSV cache file for giavang.org scraped data",
    )
    parser.add_argument(
        "--scraped-file",
        default="GOLD_PRICE_1_YEAR_SCRAPED.csv",
        help="Local scraped gold CSV (1-year)",
    )
    parser.add_argument(
        "--gold-file",
        default="GOLD_PRICE.csv",
        help="Local multi-brand gold CSV",
    )
    parser.add_argument(
        "--interest-file",
        default="interest_rate.csv",
        help="Local interest_rate.csv (recent state/market rates from webgia.com)",
    )

    parser.add_argument(
        "--output",
        default="master_dss_dataset_full.csv",
        help="Output CSV matching master_dss_dataset format",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip giavang.org scraping; only use cache and local files",
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

    # ---- Step 1: SJC gold prices from giavang.org ----
    cache_path = Path(args.sjc_cache) if args.sjc_cache else None
    if args.skip_scrape:
        print("[INFO] Skipping giavang.org scrape (--skip-scrape)")
        if cache_path and cache_path.exists():
            giavang_df = pd.read_csv(cache_path, parse_dates=["date"])
            print(f"[INFO] Loaded cache: {len(giavang_df)} rows")
        else:
            giavang_df = pd.DataFrame(
                columns=["date", "buy_price", "sell_price"]
            )
    else:
        giavang_df = fetch_sjc_from_giavang(
            session=session,
            start_date=args.start,
            end_date=args.end,
            sleep_seconds=args.sleep,
            cache_path=cache_path,
        )

    # ---- Step 2: macro data from Yahoo / FRED ----
    print("\n--- Fetching macro data ---")
    usd_vnd = fetch_yahoo_close_series(
        session, "VND=X", "usd_vnd_rate", args.start, args.end, args.sleep
    )
    dxy = fetch_yahoo_close_series(
        session, "DX-Y.NYB", "dxy_index", args.start, args.end, args.sleep
    )
    world = fetch_yahoo_close_series(
        session,
        "GC=F",
        "World_Price_USD_Ounce",
        args.start,
        args.end,
        args.sleep,
    )
    try:
        fed = fetch_fred_series(
            session, "DFF", "fed_rate", args.start, args.end, args.sleep
        )
    except Exception as exc:
        print(
            f"[WARN] FRED DFF failed ({exc}). "
            "Using constant fed_rate=4.5 for range."
        )
        fed = pd.DataFrame(
            {
                "date": pd.date_range(
                    args.start, args.end, freq="D"
                ).normalize(),
                "fed_rate": 4.5,
            }
        )

    # ---- Step 3: interest rates (World Bank + interest_rate.csv) ----
    print("\n--- Fetching Vietnam deposit interest rates ---")
    wb_rates = fetch_worldbank_vn_deposit_rate(
        session,
        start_year=pd.Timestamp(args.start).year,
        end_year=pd.Timestamp(args.end).year,
    )
    interest_csv_path = Path(args.interest_file)
    if interest_csv_path.exists():
        print(f"[INFO] Will use {interest_csv_path} for recent state/market split")
    else:
        print(f"[INFO] {interest_csv_path} not found; using World Bank only")

    # ---- Step 3b: extract XAUUSD world gold price from GOLD_PRICE.csv ----
    xauusd_df = extract_xauusd_from_gold_price(Path(args.gold_file))

    # ---- Step 4: merge gold sources ----
    print("\n--- Merging gold data sources ---")
    gold_df = merge_all_gold_sources(
        giavang_df=giavang_df,
        scraped_path=Path(args.scraped_file),
        gold_price_path=Path(args.gold_file),
        start_date=args.start,
        end_date=args.end,
    )

    if gold_df.empty:
        print("[ERROR] No gold data available. Cannot build master dataset.")
        return

    # ---- Step 5: build full master dataset ----
    print("\n--- Building master dataset ---")
    master = build_full_master(
        gold_df=gold_df,
        usd_vnd_df=usd_vnd,
        dxy_df=dxy,
        fed_df=fed,
        world_gold_df=world,
        wb_rates=wb_rates,
        interest_csv_path=interest_csv_path,
        xauusd_df=xauusd_df,
    )

    # ---- Step 6: save ----
    output_path = Path(args.output)
    master.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"\n[DONE] Saved: {output_path} "
        f"({len(master)} rows, "
        f"{master['timestamp'].min()} -> {master['timestamp'].max()})"
    )
    print(f"[DONE] Gold codes: {sorted(master['gold_code'].unique())}")
    print(f"[DONE] Columns: {list(master.columns)}")

    # ---- Data source documentation ----
    print("\n" + "=" * 70)
    print("DATA SOURCES FOR EACH ATTRIBUTE")
    print("=" * 70)
    print("timestamp          : Date of observation")
    print("gold_code          : Gold product code (BT9999NTT, BTSJC, ...)")
    print("buy_price          : Domestic gold buy price (VND)")
    print("  2009-07 -> ~2024 : giavang.org (SJC historical)")
    print("  2025-01 -> 2026-01: GOLD_PRICE_1_YEAR_SCRAPED.csv")
    print("  2026-01 -> present: GOLD_PRICE.csv (multi-brand)")
    print("sell_price         : Domestic gold sell price (VND)")
    print("  (same sources as buy_price)")
    print("usd_vnd_rate       : USD/VND exchange rate")
    print("  Source: Yahoo Finance (VND=X)")
    print("interest_rate_state: Vietnam deposit interest rate (% p.a.)")
    print("  2009-2023: World Bank FR.INR.DPST (IMF IFS), yearly")
    print("    API: api.worldbank.org/v2/country/VN/indicator/FR.INR.DPST")
    print("  2024+: interest_rate.csv from webgia.com (state=AVG Big4)")
    print("interest_rate_market: Vietnam market deposit rate (% p.a.)")
    print("  2009-2023: = interest_rate_state (no separate source)")
    print("  2024+: interest_rate.csv from webgia.com (market=TOP3 trusted)")
    print("interest_rate_spread: = market - state (calculated)")
    print("dxy_index          : US Dollar Index")
    print("  Source: Yahoo Finance (DX-Y.NYB)")
    print("fed_rate           : US Federal Funds Rate")
    print("  Source: FRED (DFF series)")
    print("World_Price_USD_Ounce: World gold spot price (USD/oz)")
    print("  2009 -> 2026-01  : Yahoo Finance (GC=F gold futures)")
    print("  2026-01 -> present: XAUUSD from GOLD_PRICE.csv (preferred)")
    print("World_Price_VND    : = World_Price_USD_Ounce * usd_vnd * 1.20565")
    print("Domestic_Premium   : = sell_price - World_Price_VND")
    print("MA_7, MA_30        : Rolling mean of sell_price (7/30 days)")
    print("RSI_14             : Relative Strength Index (14 days)")
    print("Momentum_*_Pct     : % change over 1/3/7 days")
    print("Target_Price_Next_Shift: Next day's sell_price")
    print("Target_Trend       : 1 if next day price > today, else 0")
    print("=" * 70)


if __name__ == "__main__":
    main()
