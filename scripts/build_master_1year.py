#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chuẩn hóa GOLD_PRICE_1_YEAR_SCRAPED.csv + MACRO_FEATURES_1_YEAR.csv thành thư mục input
cho prepare_gold_dss_pipeline (GOLD_PRICE.csv có domestic + XAUUSD, usd_vnd, dxy, fed, interest).
Chạy fetch_macro_1_year.py trước để có MACRO_FEATURES_1_YEAR.csv.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


WORLD_GOLD_CODE = "XAUUSD"


def normalize_col(text: object) -> str:
    s = str(text).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s


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


def read_csv_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(r"[^\d,.\-]", "", regex=True)
    )
    out = pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build input_1year from scraped gold + macro CSV for prepare_gold_dss_pipeline."
    )
    parser.add_argument(
        "--scraped",
        type=Path,
        default=Path("GOLD_PRICE_1_YEAR_SCRAPED.csv"),
        help="Scraped gold CSV (Ngày, Mã vàng, Giá mua, Giá bán)",
    )
    parser.add_argument(
        "--macro",
        type=Path,
        default=Path("MACRO_FEATURES_1_YEAR.csv"),
        help="Macro daily CSV from fetch_macro_1_year.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input_1year"),
        help="Output directory for GOLD_PRICE.csv and macro CSVs",
    )
    args = parser.parse_args()

    if not args.scraped.exists():
        raise FileNotFoundError(f"Scraped file not found: {args.scraped}")
    if not args.macro.exists():
        raise FileNotFoundError(
            f"Macro file not found: {args.macro}. Run: python scripts/fetch_macro_1_year.py --start 2025-01-25 --end 2026-01-25"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Domestic gold from scraped ---
    raw = read_csv_fallback(args.scraped)
    date_col = resolve_col(raw, ["Ngày", "ngay", "date", "timestamp"])
    code_col = resolve_col(raw, ["Mã vàng", "ma_vang", "gold_code"])
    buy_col = resolve_col(raw, ["Giá mua", "gia_mua", "buy_price"])
    sell_col = resolve_col(raw, ["Giá bán", "gia_ban", "sell_price"])
    if not date_col or not code_col or not buy_col or not sell_col:
        raise KeyError(
            f"Cannot resolve columns. Scraped columns: {list(raw.columns)}. "
            "Need: Ngày, Mã vàng, Giá mua, Giá bán."
        )

    domestic = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[date_col], errors="coerce").dt.normalize(),
        "gold_code": raw[code_col].astype(str).str.strip().str.upper(),
        "buy_price": parse_numeric(raw[buy_col]),
        "sell_price": parse_numeric(raw[sell_col]),
    })
    domestic = domestic.dropna(subset=["timestamp", "gold_code", "sell_price"])
    domestic = domestic.drop_duplicates(subset=["timestamp", "gold_code"], keep="last")
    domestic = domestic.sort_values(["timestamp", "gold_code"], kind="mergesort").reset_index(drop=True)
    print(f"[OK] Domestic gold: {len(domestic)} rows ({domestic['timestamp'].min().date()} -> {domestic['timestamp'].max().date()})")

    # --- World gold (XAUUSD) from macro ---
    macro = read_csv_fallback(args.macro)
    m_date = resolve_col(macro, ["date", "timestamp", "ngay"])
    m_world = resolve_col(macro, ["World_Price_USD_Ounce", "world_price_usd_ounce"])
    if not m_date or not m_world:
        raise KeyError(f"Macro must have 'date' and 'World_Price_USD_Ounce'. Columns: {list(macro.columns)}")
    macro["__date__"] = pd.to_datetime(macro[m_date], errors="coerce").dt.normalize()
    macro["__world__"] = pd.to_numeric(macro[m_world], errors="coerce")
    macro = macro.dropna(subset=["__date__", "__world__"]).drop_duplicates(subset=["__date__"], keep="last")
    world = pd.DataFrame({
        "timestamp": macro["__date__"].values,
        "gold_code": WORLD_GOLD_CODE,
        "buy_price": macro["__world__"].values,
        "sell_price": macro["__world__"].values,
    })
    print(f"[OK] World (XAUUSD): {len(world)} rows")

    # --- Combined gold ---
    gold = pd.concat([domestic, world], ignore_index=True)
    gold = gold.sort_values(["timestamp", "gold_code"], kind="mergesort").reset_index(drop=True)
    gold_path = args.output_dir / "GOLD_PRICE.csv"
    gold.to_csv(gold_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] {gold_path} ({len(gold)} rows)")

    # --- Macro tables (pipeline expects timestamp column); resolve from macro (has original CSV columns) ---
    macro_work = macro.copy()
    macro_work["timestamp"] = macro_work["__date__"]

    usd_col = resolve_col(macro_work, ["usd_vnd_rate", "usd_vnd"])
    if usd_col:
        usd = macro_work[["timestamp", usd_col]].rename(columns={usd_col: "usd_vnd_rate"})
        usd.to_csv(args.output_dir / "usd_vnd_rate_live.csv", index=False, encoding="utf-8-sig")
        print(f"[DONE] {args.output_dir / 'usd_vnd_rate_live.csv'}")

    dxy_col = resolve_col(macro_work, ["dxy_index", "dxy"])
    if dxy_col:
        dxy = macro_work[["timestamp", dxy_col]].rename(columns={dxy_col: "dxy_index"})
        dxy.to_csv(args.output_dir / "dxy_history.csv", index=False, encoding="utf-8-sig")
        print(f"[DONE] {args.output_dir / 'dxy_history.csv'}")

    fed_col = resolve_col(macro_work, ["fed_rate", "fed_funds_rate"])
    if fed_col:
        fed = macro_work[["timestamp", fed_col]].rename(columns={fed_col: "fed_rate"})
        fed.to_csv(args.output_dir / "fed_rate_live.csv", index=False, encoding="utf-8-sig")
        print(f"[DONE] {args.output_dir / 'fed_rate_live.csv'}")

    i_state = resolve_col(macro_work, ["interest_rate_state"])
    i_market = resolve_col(macro_work, ["interest_rate_market"])
    i_spread = resolve_col(macro_work, ["interest_rate_spread"])
    if i_state and i_market:
        interest = macro_work[["timestamp"]].copy()
        interest["interest_rate_state"] = pd.to_numeric(macro_work[i_state], errors="coerce")
        interest["interest_rate_market"] = pd.to_numeric(macro_work[i_market], errors="coerce")
        if i_spread:
            interest["interest_rate_spread"] = pd.to_numeric(macro_work[i_spread], errors="coerce")
        else:
            interest["interest_rate_spread"] = interest["interest_rate_market"] - interest["interest_rate_state"]
        interest = interest.dropna(subset=["interest_rate_state", "interest_rate_market"])
        interest.to_csv(args.output_dir / "interest_rate.csv", index=False, encoding="utf-8-sig")
        print(f"[DONE] {args.output_dir / 'interest_rate.csv'}")

    print("\n[INFO] Next steps:")
    print(f"  python prepare_gold_dss_pipeline.py --output-file \"New folder/master_dss_dataset.csv\"")
    print("  cd \"New folder\" ; python train_xgboost_dss.py")


if __name__ == "__main__":
    main()
