from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"
WORLD_GOLD_CODE = "XAUUSD"
OUNCE_TO_TAEL_FACTOR = 1.20565

DEFAULT_INPUT_FILES = {
    "gold": "GOLD_PRICE.csv",
    "usd_vnd": "usd_vnd_rate_live.csv",
    "interest": "interest_rate.csv",
    "dxy_primary": "dxy_history.csv",
    "dxy_alternative": "dxy_index_rate_one_year.csv",
    "fed": "fed_rate_live.csv",
    # Optional: daily news sentiment (from fetch_news_sentiment_marketaux.py)
    "news": "NEWS_SENTIMENT.csv",
    # Optional: daily news impact from LLM (from assess_news_impact_llm.py) — 1 cột mạnh thay vì nhồi tin thô
    "news_impact": "NEWS_IMPACT_DAILY.csv",
}

DEFAULT_OUTPUT_FILE = "master_dss_dataset.csv"


def normalize_text(value: object) -> str:
    """Normalize headers to ASCII snake_case for robust alias matching."""
    text = str(value).strip()

    # Repair common mojibake/encoding artifacts before normalization.
    text = (
        text.replace("Ä", "Đ")
        .replace("Ä‘", "đ")
        .replace("Ð", "Đ")
        .replace("ð", "đ")
        .replace("đ", "d")
        .replace("Đ", "D")
    )

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return text


def read_csv_with_fallback_encodings(path: Path) -> pd.DataFrame:
    """Read CSV with fallback encodings commonly seen in Excel exports."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Resolve one actual column name from candidate aliases."""
    normalized_to_actual: dict[str, str] = {}
    for column in df.columns:
        normalized_to_actual[normalize_text(column)] = column

    normalized_candidates = [normalize_text(name) for name in candidates]
    for candidate in normalized_candidates:
        if candidate in normalized_to_actual:
            return normalized_to_actual[candidate]

    # Relaxed contains matching to survive slightly noisy names.
    for candidate in normalized_candidates:
        for normalized_column, actual_column in normalized_to_actual.items():
            if candidate and candidate in normalized_column:
                return actual_column

    return None


def rename_by_aliases(
    df: pd.DataFrame,
    alias_map: dict[str, Iterable[str]],
    required_targets: Iterable[str],
    table_name: str,
) -> pd.DataFrame:
    """Rename source columns to standardized names based on alias lists."""
    rename_map: dict[str, str] = {}
    missing_targets: list[str] = []

    for target, candidates in alias_map.items():
        source = resolve_column(df, candidates)
        if source is None:
            if target in required_targets:
                missing_targets.append(target)
            continue
        rename_map[source] = target

    if missing_targets:
        missing = ", ".join(sorted(missing_targets))
        raise KeyError(f"[{table_name}] Missing required columns: {missing}")

    return df.rename(columns=rename_map)


def parse_datetime_value(value: object, local_timezone: str = LOCAL_TIMEZONE) -> pd.Timestamp:
    """
    Parse one datetime value and normalize to a timezone-naive local timestamp.

    - If input is timezone-aware, convert to local timezone.
    - If input is naive, assume it is local timezone.
    - Final output is tz-naive `datetime64[ns]` for merge_asof compatibility.
    """
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT

    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(local_timezone, ambiguous="NaT", nonexistent="shift_forward")
    else:
        timestamp = timestamp.tz_convert(local_timezone)

    return timestamp.tz_localize(None)


def parse_datetime_series(series: pd.Series, local_timezone: str = LOCAL_TIMEZONE) -> pd.Series:
    return series.map(lambda value: parse_datetime_value(value, local_timezone=local_timezone))


def parse_numeric_value(value: object) -> float:
    """Parse flexible numeric text: 95.500.000 / 2,945.8 / 0.0 triệu."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "":
        return np.nan

    text = text.replace("\xa0", "")
    text = re.sub(r"[^\d,.\-]", "", text)
    if text == "":
        return np.nan

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        if text.count(",") == 1:
            left, right = text.split(",", 1)
            text = f"{left}.{right}" if len(right) <= 2 else left + right
        else:
            text = text.replace(",", "")
    elif "." in text and text.count(".") > 1:
        text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return series.map(parse_numeric_value).astype("float64")


def prepare_table(
    path: Path,
    alias_map: dict[str, Iterable[str]],
    required_targets: Iterable[str],
    keep_columns: Iterable[str],
    numeric_columns: Iterable[str],
    table_name: str,
    local_timezone: str = LOCAL_TIMEZONE,
    dedupe_subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load one table, rename columns, parse datatypes, and sort chronologically."""
    df = read_csv_with_fallback_encodings(path).copy()
    df = rename_by_aliases(df, alias_map=alias_map, required_targets=required_targets, table_name=table_name)

    df["timestamp"] = parse_datetime_series(df["timestamp"], local_timezone=local_timezone)
    df = df.dropna(subset=["timestamp"]).copy()

    for column in numeric_columns:
        if column in df.columns:
            df[column] = parse_numeric_series(df[column])

    selected_columns = [column for column in keep_columns if column in df.columns]
    df = df.loc[:, selected_columns].copy()

    if dedupe_subset is not None:
        subset = [column for column in dedupe_subset if column in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="last")

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def merge_latest_asof(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    right_value_columns: Iterable[str],
    left_time_col: str = "timestamp",
    right_time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Merge latest known values with `merge_asof(..., direction="backward")`.

    `direction="backward"` prevents look-ahead bias because each left timestamp
    can only see right-side records that already existed at or before that time.
    """
    value_columns = [column for column in right_value_columns if column in right_df.columns]
    if not value_columns:
        return left_df

    # merge_asof requires both sides to be sorted by merge key.
    left_sorted = left_df.sort_values(left_time_col, kind="mergesort").reset_index(drop=True)
    right_sorted = (
        right_df[[right_time_col, *value_columns]]
        .dropna(subset=[right_time_col])
        .sort_values(right_time_col, kind="mergesort")
        .drop_duplicates(subset=[right_time_col], keep="last")
        .reset_index(drop=True)
    )

    merged_df = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_time_col,
        right_on=right_time_col,
        direction="backward",
        allow_exact_matches=True,
    )

    if right_time_col != left_time_col and right_time_col in merged_df.columns:
        merged_df = merged_df.drop(columns=[right_time_col])
    return merged_df


def fill_macro_columns_no_na(df: pd.DataFrame, macro_columns: Iterable[str]) -> pd.DataFrame:
    """
    Fill macro columns by `ffill` then `bfill` within each gold series.

    This removes leading NaN (before first macro update) and normal gaps.
    """
    macro_columns = [column for column in macro_columns if column in df.columns]
    if not macro_columns:
        return df

    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    df[macro_columns] = df.groupby("gold_code", dropna=False)[macro_columns].ffill().bfill()

    missing_counts = df[macro_columns].isna().sum()
    remaining_missing = missing_counts[missing_counts > 0]
    if not remaining_missing.empty:
        raise ValueError(
            "NaN values still exist in macro columns after ffill+bfill: "
            + remaining_missing.to_dict().__str__()
        )

    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def build_world_price_series(df_world_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a robust world-price timeline for XAUUSD mapping.

    Primary source follows request: `Giá bán` (`sell_price`).
    In this data source, many XAUUSD sell prices are zero, so we treat non-positive
    values as invalid and fallback to `buy_price` to keep a usable world series.
    """
    required_columns = {"timestamp", "sell_price", "buy_price"}
    missing = required_columns - set(df_world_raw.columns)
    if missing:
        raise KeyError(f"df_world is missing required columns: {sorted(missing)}")

    df_world = df_world_raw.loc[:, ["timestamp", "sell_price", "buy_price"]].copy()
    df_world["World_Price_USD_Ounce"] = pd.to_numeric(df_world["sell_price"], errors="coerce").astype("float64")

    invalid_sell_mask = df_world["World_Price_USD_Ounce"].isna() | df_world["World_Price_USD_Ounce"].le(0)
    fallback_buy = pd.to_numeric(df_world["buy_price"], errors="coerce").astype("float64")
    fallback_buy = fallback_buy.where(fallback_buy.gt(0))
    df_world.loc[invalid_sell_mask, "World_Price_USD_Ounce"] = fallback_buy.loc[invalid_sell_mask]

    df_world = df_world.dropna(subset=["timestamp", "World_Price_USD_Ounce"]).copy()
    df_world = df_world[df_world["World_Price_USD_Ounce"] > 0].copy()
    df_world = (
        df_world.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    if df_world.empty:
        raise ValueError("World price series is empty after dropping invalid XAUUSD prices.")

    return df_world[["timestamp", "World_Price_USD_Ounce"]]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped technical indicators and momentum features."""
    try:
        from ta.momentum import RSIIndicator
    except ImportError as exc:
        raise ImportError("Missing dependency 'ta'. Install via: pip install ta") from exc

    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped_sell = df.groupby("gold_code", dropna=False)["sell_price"]

    df["MA_7"] = grouped_sell.transform(lambda s: s.rolling(window=7, min_periods=7).mean())
    df["MA_30"] = grouped_sell.transform(lambda s: s.rolling(window=30, min_periods=30).mean())
    df["RSI_14"] = grouped_sell.transform(
        lambda s: RSIIndicator(close=s.astype("float64"), window=14, fillna=False).rsi()
    )
    df["Momentum_1D_Pct"] = grouped_sell.transform(lambda s: s.pct_change(periods=1) * 100.0)
    df["Momentum_3D_Pct"] = grouped_sell.transform(lambda s: s.pct_change(periods=3) * 100.0)
    df["Momentum_7D_Pct"] = grouped_sell.transform(lambda s: s.pct_change(periods=7) * 100.0)
    return df


def add_supervised_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target columns for XGBoost:
    - Target_Price_Next_Shift = next sell_price in the same gold_code series.
    - Target_Trend = 1 if next price > current price, else 0.
    """
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped_sell = df.groupby("gold_code", dropna=False)["sell_price"]

    df["Target_Price_Next_Shift"] = grouped_sell.shift(-1)
    df["Target_Trend"] = (df["Target_Price_Next_Shift"] > df["sell_price"]).astype("int8")

    # Drop last row of each gold series where next-shift target is unavailable.
    df = df.dropna(subset=["Target_Price_Next_Shift"]).reset_index(drop=True)
    df["Target_Trend"] = df["Target_Trend"].astype("int8")
    return df


def resolve_input_paths(input_dir: Path) -> dict[str, Path]:
    """Resolve required source paths with DXY fallback filename support."""
    paths = {
        "gold": input_dir / DEFAULT_INPUT_FILES["gold"],
        "usd_vnd": input_dir / DEFAULT_INPUT_FILES["usd_vnd"],
        "interest": input_dir / DEFAULT_INPUT_FILES["interest"],
        "fed": input_dir / DEFAULT_INPUT_FILES["fed"],
    }

    dxy_primary = input_dir / DEFAULT_INPUT_FILES["dxy_primary"]
    dxy_alternative = input_dir / DEFAULT_INPUT_FILES["dxy_alternative"]
    if dxy_primary.exists():
        paths["dxy"] = dxy_primary
    elif dxy_alternative.exists():
        paths["dxy"] = dxy_alternative
    else:
        raise FileNotFoundError(
            f"DXY file not found. Expected one of: {dxy_primary.name}, {dxy_alternative.name}"
        )

    missing_paths = [str(path) for key, path in paths.items() if key not in ("dxy",) and not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required input file(s): {missing_paths}")

    # Optional news sentiment file
    news_path = input_dir / DEFAULT_INPUT_FILES["news"]
    if news_path.exists():
        paths["news"] = news_path
    # Optional news impact (LLM-assessed)
    impact_path = input_dir / DEFAULT_INPUT_FILES["news_impact"]
    if impact_path.exists():
        paths["news_impact"] = impact_path

    return paths


def build_master_dataset(input_dir: Path, local_timezone: str = LOCAL_TIMEZONE) -> pd.DataFrame:
    """End-to-end DSS preprocessing pipeline with robust world + macro + target fixes."""
    paths = resolve_input_paths(input_dir=input_dir)
    logging.info("Input files resolved: %s", {k: v.name for k, v in paths.items()})

    gold_aliases = {
        "timestamp": ("timestamp", "datetime", "ngay", "date", "updated_at_vn", "cap_nhat_du_lieu", "thoi_diem_cap_nhat_du_lieu"),
        "gold_code": ("gold_code", "ma_vang"),
        "buy_price": ("buy_price", "gia_mua"),
        "sell_price": ("sell_price", "gia_ban"),
    }
    usd_aliases = {
        "timestamp": ("timestamp", "datetime", "date", "updated_at_vn"),
        "usd_vnd_rate": ("usd_vnd_rate", "usd_vnd", "usd_vnd_exchange_rate"),
    }
    interest_aliases = {
        "timestamp": ("timestamp", "datetime", "date", "updated_at_vn"),
        "interest_rate_state": ("interest_rate_state",),
        "interest_rate_market": ("interest_rate_market",),
        "interest_rate_spread": ("interest_rate_spread", "spread", "interest_spread"),
    }
    dxy_aliases = {
        "timestamp": ("timestamp", "datetime", "date", "updated_at_vn"),
        "dxy_index": ("dxy_index", "dxy", "us_dollar_index"),
    }
    fed_aliases = {
        "timestamp": ("timestamp", "datetime", "date", "updated_at_vn"),
        "fed_rate": ("fed_rate", "fed_funds_rate", "federal_funds_rate"),
    }

    df_gold = prepare_table(
        path=paths["gold"],
        alias_map=gold_aliases,
        required_targets=("timestamp", "gold_code", "buy_price", "sell_price"),
        keep_columns=("timestamp", "gold_code", "buy_price", "sell_price"),
        numeric_columns=("buy_price", "sell_price"),
        table_name="GOLD_PRICE",
        local_timezone=local_timezone,
        dedupe_subset=("timestamp", "gold_code"),
    )
    df_usd = prepare_table(
        path=paths["usd_vnd"],
        alias_map=usd_aliases,
        required_targets=("timestamp", "usd_vnd_rate"),
        keep_columns=("timestamp", "usd_vnd_rate"),
        numeric_columns=("usd_vnd_rate",),
        table_name="usd_vnd_rate_live",
        local_timezone=local_timezone,
        dedupe_subset=("timestamp",),
    )
    df_interest = prepare_table(
        path=paths["interest"],
        alias_map=interest_aliases,
        required_targets=("timestamp", "interest_rate_state", "interest_rate_market", "interest_rate_spread"),
        keep_columns=("timestamp", "interest_rate_state", "interest_rate_market", "interest_rate_spread"),
        numeric_columns=("interest_rate_state", "interest_rate_market", "interest_rate_spread"),
        table_name="interest_rate",
        local_timezone=local_timezone,
        dedupe_subset=("timestamp",),
    )
    df_dxy = prepare_table(
        path=paths["dxy"],
        alias_map=dxy_aliases,
        required_targets=("timestamp", "dxy_index"),
        keep_columns=("timestamp", "dxy_index"),
        numeric_columns=("dxy_index",),
        table_name=paths["dxy"].name,
        local_timezone=local_timezone,
        dedupe_subset=("timestamp",),
    )
    df_fed = prepare_table(
        path=paths["fed"],
        alias_map=fed_aliases,
        required_targets=("timestamp", "fed_rate"),
        keep_columns=("timestamp", "fed_rate"),
        numeric_columns=("fed_rate",),
        table_name="fed_rate_live",
        local_timezone=local_timezone,
        dedupe_subset=("timestamp",),
    )

    # Step 1: split world (XAUUSD) and domestic rows.
    df_gold["gold_code"] = df_gold["gold_code"].astype("string").str.strip().str.upper()
    world_mask = df_gold["gold_code"].eq(WORLD_GOLD_CODE).fillna(False)

    df_world_raw = df_gold.loc[world_mask].copy()
    df_domestic = df_gold.loc[~world_mask].copy()
    df_domestic = df_domestic.dropna(subset=["gold_code", "sell_price"]).copy()
    df_domestic = df_domestic.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Step 2: macro integration using backward as-of (no look-ahead).
    df_domestic = merge_latest_asof(df_domestic, df_usd, right_value_columns=("usd_vnd_rate",))
    df_domestic = merge_latest_asof(
        df_domestic,
        df_interest,
        right_value_columns=("interest_rate_state", "interest_rate_market", "interest_rate_spread"),
    )
    df_domestic = merge_latest_asof(df_domestic, df_dxy, right_value_columns=("dxy_index",))
    df_domestic = merge_latest_asof(df_domestic, df_fed, right_value_columns=("fed_rate",))

    macro_columns = [
        "usd_vnd_rate",
        "dxy_index",
        "fed_rate",
        "interest_rate_state",
        "interest_rate_market",
        "interest_rate_spread",
    ]
    df_domestic = fill_macro_columns_no_na(df_domestic, macro_columns=macro_columns)

    # Optional: merge daily news sentiment (if NEWS_SENTIMENT.csv exists)
    news_path = paths.get("news")
    if news_path is not None and news_path.exists():
        logging.info("Merging news sentiment from %s", news_path.name)
        df_news_raw = read_csv_with_fallback_encodings(news_path).copy()

        date_col = resolve_column(df_news_raw, ("timestamp", "date"))
        if date_col is None:
            raise KeyError("[NEWS] Missing required date/timestamp column.")

        df_news_raw["timestamp"] = parse_datetime_series(df_news_raw[date_col], local_timezone=local_timezone)
        df_news_raw = df_news_raw.dropna(subset=["timestamp"]).copy()

        # Normalize expected column names
        news_aliases = {
            "gold_sentiment": ("gold_sentiment", "sentiment", "avg_sentiment"),
            "gold_news_count": ("gold_news_count", "news_count", "article_count"),
        }
        df_news_raw = rename_by_aliases(
            df_news_raw,
            alias_map=news_aliases,
            required_targets=("gold_sentiment", "gold_news_count"),
            table_name="NEWS_SENTIMENT",
        )

        df_news = df_news_raw[["timestamp", "gold_sentiment", "gold_news_count"]].copy()
        df_domestic = merge_latest_asof(
            df_domestic,
            df_news,
            right_value_columns=("gold_sentiment", "gold_news_count"),
        )

    # Optional: merge daily news impact from LLM (NEWS_IMPACT_DAILY.csv) — 1 cột news_impact
    impact_path = paths.get("news_impact")
    if impact_path is not None and impact_path.exists():
        logging.info("Merging news impact from %s", impact_path.name)
        df_imp_raw = read_csv_with_fallback_encodings(impact_path).copy()
        date_col = resolve_column(df_imp_raw, ("timestamp", "date"))
        if date_col is not None and "news_impact" in df_imp_raw.columns:
            df_imp_raw["timestamp"] = parse_datetime_series(df_imp_raw[date_col], local_timezone=local_timezone)
            df_imp_raw = df_imp_raw.dropna(subset=["timestamp", "news_impact"]).copy()
            df_imp = df_imp_raw[["timestamp", "news_impact"]].copy()
            df_domestic = merge_latest_asof(
                df_domestic,
                df_imp,
                right_value_columns=("news_impact",),
            )

    # Step 3: robust world-price mapping and premium recalculation.
    df_world = build_world_price_series(df_world_raw=df_world_raw)
    df_domestic = merge_latest_asof(
        df_domestic,
        df_world,
        right_value_columns=("World_Price_USD_Ounce",),
    )

    # Fill edge cases where domestic rows start before first world quote.
    df_domestic = df_domestic.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    df_domestic["World_Price_USD_Ounce"] = (
        df_domestic.groupby("gold_code", dropna=False)["World_Price_USD_Ounce"].ffill().bfill()
    )
    if df_domestic["World_Price_USD_Ounce"].isna().any():
        raise ValueError("World_Price_USD_Ounce still contains NaN after asof + ffill + bfill.")

    df_domestic["World_Price_VND"] = (
        df_domestic["World_Price_USD_Ounce"] * df_domestic["usd_vnd_rate"] * OUNCE_TO_TAEL_FACTOR
    )
    df_domestic["Domestic_Premium"] = df_domestic["sell_price"] - df_domestic["World_Price_VND"]

    # Keep technical features (existing requirement) and add supervised targets.
    df_domestic = add_technical_features(df_domestic)
    df_domestic = add_supervised_targets(df_domestic)

    # Final strict chronological ordering for downstream training splits.
    df_domestic = df_domestic.sort_values(["timestamp", "gold_code"], kind="mergesort").reset_index(drop=True)
    return df_domestic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned DSS master dataset for gold forecasting.")
    parser.add_argument("--input-dir", type=Path, default=Path("input_1year"), help="Directory containing source CSV files (GOLD_PRICE.csv, usd_vnd_rate_live.csv, ...).")
    parser.add_argument("--output-file", type=Path, default=Path(DEFAULT_OUTPUT_FILE), help="Output CSV path.")
    parser.add_argument(
        "--timezone",
        type=str,
        default=LOCAL_TIMEZONE,
        help="Timezone used to normalize timestamps before converting to tz-naive.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    master_df = build_master_dataset(input_dir=args.input_dir, local_timezone=args.timezone)
    master_df = master_df.rename(columns={"World_Price_VND": "world_price_vnd", "Domestic_Premium": "domestic_premium"})

    output_path = args.output_file if args.output_file.is_absolute() else args.input_dir / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    macro_columns = [
        "usd_vnd_rate",
        "dxy_index",
        "fed_rate",
        "interest_rate_state",
        "interest_rate_market",
        "interest_rate_spread",
    ]
    macro_na = master_df[macro_columns].isna().sum().to_dict()

    logging.info("Rows: %s | Columns: %s", len(master_df), len(master_df.columns))
    logging.info("Macro NaN counts: %s", macro_na)
    logging.info(
        "World_Price_USD_Ounce min/max: %.4f / %.4f",
        master_df["World_Price_USD_Ounce"].min(),
        master_df["World_Price_USD_Ounce"].max(),
    )
    logging.info("Saved output to: %s", output_path.resolve())
    print(master_df.head(10))


if __name__ == "__main__":
    main()
