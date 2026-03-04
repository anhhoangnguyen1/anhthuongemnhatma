from __future__ import annotations

import re
import unicodedata
from functools import reduce
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"
OUTPUT_FILE = "master_dss_dataset.csv"


def strip_accents(value: object) -> str:
    """Return a lowercase-friendly text representation while preserving digits."""
    text = str(value).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def canonicalize_name(value: object) -> str:
    """Convert arbitrary labels to a stable ASCII snake_case key."""
    text = strip_accents(value)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text.lower()).strip("_")
    return text


def read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    """Handle common CSV encodings from spreadsheets and local exports."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def build_workbook_catalog(base_dir: Path) -> list[dict[str, object]]:
    """Collect workbook paths and their canonicalized sheet names once."""
    catalog: list[dict[str, object]] = []
    for path in sorted(base_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        try:
            excel_file = pd.ExcelFile(path)
        except Exception:
            continue

        sheet_map = {canonicalize_name(sheet): sheet for sheet in excel_file.sheet_names}
        catalog.append(
            {
                "path": path,
                "stem": canonicalize_name(path.stem),
                "sheet_map": sheet_map,
            }
        )
    return catalog


def locate_tabular_source(
    base_dir: Path,
    file_hints: Iterable[str],
    sheet_hints: Iterable[str] = (),
) -> tuple[Path, str | None]:
    """
    Resolve a CSV first, then an Excel sheet fallback, using canonicalized names.
    """
    normalized_file_hints = [canonicalize_name(hint) for hint in file_hints]
    normalized_sheet_hints = [canonicalize_name(hint) for hint in sheet_hints]

    for path in sorted(base_dir.glob("*.csv")):
        stem = canonicalize_name(path.stem)
        if any(hint == stem or hint in stem for hint in normalized_file_hints):
            return path, None

    workbook_catalog = build_workbook_catalog(base_dir)
    best_match: tuple[int, int, Path, str] | None = None

    for workbook in workbook_catalog:
        path = workbook["path"]
        stem = workbook["stem"]
        sheet_map = workbook["sheet_map"]

        for priority, sheet_hint in enumerate(normalized_sheet_hints):
            if sheet_hint not in sheet_map:
                continue

            file_score = 1 if any(hint in stem for hint in normalized_file_hints) else 0
            candidate = (file_score, -priority, path, sheet_map[sheet_hint])
            if best_match is None or candidate > best_match:
                best_match = candidate

    if best_match is not None:
        _, _, path, sheet_name = best_match
        return path, sheet_name

    sources = ", ".join([*normalized_file_hints, *normalized_sheet_hints])
    raise FileNotFoundError(f"Could not locate a source for: {sources}")


def load_table(base_dir: Path, file_hints: Iterable[str], sheet_hints: Iterable[str] = ()) -> pd.DataFrame:
    """Load either a standalone CSV or a sheet from an Excel workbook."""
    path, sheet_name = locate_tabular_source(base_dir, file_hints=file_hints, sheet_hints=sheet_hints)
    if path.suffix.lower() == ".csv":
        df = read_csv_with_fallbacks(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)

    df = df.copy()
    df.columns = [canonicalize_name(column) for column in df.columns]
    return df


def rename_first_match(
    df: pd.DataFrame,
    target_column: str,
    candidate_columns: Iterable[str],
    required: bool = True,
) -> pd.DataFrame:
    """Rename the first matching source column to a stable target column."""
    normalized_candidates = [canonicalize_name(name) for name in candidate_columns]

    if target_column in df.columns:
        return df

    for source_column in normalized_candidates:
        if source_column in df.columns:
            return df.rename(columns={source_column: target_column})

    if required:
        raise KeyError(f"Missing expected columns for '{target_column}': {normalized_candidates}")
    return df


def normalize_timestamp_value(value: object) -> pd.Timestamp:
    """Parse timestamps and drop timezone information after aligning to UTC+7."""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT

    if isinstance(timestamp, pd.Timestamp) and timestamp.tzinfo is not None:
        return timestamp.tz_convert(LOCAL_TIMEZONE).tz_localize(None)

    return pd.Timestamp(timestamp)


def normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the unified timestamp column to timezone-naive pandas timestamps."""
    df = rename_first_match(
        df,
        target_column="timestamp",
        candidate_columns=("timestamp", "datetime", "snapshot_time", "snapshort_time", "updated_at_vn", "thoi_diem_cap_nhat_du_lieu"),
    )
    df["timestamp"] = df["timestamp"].apply(normalize_timestamp_value)
    df = df.dropna(subset=["timestamp"]).copy()
    return df


def parse_numeric_like(series: pd.Series) -> pd.Series:
    """Convert spreadsheet-style numeric text such as '0.5 trieu' to floats."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype("string").map(strip_accents).str.lower().str.strip()
    cleaned = cleaned.str.replace(",", "", regex=False)

    multiplier = pd.Series(1.0, index=series.index, dtype="float64")
    multiplier.loc[cleaned.str.contains("trieu", na=False)] = 1_000_000.0
    multiplier.loc[cleaned.str.contains("ty", na=False)] = 1_000_000_000.0

    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+)")[0]
    numeric = pd.to_numeric(extracted, errors="coerce")
    return numeric * multiplier


def standardize_gold_frame(
    df: pd.DataFrame,
    feature_aliases: dict[str, Iterable[str]],
    require_gold_code: bool = True,
) -> pd.DataFrame:
    """Rename keys, keep only the requested fields, and coerce data types."""
    df = normalize_timestamp_column(df)

    if require_gold_code:
        df = rename_first_match(
            df,
            target_column="gold_code",
            candidate_columns=("gold_code", "ma_vang"),
        )
        df["gold_code"] = df["gold_code"].astype("string").str.strip()
        df = df.dropna(subset=["gold_code"]).copy()

    for target_column, candidate_columns in feature_aliases.items():
        df = rename_first_match(df, target_column=target_column, candidate_columns=candidate_columns)

    keep_columns = ["timestamp"]
    if require_gold_code:
        keep_columns.append("gold_code")
    keep_columns.extend(feature_aliases.keys())

    df = df.loc[:, [column for column in keep_columns if column in df.columns]].copy()

    for column in feature_aliases:
        if column in df.columns:
            df[column] = parse_numeric_like(df[column])

    df = df.drop_duplicates(subset=keep_columns, keep="last").reset_index(drop=True)
    return df


def load_price_data(base_dir: Path) -> pd.DataFrame:
    raw_df = load_table(
        base_dir,
        file_hints=("gold_price", "bang_gia_vang"),
        sheet_hints=("gold_price", "gold_price_clean"),
    )
    return standardize_gold_frame(
        raw_df,
        feature_aliases={
            "buy_price": ("buy_price", "gia_mua"),
            "sell_price": ("sell_price", "gia_ban"),
        },
        require_gold_code=True,
    )


def load_technical_data(base_dir: Path) -> pd.DataFrame:
    raw_df = load_table(
        base_dir,
        file_hints=("technical_indicators",),
        sheet_hints=("technical_indicators",),
    )
    return standardize_gold_frame(
        raw_df,
        feature_aliases={
            "price_change_pct": ("price_change_pct",),
            "volatility_index": ("volatility_index",),
        },
        require_gold_code=True,
    )


def load_market_factors(base_dir: Path) -> pd.DataFrame:
    raw_df = load_table(
        base_dir,
        file_hints=("vn_market_factors",),
        sheet_hints=("vn_market_factors",),
    )
    return standardize_gold_frame(
        raw_df,
        feature_aliases={
            "world_price_vnd": ("world_price_vnd",),
            "domestic_premium": ("domestic_premium",),
        },
        require_gold_code=True,
    )


def load_sentiment_data(base_dir: Path) -> pd.DataFrame:
    raw_df = load_table(
        base_dir,
        file_hints=("sentiment", "clean"),
        sheet_hints=("clean",),
    )
    return standardize_gold_frame(
        raw_df,
        feature_aliases={
            "news_volume": ("news_volume",),
            "sentiment_score": ("sentiment_score",),
        },
        require_gold_code=True,
    )


def load_macro_data(base_dir: Path, file_hint: str, sheet_hint: str, feature_columns: dict[str, Iterable[str]]) -> pd.DataFrame:
    raw_df = load_table(base_dir, file_hints=(file_hint,), sheet_hints=(sheet_hint,))
    return standardize_gold_frame(raw_df, feature_aliases=feature_columns, require_gold_code=False)


def build_base_gold_dataframe(base_dir: Path) -> pd.DataFrame:
    gold_frames = [
        load_price_data(base_dir),
        load_technical_data(base_dir),
        load_market_factors(base_dir),
        load_sentiment_data(base_dir),
    ]

    base_df = reduce(
        lambda left, right: pd.merge(left, right, on=["timestamp", "gold_code"], how="outer"),
        gold_frames,
    )
    base_df = base_df.sort_values(["timestamp", "gold_code"], kind="mergesort").reset_index(drop=True)
    return base_df


def merge_macro_asof(base_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast each macro series by taking the last known value at each gold snapshot."""
    if macro_df.empty:
        return base_df

    left = base_df.sort_values("timestamp").reset_index(drop=True)
    right = macro_df.sort_values("timestamp").reset_index(drop=True)

    merged_df = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
    )
    return merged_df


def apply_group_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill within each gold code to smooth sparse observations."""
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)

    fill_columns = [column for column in df.columns if column not in {"timestamp", "gold_code"}]
    if fill_columns:
        df[fill_columns] = df.groupby("gold_code", dropna=False)[fill_columns].ffill()

    critical_price_columns = [column for column in ("buy_price", "sell_price") if column in df.columns]
    if critical_price_columns:
        df = df.dropna(subset=critical_price_columns).copy()

    return df.reset_index(drop=True)


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Use the previous observation within each gold series as the trend baseline."""
    price_priority = ("sell_price", "buy_price")
    price_column = next((column for column in price_priority if column in df.columns), None)

    if price_column is None:
        raise KeyError("No usable price column was found to build target_trend.")

    previous_price = df.groupby("gold_code", dropna=False)[price_column].shift(1)
    df["target_trend"] = np.where(df[price_column] > previous_price, 1, 0).astype("int8")
    return df


def finalize_master_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only timestamp, gold_code, and numeric model features."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column != "target_trend"]

    if feature_columns:
        imputer = SimpleImputer(strategy="median")
        df[feature_columns] = imputer.fit_transform(df[feature_columns])

    ordered_columns = ["timestamp", "gold_code"] + [column for column in df.columns if column in numeric_columns]
    ordered_columns = [column for column in ordered_columns if column in df.columns]
    final_df = df.loc[:, ordered_columns].copy()
    final_df = final_df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    return final_df


def build_master_dataframe(base_dir: Path) -> pd.DataFrame:
    """End-to-end pipeline from raw source files to the DSS master dataset."""
    master_df = build_base_gold_dataframe(base_dir)

    macro_frames = [
        load_macro_data(
            base_dir,
            file_hint="usd_vnd_rate_live",
            sheet_hint="usd_vnd_rate_live",
            feature_columns={"usd_vnd_rate": ("usd_vnd_rate",)},
        ),
        load_macro_data(
            base_dir,
            file_hint="fed_rate_live",
            sheet_hint="fed_rate_live",
            feature_columns={"fed_rate": ("fed_rate",)},
        ),
        load_macro_data(
            base_dir,
            file_hint="cpi_live",
            sheet_hint="cpi_live",
            feature_columns={"cpi_inflation_yoy": ("cpi_inflation_yoy",)},
        ),
        load_macro_data(
            base_dir,
            file_hint="dxy_history",
            sheet_hint="dxy_history",
            feature_columns={"dxy_index": ("dxy_index",)},
        ),
        load_macro_data(
            base_dir,
            file_hint="interest_rate",
            sheet_hint="interest_rate",
            feature_columns={
                "interest_rate_state": ("interest_rate_state",),
                "interest_rate_market": ("interest_rate_market",),
            },
        ),
    ]

    for macro_df in macro_frames:
        master_df = merge_macro_asof(master_df, macro_df)

    master_df = apply_group_forward_fill(master_df)
    master_df = create_target_variable(master_df)
    master_df = finalize_master_dataframe(master_df)
    return master_df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    master_df = build_master_dataframe(base_dir)

    output_path = base_dir / OUTPUT_FILE
    master_df.to_csv(output_path, index=False)

    print("\nFinal Master DataFrame .info():")
    master_df.info()
    print("\nFinal Master DataFrame .head():")
    print(master_df.head())
    print("\nMissing values by column:")
    print(master_df.isna().sum())
    print(f"\nSaved master dataset to: {output_path}")


if __name__ == "__main__":
    main()
