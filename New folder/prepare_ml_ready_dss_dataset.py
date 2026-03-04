from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "master_dss_dataset.csv"
OUTPUT_FILE = "ml_ready_dss_dataset.csv"


def load_master_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the pre-merged DSS dataset and enforce the expected time ordering."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def clean_price_change_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair clearly invalid percentage values.

    Values outside [-100, 100] are treated as bad records, forward-filled within
    each gold series, and any leading invalid values are replaced with zero.
    """
    if "price_change_pct" not in df.columns:
        return df

    invalid_mask = df["price_change_pct"].gt(100) | df["price_change_pct"].lt(-100)
    df.loc[invalid_mask, "price_change_pct"] = np.nan
    df["price_change_pct"] = df.groupby("gold_code", dropna=False)["price_change_pct"].ffill()
    df["price_change_pct"] = df["price_change_pct"].fillna(0.0)
    return df


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse hourly rows into a daily modeling table.

    Prices and slow macro signals use the final observed value of the day.
    Intraday sentiment and volatility-style signals use the daily mean.
    """
    agg_map: dict[str, str] = {}

    last_value_columns = [
        "buy_price",
        "sell_price",
        "world_price_vnd",
        "domestic_premium",
        "usd_vnd_rate",
        "fed_rate",
        "cpi_inflation_yoy",
        "dxy_index",
        "interest_rate_state",
        "interest_rate_market",
    ]
    mean_value_columns = [
        "price_change_pct",
        "volatility_index",
        "news_volume",
        "sentiment_score",
    ]

    for column in last_value_columns:
        if column in df.columns:
            agg_map[column] = "last"

    for column in mean_value_columns:
        if column in df.columns:
            agg_map[column] = "mean"

    daily_df = (
        df.groupby("gold_code", dropna=False)
        .resample("D", on="timestamp")
        .agg(agg_map)
        .reset_index()
    )

    daily_df = daily_df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)

    if "sell_price" in daily_df.columns:
        daily_df = daily_df.dropna(subset=["sell_price"]).reset_index(drop=True)

    return daily_df


def safe_percentage_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute percentages while suppressing divide-by-zero artifacts."""
    valid_mask = denominator.notna() & denominator.ne(0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid_mask] = (numerator.loc[valid_mask] / denominator.loc[valid_mask]) * 100.0
    return result


def engineer_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create scale-invariant features that are better suited for ML."""
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("gold_code", dropna=False)

    if "sell_price" in df.columns:
        df["daily_return_pct"] = grouped["sell_price"].pct_change() * 100.0
    if "world_price_vnd" in df.columns:
        df["world_return_pct"] = grouped["world_price_vnd"].pct_change() * 100.0

    if {"sell_price", "world_price_vnd"}.issubset(df.columns):
        df["premium_pct"] = safe_percentage_ratio(
            df["sell_price"] - df["world_price_vnd"],
            df["world_price_vnd"],
        )

    if {"sell_price", "buy_price"}.issubset(df.columns):
        df["spread_margin_pct"] = safe_percentage_ratio(
            df["sell_price"] - df["buy_price"],
            df["buy_price"],
        )

    return df


def create_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label whether the next daily close is higher than the current daily close."""
    next_sell_price = df.groupby("gold_code", dropna=False)["sell_price"].shift(-1)
    target = pd.Series(np.nan, index=df.index, dtype="float64")

    valid_mask = next_sell_price.notna() & df["sell_price"].notna()
    target.loc[valid_mask] = (next_sell_price.loc[valid_mask] > df.loc[valid_mask, "sell_price"]).astype("int8")

    df["target_next_day_trend"] = target
    return df


def drop_and_scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove magnitude-sensitive columns, drop incomplete feature rows, and scale
    numeric inputs for downstream ML models.
    """
    columns_to_drop = [
        "buy_price",
        "sell_price",
        "world_price_vnd",
        "domestic_premium",
        "target_trend",
    ]
    existing_drop_columns = [column for column in columns_to_drop if column in df.columns]
    df = df.drop(columns=existing_drop_columns)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column != "target_next_day_trend"]

    if feature_columns:
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

    df["target_next_day_trend"] = df["target_next_day_trend"].astype("int8")
    return df


def build_ml_ready_dataset(input_path: Path) -> pd.DataFrame:
    """Run the full daily preprocessing pipeline."""
    df = load_master_dataset(input_path)
    df = clean_price_change_outliers(df)
    df = resample_to_daily(df)
    df = engineer_relative_features(df)
    df = create_next_day_target(df)
    df = drop_and_scale_features(df)
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / INPUT_FILE
    output_path = base_dir / OUTPUT_FILE

    ml_ready_df = build_ml_ready_dataset(input_path)
    ml_ready_df.to_csv(output_path, index=False)

    print("target_next_day_trend value counts:")
    print(ml_ready_df["target_next_day_trend"].value_counts(dropna=False))
    print("\nML-ready dataset preview:")
    print(ml_ready_df.head())
    print(f"\nSaved ML-ready dataset to: {output_path}")


if __name__ == "__main__":
    main()
