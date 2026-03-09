from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "master_dss_dataset.csv"
SCALER_FILE = "scaler_dss.pkl"
X_TRAIN_FILE = "X_train_dss.csv"
X_TEST_FILE = "X_test_dss.csv"
Y_TRAIN_FILE = "y_train_dss.csv"
Y_TEST_FILE = "y_test_dss.csv"
TRAIN_SPLIT_RATIO = 0.80


def load_master_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the source dataset and enforce the expected schema."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def fix_price_change_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace clearly invalid percentage entries with zero."""
    if "price_change_pct" not in df.columns:
        return df

    invalid_mask = df["price_change_pct"].gt(100) | df["price_change_pct"].lt(-100)
    df.loc[invalid_mask, "price_change_pct"] = 0.0
    return df


def resample_gold_series_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hourly observations into a daily series while preserving continuity.

    Prices and rate-like signals use the last available observation of the day.
    Volume, sentiment, and volatility-style features use the daily mean.
    Missing calendar days are retained by resampling and then forward-filled
    within each gold_code series.
    """
    last_columns = [
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
    mean_columns = [
        "price_change_pct",
        "volatility_index",
        "news_volume",
        "sentiment_score",
    ]

    agg_map: dict[str, str] = {}
    for column in last_columns:
        if column in df.columns:
            agg_map[column] = "last"
    for column in mean_columns:
        if column in df.columns:
            agg_map[column] = "mean"

    daily_df = (
        df.groupby("gold_code", dropna=False)
        .resample("D", on="timestamp")
        .agg(agg_map)
        .reset_index()
    )

    daily_df = daily_df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)

    feature_columns = [column for column in daily_df.columns if column not in {"timestamp", "gold_code"}]
    if feature_columns:
        daily_df[feature_columns] = daily_df.groupby("gold_code", dropna=False)[feature_columns].ffill()

    if "sell_price" in daily_df.columns:
        daily_df = daily_df.dropna(subset=["sell_price"]).reset_index(drop=True)

    return daily_df


def safe_percentage_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute a percentage ratio while avoiding divide-by-zero artifacts."""
    valid_mask = denominator.notna() & denominator.ne(0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid_mask] = (numerator.loc[valid_mask] / denominator.loc[valid_mask]) * 100.0
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build relative features and the next-day direction target."""
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("gold_code", dropna=False)

    df["daily_return_pct"] = grouped["sell_price"].pct_change() * 100.0
    df["world_return_pct"] = grouped["world_price_vnd"].pct_change() * 100.0
    df["premium_pct"] = safe_percentage_ratio(
        df["sell_price"] - df["world_price_vnd"],
        df["world_price_vnd"],
    )
    df["spread_margin_pct"] = safe_percentage_ratio(
        df["sell_price"] - df["buy_price"],
        df["buy_price"],
    )

    next_sell_price = grouped["sell_price"].shift(-1)
    df["target_next_day_trend"] = (next_sell_price > df["sell_price"]).astype("int8")
    df.loc[next_sell_price.isna(), "target_next_day_trend"] = np.nan

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
    df["target_next_day_trend"] = df["target_next_day_trend"].astype("int8")
    return df


def one_hot_encode_gold_code(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gold_code without dropping any category."""
    encoded_df = pd.get_dummies(df, columns=["gold_code"], drop_first=False)
    dummy_columns = [column for column in encoded_df.columns if column.startswith("gold_code_")]
    if dummy_columns:
        encoded_df[dummy_columns] = encoded_df[dummy_columns].astype("int8")
    return encoded_df


def chronological_train_test_split(
    df: pd.DataFrame, train_ratio: float = TRAIN_SPLIT_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the full encoded dataset by time order instead of random sampling."""
    sorted_df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    split_index = int(len(sorted_df) * train_ratio)

    if split_index <= 0 or split_index >= len(sorted_df):
        raise ValueError("Dataset is too small to create a valid 80/20 chronological split.")

    train_df = sorted_df.iloc[:split_index].copy()
    test_df = sorted_df.iloc[split_index:].copy()
    return train_df, test_df


def split_features_and_target(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separate numeric predictors from the target while keeping timestamps out of X."""
    target_column = "target_next_day_trend"

    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()

    X_train = train_df.drop(columns=[target_column]).copy()
    X_test = test_df.drop(columns=[target_column]).copy()

    if "timestamp" in X_train.columns:
        X_train = X_train.drop(columns=["timestamp"])
    if "timestamp" in X_test.columns:
        X_test = X_test.drop(columns=["timestamp"])

    return X_train, X_test, y_train, y_test


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit StandardScaler on the training window only, then transform both splits.

    Zero-variance columns are intentionally preserved; StandardScaler handles them
    safely by outputting zeros for the constant training values.
    """
    dummy_columns = [column for column in X_train.columns if column.startswith("gold_code_")]
    scale_columns = [column for column in X_train.columns if column not in dummy_columns]

    numeric_scale_columns = [
        column for column in scale_columns if pd.api.types.is_numeric_dtype(X_train[column])
    ]

    scaler = StandardScaler()
    if numeric_scale_columns:
        X_train.loc[:, numeric_scale_columns] = scaler.fit_transform(X_train[numeric_scale_columns])
        X_test.loc[:, numeric_scale_columns] = scaler.transform(X_test[numeric_scale_columns])
    else:
        scaler.fit(np.zeros((len(X_train), 1)))

    joblib.dump(scaler, scaler_path)
    return X_train, X_test


def save_split_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_dir: Path,
) -> None:
    """Persist the train/test artifacts for the next modeling stage."""
    X_train.to_csv(base_dir / X_TRAIN_FILE, index=False)
    X_test.to_csv(base_dir / X_TEST_FILE, index=False)
    y_train.to_frame(name="target_next_day_trend").to_csv(base_dir / Y_TRAIN_FILE, index=False)
    y_test.to_frame(name="target_next_day_trend").to_csv(base_dir / Y_TEST_FILE, index=False)


def build_training_artifacts(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """End-to-end preprocessing pipeline for modeling."""
    input_path = base_dir / INPUT_FILE
    scaler_path = base_dir / SCALER_FILE

    df = load_master_dataset(input_path)
    df = fix_price_change_outliers(df)
    df = resample_gold_series_to_daily(df)
    df = engineer_features(df)
    df = one_hot_encode_gold_code(df)

    train_df, test_df = chronological_train_test_split(df)
    X_train, X_test, y_train, y_test = split_features_and_target(train_df, test_df)
    X_train, X_test = scale_numeric_features(X_train, X_test, scaler_path=scaler_path)

    save_split_outputs(X_train, X_test, y_train, y_test, base_dir=base_dir)
    return X_train, X_test, y_train, y_test


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    X_train, X_test, y_train, y_test = build_training_artifacts(base_dir)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nTrain target distribution:")
    print(y_train.value_counts(dropna=False))
    print("\nTest target distribution:")
    print(y_test.value_counts(dropna=False))
    print(f"\nSaved scaler to: {base_dir / SCALER_FILE}")
    print(
        "Saved split files:",
        base_dir / X_TRAIN_FILE,
        base_dir / X_TEST_FILE,
        base_dir / Y_TRAIN_FILE,
        base_dir / Y_TEST_FILE,
    )


if __name__ == "__main__":
    main()
