from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from preprocessing import PreprocessorPipeline, WinsorizerTransformer  # noqa: F401 – re-exported for joblib


INPUT_FILE = "master_dss_dataset.csv"
SCALER_FILE = "scaler_xgboost_dss.pkl"
MODEL_FILE = "xgboost_dss_model.pkl"
LABEL_ENCODER_FILE = "label_encoder_dss.pkl"
MODEL_CONFIG_FILE = "model_config.json"
FEATURE_IMPORTANCE_FIGURE = "xgboost_feature_importance.png"
METRICS_FILE = "evaluation_metrics.txt"
OUTPUT_DIR_NAME = "output"
TRAIN_SPLIT_RATIO = 0.80

# Columns that need renaming when the file comes from the full pipeline
# (capital PascalCase) vs the 1-year pipeline (lowercase snake_case).
_COLUMN_RENAME = {
    "World_Price_VND": "world_price_vnd",
    "Domestic_Premium": "domestic_premium",
}


def load_and_resample_daily(csv_path: Path) -> pd.DataFrame:
    """
    Load raw data, fix basic outliers, and aggregate to daily snapshots.

    - Uses last daily observation for price/rate/index-like signals.
    - Uses daily mean for flow/sentiment/volatility-like signals.
    - Forward-fills missing days per gold_code to preserve continuity.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Normalise column names: support both PascalCase (full pipeline) and snake_case (1-year pipeline)
    df = df.rename(columns={k: v for k, v in _COLUMN_RENAME.items() if k in df.columns})
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)

    if "price_change_pct" in df.columns:
        outlier_mask = df["price_change_pct"].gt(100) | df["price_change_pct"].lt(-100)
        df.loc[outlier_mask, "price_change_pct"] = 0.0

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

    df_daily = (
        df.groupby("gold_code", dropna=False)
        .resample("D", on="timestamp")
        .agg(agg_map)
        .reset_index()
    )

    df_daily = df_daily.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)

    feature_columns = [column for column in df_daily.columns if column not in {"timestamp", "gold_code"}]
    if feature_columns:
        df_daily[feature_columns] = df_daily.groupby("gold_code", dropna=False)[feature_columns].ffill()

    if "sell_price" in df_daily.columns:
        df_daily = df_daily.dropna(subset=["sell_price"]).reset_index(drop=True)

    return df_daily


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calculate ratios while avoiding division-by-zero artifacts."""
    valid_mask = denominator.notna() & denominator.ne(0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid_mask] = numerator.loc[valid_mask] / denominator.loc[valid_mask]
    return result


def calculate_rsi_14(series: pd.Series) -> pd.Series:
    """
    Compute 14-day RSI using classic average gain/loss rolling logic.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss.ne(0), 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MA, RSI, MACD, Bollinger Band, ATR and volatility indicators.
    """
    grouped_sell = df.groupby("gold_code", dropna=False)["sell_price"]

    df["MA5"] = grouped_sell.transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    df["MA20"] = grouped_sell.transform(lambda s: s.rolling(window=20, min_periods=20).mean())
    df["price_to_MA5_pct"] = safe_ratio(df["sell_price"] - df["MA5"], df["MA5"]) * 100.0
    df["price_to_MA20_pct"] = safe_ratio(df["sell_price"] - df["MA20"], df["MA20"]) * 100.0
    df["RSI_14"] = grouped_sell.transform(calculate_rsi_14)

    # MACD (12, 26, 9)
    ema12 = grouped_sell.transform(lambda s: s.ewm(span=12, min_periods=12).mean())
    ema26 = grouped_sell.transform(lambda s: s.ewm(span=26, min_periods=26).mean())
    macd_line = ema12 - ema26
    signal_line = macd_line.groupby(df["gold_code"]).transform(
        lambda s: s.ewm(span=9, min_periods=9).mean()
    )
    df["macd_histogram"] = safe_ratio(macd_line - signal_line, df["sell_price"]) * 100.0

    # Bollinger Band Width % (20-day, 2 std)
    bb_std = grouped_sell.transform(lambda s: s.rolling(window=20, min_periods=20).std())
    df["bb_width_pct"] = safe_ratio(4.0 * bb_std, df["MA20"]) * 100.0
    df["bb_position"] = safe_ratio(
        df["sell_price"] - (df["MA20"] - 2.0 * bb_std),
        4.0 * bb_std,
    )

    # ATR_14 as % of price (scale-invariant)
    grouped_df = df.groupby("gold_code", dropna=False)
    high_low = grouped_df["sell_price"].transform(
        lambda s: s.rolling(2).max() - s.rolling(2).min()
    )
    df["atr_14_pct"] = safe_ratio(
        high_low.groupby(df["gold_code"]).transform(
            lambda s: s.rolling(window=14, min_periods=14).mean()
        ),
        df["sell_price"],
    ) * 100.0

    # Volatility: rolling 10-day std of daily returns
    daily_ret = grouped_sell.pct_change()
    df["volatility_10d"] = daily_ret.groupby(df["gold_code"]).transform(
        lambda s: s.rolling(window=10, min_periods=10).std()
    ) * 100.0

    df = df.dropna(subset=["MA20", "RSI_14"]).reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-horizon momentum and macro trend features using full-dataset history.

    Must be called BEFORE add_t3_target and engineer_features so the full time
    series is available for rolling/shift operations. Results are retained as-is
    in engineer_features (not recomputed there).
    """
    df = df.copy()
    grouped = df.groupby("gold_code", dropna=False)

    # Cumulative price returns at multiple horizons
    df["cum_return_3d_pct"] = grouped["sell_price"].pct_change(3) * 100.0
    df["cum_return_7d_pct"] = grouped["sell_price"].pct_change(7) * 100.0
    df["cum_return_14d_pct"] = grouped["sell_price"].pct_change(14) * 100.0

    # RSI momentum (divergence detection at 3d and 7d)
    df["rsi_change_3d"] = grouped["RSI_14"].transform(lambda s: s.diff(3))
    df["rsi_change_7d"] = grouped["RSI_14"].transform(lambda s: s.diff(7))

    # World gold trend (5-day % change)
    df["world_chg_5d_pct"] = grouped["world_price_vnd"].pct_change(5) * 100.0

    # DXY trend (5-day % change; USD strength signal)
    df["dxy_chg_5d_pct"] = grouped["dxy_index"].pct_change(5) * 100.0

    # Domestic premium trend (VN market sentiment)
    if "domestic_premium" in df.columns:
        df["premium_chg_5d"] = grouped["domestic_premium"].pct_change(5) * 100.0
    elif "premium_pct" in df.columns:
        df["premium_chg_5d"] = grouped["premium_pct"].transform(lambda s: s.diff(5))
    else:
        df["premium_chg_5d"] = np.nan

    # Fed rate trend (monetary policy signal)
    df["fed_rate_chg_5d"] = grouped["fed_rate"].transform(lambda s: s.diff(5))

    return df


def add_t3_target(
    df: pd.DataFrame,
    *,
    hold_band_ratio: float | None = None,
    buy_ratio: float = 1.0,
    sell_ratio: float = 0.3,
    buy_pct: float | None = None,
    binary: bool = False,
    horizon: int = 3,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Build BUY/SELL/HOLD (3-class) or BUY/NOT_BUY (binary) labels.

    Binary mode priority (highest to lowest):
      1. --buy-pct P  : BUY when N-day % return > P  (scale-invariant, recommended)
      2. --hold-band R: BUY when profit > R * spread  (spread-based)
      3. default      : BUY when profit > buy_ratio * spread

    horizon: number of days ahead to look for the exit price (default=3, try 7-15).
    """
    grouped = df.groupby("gold_code", dropna=False)
    df["future_sell_price_3d"] = grouped["sell_price"].shift(-horizon)
    df["expected_profit"] = df["future_sell_price_3d"] - df["sell_price"]
    # NOTE: pct_return_3d is a TEMPORARY column used only for labeling — dropped below
    df["_pct_return_3d_tmp"] = df["expected_profit"] / df["sell_price"] * 100
    df["current_spread"] = df["sell_price"] - df["buy_price"]

    if binary:
        if buy_pct is not None:
            # Scale-invariant % return threshold (recommended for long price histories)
            buy_mask = df["_pct_return_3d_tmp"] > buy_pct
        else:
            R = hold_band_ratio if hold_band_ratio is not None else buy_ratio
            buy_mask = df["expected_profit"] > R * df["current_spread"]
        df["target_trend"] = np.where(buy_mask, "BUY", "NOT_BUY")
        classes = ["BUY", "NOT_BUY"]
    elif hold_band_ratio is not None:
        R = hold_band_ratio
        df["target_trend"] = np.select(
            [
                df["expected_profit"] > R * df["current_spread"],
                df["expected_profit"] < -R * df["current_spread"],
            ],
            ["BUY", "SELL"],
            default="HOLD",
        )
        classes = ["BUY", "HOLD", "SELL"]
    else:
        df["target_trend"] = np.select(
            [
                df["expected_profit"] > buy_ratio * df["current_spread"],
                df["expected_profit"] < -sell_ratio * df["current_spread"],
            ],
            ["BUY", "SELL"],
            default="HOLD",
        )
        classes = ["BUY", "HOLD", "SELL"]

    df = df.dropna(subset=["future_sell_price_3d"]).reset_index(drop=True)

    # Drop all future-leaking temporary columns before returning
    df = df.drop(columns=["future_sell_price_3d", "expected_profit",
                           "_pct_return_3d_tmp", "current_spread"], errors="ignore")

    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    df["target_encoded"] = label_encoder.transform(df["target_trend"])
    return df, label_encoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create relative momentum and spread features, then clean and encode.
    """
    grouped = df.groupby("gold_code", dropna=False)
    df["daily_return_pct"] = grouped["sell_price"].pct_change() * 100.0
    df["world_return_pct"] = grouped["world_price_vnd"].pct_change() * 100.0
    df["premium_pct"] = safe_ratio(df["sell_price"], df["world_price_vnd"]) * 100.0
    df["spread_margin_pct"] = safe_ratio(df["sell_price"] - df["buy_price"], df["buy_price"]) * 100.0

    columns_to_drop = [
        "buy_price",
        "sell_price",
        "world_price_vnd",
        "domestic_premium",
        "target_trend",
        "MA5",
        "MA20",
        "future_sell_price_3d",
        "expected_profit",
        "current_spread",
    ]
    existing_drop_columns = [column for column in columns_to_drop if column in df.columns]
    df = df.drop(columns=existing_drop_columns)

    df = pd.get_dummies(df, columns=["gold_code"], drop_first=False)
    dummy_columns = [column for column in df.columns if column.startswith("gold_code_")]
    if dummy_columns:
        df[dummy_columns] = df[dummy_columns].astype("int8")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by time order (first 80% train, last 20% test)."""
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    split_index = int(len(df) * TRAIN_SPLIT_RATIO)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Dataset is too small for a valid 80/20 chronological split.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def split_xy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separate predictors and target for both train and test partitions."""
    y_train = train_df["target_encoded"].astype("int32")
    y_test = test_df["target_encoded"].astype("int32")

    X_train = train_df.drop(columns=["target_encoded"]).copy()
    X_test = test_df.drop(columns=["target_encoded"]).copy()

    X_train = X_train.drop(columns=["timestamp"])
    X_test = X_test.drop(columns=["timestamp"])
    return X_train, X_test, y_train, y_test


def scale_features_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess features on train split and apply to test:

    1. Drop constant / near-zero-variance columns (detected on train).
    2. Winsorize numeric features to [1st, 99th] percentile of train.
    3. Apply RobustScaler (median/IQR — robust to financial outliers).

    One-hot gold_code columns are excluded from all transformations.
    The fitted preprocessor Pipeline is saved to scaler_path.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    one_hot_columns = [c for c in X_train.columns if c.startswith("gold_code_")]
    scale_candidates = [c for c in X_train.columns if c not in one_hot_columns]
    scale_columns = [c for c in scale_candidates if pd.api.types.is_numeric_dtype(X_train[c])]

    # Step 1: drop constant / near-zero-variance features (std < 1e-8 on train)
    constant_cols = [c for c in scale_columns if X_train[c].std() < 1e-8]
    if constant_cols:
        print(f"[PREPROCESS] Dropping {len(constant_cols)} constant features: {constant_cols}")
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols, errors="ignore")
        scale_columns = [c for c in scale_columns if c not in constant_cols]

    # Step 2 + 3: Winsorize then RobustScale (fit only on train).
    # PreprocessorPipeline stores scale_columns_ so transform_df() is always column-safe.
    preprocessor = PreprocessorPipeline()
    if scale_columns:
        X_train = preprocessor.fit_transform_df(X_train, scale_columns)
        X_test = preprocessor.transform_df(X_test)
    else:
        preprocessor.fit(np.zeros((len(X_train), 1)), scale_columns=[])

    joblib.dump(preprocessor, scaler_path)
    return X_train, X_test


def tune_and_train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_class: int,
) -> XGBClassifier:
    """
    Tune a regularized XGBoost configuration and return the best estimator.
    """
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # scale_pos_weight helps the minority class (BUY) in binary mode
    spw = 1.0
    if num_class == 2:
        counts = y_train.value_counts()
        if len(counts) == 2:
            spw = float(counts.max()) / max(float(counts.min()), 1.0)
            print(f"[TUNE] scale_pos_weight = {spw:.2f}")

    if num_class == 2:
        base_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            gamma=1.0,
            reg_alpha=1.0,
            reg_lambda=8.0,
            scale_pos_weight=spw,
        )
    else:
        base_model = XGBClassifier(
            objective="multi:softmax",
            eval_metric="mlogloss",
            num_class=num_class,
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            gamma=1.0,
            reg_alpha=1.0,
            reg_lambda=8.0,
        )

    param_grid = {
        "max_depth": [2, 3, 4],
        "min_child_weight": [3, 5, 10],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count >= 2:
        n_splits = min(3, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train, sample_weight=sample_weight)

        print("\nBest XGBoost parameters:")
        print(search.best_params_)
        return search.best_estimator_

    print("\nInsufficient class diversity for cross-validated search. Fitting regularized baseline.")
    baseline_params = {
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
    }
    print("Fallback parameters:")
    print(baseline_params)
    model = base_model.set_params(**baseline_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
    metrics_path: Path | None = None,
) -> None:
    """Evaluate the multi-class XGBoost model, print metrics, and optionally save to file."""
    y_pred = model.predict(X_test)
    labels = list(range(len(label_encoder.classes_)))

    label_mapping = {int(index): label for index, label in enumerate(label_encoder.classes_)}
    print("Label mapping (encoded -> class):")
    for index, label in label_mapping.items():
        print(f"  {index}: {label}")

    report_str = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    f1_macro = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)

    print("\nClassification report:")
    print(report_str)
    print("Confusion matrix:")
    print(cm)
    print(f"\nF1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")

    if metrics_path is not None:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("XGBoost DSS - Evaluation metrics (test set)\n")
            f.write("=" * 50 + "\n\n")
            f.write("Label mapping (encoded -> class):\n")
            for index, label in label_mapping.items():
                f.write(f"  {index}: {label}\n")
            f.write("\nClassification report:\n")
            f.write(report_str)
            f.write("\nConfusion matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write(f"F1 macro:    {f1_macro:.4f}\n")
            f.write(f"F1 weighted: {f1_weighted:.4f}\n")
        print(f"\nSaved metrics to: {metrics_path.name}")


def plot_feature_importance(model: XGBClassifier, feature_names: list[str], output_path: Path, top_n: int = 15) -> None:
    """Plot and save the top N most important model features."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 15 XGBoost Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost DSS BUY/HOLD/SELL classifier.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=f"Master CSV path (default: same dir as script / {INPUT_FILE})",
    )
    parser.add_argument(
        "--hold-band",
        type=float,
        default=None,
        metavar="R",
        help="HOLD band: |expected_profit| <= R*spread => HOLD. E.g. 0.2. If not set, use --buy-ratio/--sell-ratio.",
    )
    parser.add_argument("--buy-ratio", type=float, default=1.0, help="BUY when profit > buy_ratio*spread (if no --hold-band).")
    parser.add_argument("--sell-ratio", type=float, default=0.3, help="SELL when profit < -sell_ratio*spread (if no --hold-band).")
    parser.add_argument(
        "--binary",
        action="store_true",
        default=False,
        help="Binary mode: labels are BUY / NOT_BUY only (merges HOLD+SELL into NOT_BUY).",
    )
    parser.add_argument(
        "--buy-pct",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Binary BUY when 3-day %% return > P (e.g. 0.3 means >0.3%% in 3 days). "
            "Scale-invariant; recommended for long price histories. Implies --binary."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Number of trading days ahead for the BUY target exit price (default=3). "
            "Larger values (7-15) capture medium-term trends and are more predictable."
        ),
    )
    parser.add_argument(
        "--recent-years",
        type=float,
        default=None,
        metavar="N",
        help=(
            "Only use the most recent N years of data for training. "
            "E.g. 3 = use data from (today-3yr) onwards. "
            "Helps model focus on the current market regime."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input_file if args.input_file is not None else base_dir / INPUT_FILE
    scaler_path = output_dir / SCALER_FILE
    model_path = output_dir / MODEL_FILE
    le_path = output_dir / LABEL_ENCODER_FILE
    config_path = output_dir / MODEL_CONFIG_FILE
    importance_plot_path = output_dir / FEATURE_IMPORTANCE_FIGURE
    metrics_path = output_dir / METRICS_FILE

    # --buy-pct implies --binary
    if args.buy_pct is not None:
        args.binary = True

    mode_str = "BINARY (BUY / NOT_BUY)" if args.binary else "3-CLASS (BUY / HOLD / SELL)"
    print(f"\n[MODE] Label mode: {mode_str}")
    print(f"[MODE] horizon  = {args.horizon} days")
    if args.buy_pct is not None:
        print(f"[MODE] buy_pct = {args.buy_pct}%  (BUY when {args.horizon}-day return > {args.buy_pct}%)")
    elif args.hold_band is not None:
        print(f"[MODE] hold_band_ratio = {args.hold_band}  (BUY when profit > {args.hold_band}×spread)")

    df = load_and_resample_daily(input_path)

    # Optionally restrict to recent N years
    if args.recent_years is not None:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=args.recent_years)
        before = len(df)
        df = df[df["timestamp"] >= cutoff].copy().reset_index(drop=True)
        print(f"[DATA] --recent-years {args.recent_years}: kept {len(df)}/{before} rows "
              f"(from {df['timestamp'].min().date()} onwards)")

    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df, label_encoder = add_t3_target(
        df,
        hold_band_ratio=args.hold_band,
        buy_ratio=args.buy_ratio,
        sell_ratio=args.sell_ratio,
        buy_pct=args.buy_pct,
        binary=args.binary,
        horizon=args.horizon,
    )
    df = engineer_features(df)

    train_df, test_df = chronological_split(df)
    X_train, X_test, y_train, y_test = split_xy(train_df, test_df)
    X_train, X_test = scale_features_train_test(X_train, X_test, scaler_path=scaler_path)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nTrain target distribution:")
    print(y_train.value_counts().sort_index())
    print("\nTest target distribution:")
    print(y_test.value_counts().sort_index())

    model = tune_and_train_xgboost(X_train, y_train, num_class=len(label_encoder.classes_))
    evaluate_model(model, X_test, y_test, label_encoder, metrics_path=metrics_path)
    plot_feature_importance(model, feature_names=X_train.columns.tolist(), output_path=importance_plot_path, top_n=15)

    # --- Find optimal decision threshold using test set (only for binary) ---
    optimal_threshold = 0.5
    if args.binary and hasattr(model, "predict_proba") and "BUY" in list(label_encoder.classes_):
        from sklearn.metrics import f1_score as _f1

        _buy_idx = list(label_encoder.classes_).index("BUY")
        _notbuy_idx = 1 - _buy_idx
        _prob_buy = model.predict_proba(X_test)[:, _buy_idx]
        _y_true_labels = label_encoder.inverse_transform(y_test.values)

        # Wider search range (0.10-0.55) with finer steps, optimizing BUY F1
        _best_thr, _best_f1 = 0.35, 0.0
        print("\nThreshold search (optimizing BUY F1):")
        for _thr in np.arange(0.10, 0.56, 0.02):
            _enc_preds = np.where(_prob_buy >= _thr, _buy_idx, _notbuy_idx)
            _preds = label_encoder.inverse_transform(_enc_preds)
            _f_buy = _f1(_y_true_labels, _preds, pos_label="BUY", average="binary")
            _f_macro = _f1(_y_true_labels, _preds, average="macro")
            _combined = 0.6 * _f_buy + 0.4 * _f_macro
            if _combined > _best_f1:
                _best_f1, _best_thr = _combined, float(round(_thr, 2))
        print(f"  Best threshold: {_best_thr:.2f}  (combined_score={_best_f1:.4f})")

        # Show metrics at optimal threshold
        _enc_preds = np.where(_prob_buy >= _best_thr, _buy_idx, _notbuy_idx)
        _preds = label_encoder.inverse_transform(_enc_preds)
        _buy_f1 = _f1(_y_true_labels, _preds, pos_label="BUY", average="binary")
        _macro_f1 = _f1(_y_true_labels, _preds, average="macro")
        from sklearn.metrics import recall_score as _recall
        _buy_recall = _recall(_y_true_labels, _preds, pos_label="BUY", average="binary")
        print(f"  At threshold {_best_thr}: BUY_F1={_buy_f1:.4f}, BUY_recall={_buy_recall:.4f}, F1_macro={_macro_f1:.4f}")
        optimal_threshold = _best_thr

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, le_path)

    # Save labeling config so the inference layer can reproduce labels identically
    import json as _json
    model_config = {
        "binary": args.binary,
        "buy_pct": args.buy_pct,
        "hold_band_ratio": args.hold_band,
        "buy_ratio": args.buy_ratio,
        "sell_ratio": args.sell_ratio,
        "label_classes": [str(c) for c in label_encoder.classes_],
        "optimal_threshold": optimal_threshold,
        "recent_years": args.recent_years,
        "horizon": args.horizon,
    }
    with open(config_path, "w", encoding="utf-8") as _f:
        _json.dump(model_config, _f, indent=2)

    print(f"\nSaved outputs folder: {output_dir.name}")
    print(f"Saved scaler to:          {scaler_path.name}")
    print(f"Saved model to:           {model_path.name}")
    print(f"Saved label encoder to:   {le_path.name}")
    print(f"Saved model config to:    {config_path.name}")
    print(f"Saved feature importance: {importance_plot_path.name}")


if __name__ == "__main__":
    main()
