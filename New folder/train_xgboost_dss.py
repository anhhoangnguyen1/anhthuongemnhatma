from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


INPUT_FILE = "master_dss_dataset.csv"
SCALER_FILE = "scaler_xgboost_dss.pkl"
MODEL_FILE = "xgboost_dss_model.pkl"
FEATURE_IMPORTANCE_FIGURE = "xgboost_feature_importance.png"
METRICS_FILE = "evaluation_metrics.txt"
OUTPUT_DIR_NAME = "output"
TRAIN_SPLIT_RATIO = 0.80


def load_and_resample_daily(csv_path: Path) -> pd.DataFrame:
    """
    Load raw data, fix basic outliers, and aggregate to daily snapshots.

    - Uses last daily observation for price/rate/index-like signals.
    - Uses daily mean for flow/sentiment/volatility-like signals.
    - Forward-fills missing days per gold_code to preserve continuity.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
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
    Add MA and RSI technical indicators from sell_price.
    """
    grouped_sell = df.groupby("gold_code", dropna=False)["sell_price"]

    df["MA5"] = grouped_sell.transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    df["MA20"] = grouped_sell.transform(lambda s: s.rolling(window=20, min_periods=20).mean())
    df["price_to_MA5_pct"] = safe_ratio(df["sell_price"] - df["MA5"], df["MA5"]) * 100.0
    df["price_to_MA20_pct"] = safe_ratio(df["sell_price"] - df["MA20"], df["MA20"]) * 100.0
    df["RSI_14"] = grouped_sell.transform(calculate_rsi_14)

    # Keep only rows where long-window indicators are defined.
    df = df.dropna(subset=["MA20", "RSI_14"]).reset_index(drop=True)
    return df


def add_t3_target(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Build BUY/SELL/HOLD labels using a 3-day holding horizon and current spread.
    """
    grouped = df.groupby("gold_code", dropna=False)
    df["future_sell_price_3d"] = grouped["sell_price"].shift(-3)
    df["expected_profit"] = df["future_sell_price_3d"] - df["sell_price"]
    df["current_spread"] = df["sell_price"] - df["buy_price"]

    df["target_trend"] = np.select(
        [
            df["expected_profit"] > df["current_spread"],
            df["expected_profit"] < -df["current_spread"],
        ],
        [
            "BUY",
            "SELL",
        ],
        default="HOLD",
    )

    df = df.dropna(subset=["future_sell_price_3d"]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    # Keep a stable mapping for BUY/HOLD/SELL even if one class is absent.
    label_encoder.fit(["BUY", "HOLD", "SELL"])
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
    Fit StandardScaler on train split only and transform both splits.

    One-hot gold_code columns are excluded from scaling.
    Constant columns are intentionally kept; StandardScaler handles them safely.
    """
    one_hot_columns = [column for column in X_train.columns if column.startswith("gold_code_")]
    scale_candidates = [column for column in X_train.columns if column not in one_hot_columns]
    scale_columns = [column for column in scale_candidates if pd.api.types.is_numeric_dtype(X_train[column])]

    scaler = StandardScaler()
    if scale_columns:
        X_train.loc[:, scale_columns] = scaler.fit_transform(X_train[scale_columns])
        X_test.loc[:, scale_columns] = scaler.transform(X_test[scale_columns])
    else:
        scaler.fit(np.zeros((len(X_train), 1)))

    joblib.dump(scaler, scaler_path)
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

    base_model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=num_class,
        n_estimators=300,
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
        print(f"\nSaved metrics to: {metrics_path}")


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


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = base_dir / INPUT_FILE
    scaler_path = output_dir / SCALER_FILE
    model_path = output_dir / MODEL_FILE
    importance_plot_path = output_dir / FEATURE_IMPORTANCE_FIGURE
    metrics_path = output_dir / METRICS_FILE

    df = load_and_resample_daily(input_path)
    df = add_technical_indicators(df)
    df, label_encoder = add_t3_target(df)
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

    joblib.dump(model, model_path)
    print(f"\nSaved outputs folder: {output_dir}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved feature importance plot to: {importance_plot_path}")


if __name__ == "__main__":
    main()
