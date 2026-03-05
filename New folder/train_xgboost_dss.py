from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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


def add_t3_target(
    df: pd.DataFrame,
    *,
    sell_threshold_ratio: float = 0.3,
    buy_threshold_ratio: float = 1.0,
    hold_band_ratio: float | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Gán nhãn BUY / HOLD / SELL theo horizon 3 ngày và spread (research-based).

    Công thức:
    - expected_profit = giá_bán_sau_3_ngày - giá_bán_hiện_tại
    - current_spread = giá_bán - giá_mua (chi phí giao dịch, vàng VN ~3M ~1.5% giá)

    Hai chế độ:
    A) hold_band_ratio is None (mặc định): ngưỡng bất đối xứng
       - BUY:  expected_profit > buy_threshold_ratio * spread
       - SELL: expected_profit < -sell_threshold_ratio * spread
       - HOLD: còn lại (vùng giữa rộng)
    B) hold_band_ratio = số (vd 0.2): vùng HOLD đối xứng, dễ cân bằng 3 lớp
       - HOLD: |expected_profit| <= hold_band_ratio * spread (biến động gần 0)
       - BUY:  expected_profit > hold_band_ratio * spread
       - SELL: expected_profit < -hold_band_ratio * spread
    """
    grouped = df.groupby("gold_code", dropna=False)
    df["future_sell_price_3d"] = grouped["sell_price"].shift(-3)
    df["expected_profit"] = df["future_sell_price_3d"] - df["sell_price"]
    df["current_spread"] = df["sell_price"] - df["buy_price"]

    spread = df["current_spread"]
    ep = df["expected_profit"]

    if hold_band_ratio is not None:
        band = hold_band_ratio * spread
        df["target_trend"] = np.select(
            [ep > band, ep < -band],
            ["BUY", "SELL"],
            default="HOLD",
        )
    else:
        buy_threshold = buy_threshold_ratio * spread
        sell_threshold = sell_threshold_ratio * spread
        df["target_trend"] = np.select(
            [ep > buy_threshold, ep < -sell_threshold],
            ["BUY", "SELL"],
            default="HOLD",
        )

    df = df.dropna(subset=["future_sell_price_3d"]).reset_index(drop=True)

    label_encoder = LabelEncoder()
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


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by time order (first train_ratio train, rest test). Default TRAIN_SPLIT_RATIO (0.8)."""
    ratio = train_ratio if train_ratio is not None else TRAIN_SPLIT_RATIO
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    split_index = int(len(df) * ratio)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError(f"Dataset too small for chronological split with ratio {ratio}.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by stratify=y so train and test both have BUY/HOLD/SELL.
    CẢNH BÁO: Xáo trộn thứ tự thời gian, chỉ dùng để đánh giá (xem model có học được không khi test có đủ lớp).
    Không dùng cho mô hình production (gây data leakage thời gian).
    """
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    y = df["target_encoded"]
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=1 - train_ratio,
        stratify=y,
        random_state=random_state,
    )
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()
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
    *,
    hold_class_index: int = 1,
    hold_weight_extra: float = 1.3,
) -> XGBClassifier:
    """
    Tune a regularized XGBoost configuration and return the best estimator.
    hold_class_index: lớp HOLD (1 nếu label_encoder order là BUY=0, HOLD=1, SELL=2).
    hold_weight_extra: nhân thêm trọng số cho HOLD để model không bỏ qua lớp giữa.
    """
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if hold_weight_extra != 1.0:
        sample_weight = sample_weight * np.where(y_train == hold_class_index, hold_weight_extra, 1.0)

    base_model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=num_class,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        gamma=1.0,
        reg_alpha=1.5,
        reg_lambda=10.0,
    )

    param_grid = {
        "max_depth": [2, 3],
        "min_child_weight": [5, 10],
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
    extra_warnings: list[str] | None = None,
) -> dict:
    """
    Evaluate the multi-class XGBoost model, print metrics, optionally save to file.
    Returns dict with accuracy, f1_macro, f1_weighted, report_str, cm, test_support.
    """
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

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)
    test_support = {label_encoder.classes_[i]: int((y_test == i).sum()) for i in labels}

    print("\nClassification report:")
    print(report_str)
    print("Confusion matrix:")
    print(cm)
    print(f"\nAccuracy:     {acc:.4f}")
    print(f"F1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")

    if metrics_path is not None:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("XGBoost DSS - Evaluation metrics (test set)\n")
            f.write("=" * 50 + "\n\n")
            if extra_warnings:
                f.write("*** CẢNH BÁO ***\n")
                for w in extra_warnings:
                    f.write(f"  - {w}\n")
                f.write("\n")
            f.write("Label mapping (encoded -> class):\n")
            for index, label in label_mapping.items():
                f.write(f"  {index}: {label}\n")
            f.write("\nClassification report:\n")
            f.write(report_str)
            f.write("\nConfusion matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write(f"Accuracy:     {acc:.4f}\n")
            f.write(f"F1 macro:    {f1_macro:.4f}\n")
            f.write(f"F1 weighted: {f1_weighted:.4f}\n")
        print(f"\nSaved metrics to: {metrics_path}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report_str": report_str,
        "confusion_matrix": cm,
        "test_support": test_support,
    }


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


def _train_and_evaluate_one_split(
    df: pd.DataFrame,
    label_encoder: LabelEncoder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler_path: Path,
    num_class: int,
) -> dict:
    """One split: scale, train, evaluate. Returns metrics dict (no model save)."""
    X_train, X_test, y_train, y_test = split_xy(train_df, test_df)
    X_train, X_test = scale_features_train_test(X_train, X_test, scaler_path=scaler_path)
    model = tune_and_train_xgboost(X_train, y_train, num_class=num_class)
    return evaluate_model(model, X_test, y_test, label_encoder, metrics_path=None)


def run_multi_split_evaluation(
    df: pd.DataFrame,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    """
    Chạy đánh giá với nhiều cách chia: chronological 70/30, 80/20, 90/10 và stratified 80/20.
    Ghi kết quả và khuyến nghị vào evaluation_multi_splits.txt.
    """
    scaler_path = output_dir / "scaler_temp_multi.pkl"
    num_class = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)
    results: list[dict] = []

    for ratio_name, train_ratio in [("70/30", 0.7), ("80/20", 0.8), ("90/10", 0.9)]:
        print(f"\n{'='*60}\nChronological split {ratio_name}\n{'='*60}")
        try:
            train_df, test_df = chronological_split(df, train_ratio=train_ratio)
            m = _train_and_evaluate_one_split(
                df, label_encoder, train_df, test_df, scaler_path, num_class
            )
            m["split"] = f"chrono_{ratio_name}"
            m["test_support"] = {
                c: int((test_df["target_encoded"] == class_names.index(c)).sum())
                for c in class_names
            }
            results.append(m)
        except Exception as e:
            print(f"Chronological {ratio_name} failed: {e}")
            results.append({"split": f"chrono_{ratio_name}", "error": str(e)})

    print(f"\n{'='*60}\nStratified 80/20 (chỉ đánh giá, không dùng production)\n{'='*60}")
    try:
        train_df, test_df = stratified_split(df, train_ratio=0.8)
        m = _train_and_evaluate_one_split(
            df, label_encoder, train_df, test_df, scaler_path, num_class
        )
        m["split"] = "stratified_80/20"
        m["test_support"] = {
            c: int((test_df["target_encoded"] == class_names.index(c)).sum())
            for c in class_names
        }
        results.append(m)
    except Exception as e:
        print(f"Stratified failed: {e}")
        results.append({"split": "stratified_80/20", "error": str(e)})

    report_path = output_dir / "evaluation_multi_splits.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("XGBoost DSS - Đánh giá nhiều cách chia train/test\n")
        f.write("=" * 60 + "\n\n")
        f.write("Chronological = giữ thứ tự thời gian (phù hợp production).\n")
        f.write("Stratified = test có đủ lớp BUY/HOLD/SELL (chỉ để so sánh).\n\n")
        f.write("-" * 60 + "\n")
        for r in results:
            if "error" in r:
                f.write(f"Split: {r['split']} -> LỖI: {r['error']}\n\n")
                continue
            f.write(f"Split: {r['split']}\n")
            f.write(f"  Accuracy: {r.get('accuracy', 0):.4f}  F1 macro: {r.get('f1_macro', 0):.4f}  F1 weighted: {r.get('f1_weighted', 0):.4f}\n")
            f.write(f"  Test support: {r.get('test_support', {})}\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("KHUYẾN NGHỊ CẢI THIỆN:\n\n")
        f.write("1. Đa dạng dataset: thêm dữ liệu (nhiều tháng, nhiều mã vàng); điều chỉnh add_t3_target để có mẫu SELL.\n")
        f.write("2. Chia dữ liệu: Chronological cho production; Stratified chỉ để kiểm tra khi test thiếu lớp.\n")
        f.write("3. Mất cân bằng: đã dùng sample_weight='balanced'; có thể thử SMOTE/oversample BUY,SELL hoặc undersample HOLD.\n")
        f.write("4. Features: thêm/cải thiện chỉ báo kỹ thuật; xem feature importance sau mỗi lần train.\n")
    if scaler_path.exists():
        scaler_path.unlink()
    print(f"\nSaved multi-split evaluation to: {report_path}")


def _run_check_labels(df_after_tech: pd.DataFrame, output_dir: Path) -> None:
    """
    Kiểm tra phân bố nhãn BUY/HOLD/SELL và thống kê điều kiện SELL.
    df_after_tech: df sau add_technical_indicators (có sell_price, buy_price).
    """
    grouped = df_after_tech.groupby("gold_code", dropna=False)
    df = df_after_tech.copy()
    df["future_sell_price_3d"] = grouped["sell_price"].shift(-3)
    df["expected_profit"] = df["future_sell_price_3d"] - df["sell_price"]
    df["current_spread"] = df["sell_price"] - df["buy_price"]
    df = df.dropna(subset=["future_sell_price_3d"]).reset_index(drop=True)

    ep = df["expected_profit"]
    cs = df["current_spread"]
    print("\n" + "=" * 60)
    print("KIEM TRA NHAN TRONG DATASET (sau add_t3_target)")
    print("=" * 60)
    print("\nThong ke:")
    print(f"  expected_profit: min={ep.min():.0f}, max={ep.max():.0f}, mean={ep.mean():.0f} (VND)")
    print(f"  current_spread:  min={cs.min():.0f}, max={cs.max():.0f}, mean={cs.mean():.0f} (VND)")
    print("\nSo mau SELL theo nguong (expected_profit < -ratio*current_spread):")
    for ratio in [1.0, 0.5, 0.3, 0.0]:
        if ratio == 0.0:
            n_sell = (ep < 0).sum()
            print(f"  ratio=0 (SELL khi expected_profit < 0): {n_sell}")
        else:
            n_sell = (ep < -ratio * cs).sum()
            print(f"  ratio={ratio}: {n_sell}")
    # Phân bố với ratio mặc định 0.3 (dùng trong add_t3_target; với spread 3M → lỗ > 900k)
    default_sell_ratio = 0.3
    sell_mask = ep < -default_sell_ratio * cs
    buy_mask = ep > cs
    hold_mask = ~sell_mask & ~buy_mask
    n_buy, n_hold, n_sell = buy_mask.sum(), hold_mask.sum(), sell_mask.sum()
    print("\nPhan bo nhan (nguong mac dinh sell_ratio=0.3, buy_ratio=1.0):")
    print(f"  BUY:  {n_buy}")
    print(f"  HOLD: {n_hold}")
    print(f"  SELL: {n_sell}")
    if n_sell == 0:
        print("\n*** Van khong co SELL. Thu --sell-ratio 0.2 hoac thu thap them du lieu.")
    out_path = output_dir / "label_check_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Phan bo nhan (sell_ratio=0.3, buy_ratio=1.0): BUY=%d, HOLD=%d, SELL=%d\n" % (n_buy, n_hold, n_sell))
        f.write("expected_profit: min=%s, max=%s\n" % (ep.min(), ep.max()))
        f.write("current_spread: min=%s, max=%s\n" % (cs.min(), cs.max()))
    print(f"\nDa ghi: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost DSS (BUY/HOLD/SELL).")
    parser.add_argument(
        "--multi-split-eval",
        action="store_true",
        help="Chỉ chạy đánh giá nhiều cách chia (70/30, 80/20, 90/10 + stratified), ghi evaluation_multi_splits.txt.",
    )
    parser.add_argument(
        "--check-labels",
        action="store_true",
        help="Chỉ kiểm tra phân bố nhãn BUY/HOLD/SELL và thống kê điều kiện SELL (không train).",
    )
    parser.add_argument(
        "--sell-ratio",
        type=float,
        default=0.3,
        help="Tỷ lệ spread cho SELL: expected_profit < -sell_ratio*spread (mặc định 0.3; 0.5/1.0 = khắt hơn).",
    )
    parser.add_argument(
        "--buy-ratio",
        type=float,
        default=1.0,
        help="Tỷ lệ spread cho BUY: expected_profit > buy_ratio*spread (mặc định 1.0).",
    )
    parser.add_argument(
        "--hold-band",
        type=float,
        default=None,
        metavar="R",
        help="Vùng HOLD đối xứng: HOLD khi |expected_profit| <= R*spread, BUY/SELL ngoài R (vd 0.2). Bỏ qua nếu không set (dùng sell-ratio/buy-ratio).",
    )
    args = parser.parse_args()

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

    if args.check_labels:
        _run_check_labels(df, output_dir)
        return

    df, label_encoder = add_t3_target(
        df,
        sell_threshold_ratio=args.sell_ratio,
        buy_threshold_ratio=args.buy_ratio,
        hold_band_ratio=args.hold_band,
    )
    df = engineer_features(df)

    if args.multi_split_eval:
        run_multi_split_evaluation(df, label_encoder, output_dir)
        return

    train_df, test_df = chronological_split(df)
    X_train, X_test, y_train, y_test = split_xy(train_df, test_df)
    X_train, X_test = scale_features_train_test(X_train, X_test, scaler_path=scaler_path)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nTrain target distribution:")
    print(y_train.value_counts().sort_index())
    print("\nTest target distribution:")
    print(y_test.value_counts().sort_index())

    extra_warnings: list[str] = []
    for i, c in enumerate(label_encoder.classes_):
        if (y_test == i).sum() == 0:
            extra_warnings.append(
                f"Tập test không có mẫu lớp '{c}' (support=0). Cân nhắc --multi-split-eval hoặc thu thập thêm dữ liệu."
            )
    if extra_warnings:
        print("\n*** CẢNH BÁO ***")
        for w in extra_warnings:
            print(" ", w)

    model = tune_and_train_xgboost(X_train, y_train, num_class=len(label_encoder.classes_))
    evaluate_model(
        model, X_test, y_test, label_encoder,
        metrics_path=metrics_path,
        extra_warnings=extra_warnings if extra_warnings else None,
    )
    plot_feature_importance(model, feature_names=X_train.columns.tolist(), output_path=importance_plot_path, top_n=15)

    joblib.dump(model, model_path)
    print(f"\nSaved outputs folder: {output_dir}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved feature importance plot to: {importance_plot_path}")


if __name__ == "__main__":
    main()
