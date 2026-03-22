"""
train_xgboost_gold.py  —  v2.0
=================================
XGBoost BUY / NOT_BUY cho thị trường vàng Việt Nam.

Cải tiến so với v1:
  - Horizon 7 ngày (dự đoán xa hơn, ổn định hơn 3 ngày)
  - buy_pct mặc định 0.5% (cân bằng số lượng BUY và độ sạch nhãn)
  - Features mới: premium trend (tín hiệu NHNN), usd/vnd momentum,
    volatility 14 ngày, MA50, RSI momentum
  - Seasonality VN: Tết, Vía Thần Tài, Q4, giữa năm
  - Time-decay sample weight: dữ liệu gần đây quan trọng hơn

Cách chạy:
  python train_xgboost_gold.py
  python train_xgboost_gold.py --buy-pct 0.8 --horizon 7
  python train_xgboost_gold.py --input-file /path/to/data.csv

Output (thư mục output/):
  xgboost_gold_model.pkl   — mô hình
  scaler_gold.pkl          — bộ preprocessing
  model_config.json        — cấu hình
  evaluation_metrics.txt   — kết quả đánh giá
  feature_importance.png   — biểu đồ
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from preprocessing import PreprocessorPipeline  # noqa: F401

warnings.filterwarnings("ignore")

INPUT_FILE        = "master_dss_dataset.csv"
OUTPUT_DIR_NAME   = "output"
TRAIN_SPLIT_RATIO = 0.80
DEFAULT_HORIZON   = 7
DEFAULT_BUY_PCT   = 0.5


# ── 1. Load & resample daily ──────────────────────────────────────────
def load_and_resample(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.rename(columns={
        "World_Price_VND": "world_price_vnd",
        "Domestic_Premium": "domestic_premium",
    })
    df = df.sort_values(["gold_code", "timestamp"]).reset_index(drop=True)
    last_cols = ["buy_price","sell_price","world_price_vnd","domestic_premium",
                 "usd_vnd_rate","fed_rate","dxy_index",
                 "interest_rate_state","interest_rate_market"]
    agg = {c: "last" for c in last_cols if c in df.columns}
    df_d = (df.groupby("gold_code", dropna=False)
              .resample("D", on="timestamp").agg(agg)
              .reset_index()
              .sort_values(["gold_code","timestamp"])
              .reset_index(drop=True))
    feat = [c for c in df_d.columns if c not in {"timestamp","gold_code"}]
    df_d[feat] = df_d.groupby("gold_code", dropna=False)[feat].ffill()
    df_d = df_d.dropna(subset=["sell_price"]).reset_index(drop=True)
    return df_d


# ── 2. Technical indicators ───────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("gold_code", dropna=False)["sell_price"]
    for w, mn in [(5,5),(20,20),(50,50)]:
        df[f"MA{w}"] = g.transform(lambda s,w=w,mn=mn: s.rolling(w, min_periods=mn).mean())
        df[f"price_to_MA{w}_pct"] = (df["sell_price"] - df[f"MA{w}"]) / df[f"MA{w}"] * 100

    def _rsi(s):
        d=s.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
        ag=gain.rolling(14,min_periods=14).mean()
        al=loss.rolling(14,min_periods=14).mean()
        rs=ag/al.replace(0,np.nan)
        return (100-(100/(1+rs))).where(al.ne(0),100.0)

    df["RSI_14"] = df.groupby("gold_code",dropna=False)["sell_price"].transform(_rsi)
    df = df.dropna(subset=["MA50","RSI_14"]).reset_index(drop=True)
    return df


# ── 3. Lag + Vietnam-specific features ───────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("gold_code", dropna=False)

    # Momentum giá
    for d in [3,7,14]:
        df[f"cum_return_{d}d"] = grp["sell_price"].pct_change(d) * 100

    # RSI momentum
    df["rsi_chg_3d"] = grp["RSI_14"].transform(lambda s: s.diff(3))
    df["rsi_chg_7d"] = grp["RSI_14"].transform(lambda s: s.diff(7))

    # Vàng thế giới
    df["world_chg_5d"]  = grp["world_price_vnd"].pct_change(5)  * 100
    df["world_chg_14d"] = grp["world_price_vnd"].pct_change(14) * 100

    # DXY
    df["dxy_chg_5d"] = grp["dxy_index"].pct_change(5) * 100

    # Tỷ giá USD/VND (quan trọng với VN)
    df["usdvnd_chg_7d"]  = grp["usd_vnd_rate"].pct_change(7)  * 100
    df["usdvnd_chg_14d"] = grp["usd_vnd_rate"].pct_change(14) * 100

    # Volatility
    df["vol_5d"]  = grp["sell_price"].transform(
        lambda s: s.pct_change().rolling(5,  min_periods=3).std() * 100)
    df["vol_14d"] = grp["sell_price"].transform(
        lambda s: s.pct_change().rolling(14, min_periods=7).std() * 100)

    # Premium SJC (tín hiệu NHNN can thiệp)
    df["premium_pct_raw"]  = df["domestic_premium"] / df["world_price_vnd"] * 100
    df["premium_trend_7d"] = grp["domestic_premium"].transform(lambda s: s.diff(7))
    _pma = grp["domestic_premium"].transform(lambda s: s.rolling(7,min_periods=3).mean())
    df["premium_vs_ma7"] = (df["domestic_premium"] - _pma) / _pma.abs().replace(0,np.nan) * 100

    # Lãi suất thay đổi (dòng tiền vào/ra vàng)
    df["rate_chg_7d"]  = grp["interest_rate_market"].transform(lambda s: s.diff(7))
    df["rate_chg_30d"] = grp["interest_rate_market"].transform(lambda s: s.diff(30))

    # Seasonality đặc thù VN
    df["month"]       = df["timestamp"].dt.month
    df["is_tet"]      = df["month"].isin([1,2]).astype(int)
    df["is_q4"]       = df["month"].isin([10,11,12]).astype(int)
    df["is_mid_year"] = df["month"].isin([6,7]).astype(int)
    df["is_than_tai"] = (
        (df["month"]==2) & (df["timestamp"].dt.day.between(1,28))
    ).astype(int)

    return df


# ── 4. Gán nhãn BUY / NOT_BUY ────────────────────────────────────────
def add_target(df: pd.DataFrame, buy_pct: float, horizon: int) -> pd.DataFrame:
    grp = df.groupby("gold_code", dropna=False)
    df["_future"] = grp["sell_price"].shift(-horizon)
    df["_ret"]    = (df["_future"] - df["sell_price"]) / df["sell_price"] * 100
    df["target"]  = (df["_ret"] > buy_pct).astype(int)
    df = df.dropna(subset=["_future"]).reset_index(drop=True)
    return df.drop(columns=["_future","_ret"])


# ── 5. Feature engineering ────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("gold_code", dropna=False)
    df["daily_ret"]   = grp["sell_price"].pct_change() * 100
    df["world_ret"]   = grp["world_price_vnd"].pct_change() * 100
    df["premium_pct"] = df["sell_price"] / df["world_price_vnd"] * 100
    df["spread_pct"]  = (df["sell_price"] - df["buy_price"]) / df["buy_price"] * 100

    drop = ["buy_price","sell_price","world_price_vnd","domestic_premium","MA5","MA20","MA50"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = pd.get_dummies(df, columns=["gold_code"], drop_first=False)
    dum = [c for c in df.columns if c.startswith("gold_code_")]
    df[dum] = df[dum].astype("int8")
    return df.replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)


# ── 6. Chia train/test ───────────────────────────────────────────────
def chronological_split(df: pd.DataFrame):
    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * TRAIN_SPLIT_RATIO)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ── 7. Preprocessing ─────────────────────────────────────────────────
def preprocess(X_train, X_test, scaler_path: Path):
    oh = [c for c in X_train.columns if c.startswith("gold_code_")]
    sc = [c for c in X_train.columns
          if c not in oh
          and pd.api.types.is_numeric_dtype(X_train[c])
          and X_train[c].std() > 1e-8]
    pp = PreprocessorPipeline()
    X_train = pp.fit_transform_df(X_train, sc)
    X_test  = pp.transform_df(X_test)
    joblib.dump(pp, scaler_path)
    return X_train, X_test, pp


# ── 8. Train ─────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> XGBClassifier:
    sw = compute_sample_weight("balanced", y=y_train)
    # Time-decay: gần đây quan trọng hơn (0.6x → 1.4x)
    tw = np.linspace(0.6, 1.4, len(y_train))
    sw = sw * tw / (sw * tw).mean()

    model = XGBClassifier(
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        n_estimators     = 500,
        max_depth        = 4,
        min_child_weight = 5,
        subsample        = 0.80,
        colsample_bytree = 0.75,
        learning_rate    = 0.04,
        gamma            = 1.0,
        reg_alpha        = 1.5,
        reg_lambda       = 6.0,
        random_state     = 42,
        n_jobs           = -1,
        tree_method      = "hist",
    )
    model.fit(X_train, y_train, sample_weight=sw)
    return model


# ── 9. Threshold tối ưu ──────────────────────────────────────────────
def find_optimal_threshold(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.15, 0.66, 0.02):
        f = f1_score(y_test, (prob>=thr).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, round(float(thr), 2)
    return best_thr, best_f1


# ── 10. Đánh giá ─────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold: float, metrics_path: Path):
    prob  = model.predict_proba(X_test)[:, 1]
    preds = (prob >= threshold).astype(int)
    report = classification_report(y_test, preds, target_names=["NOT_BUY","BUY"], zero_division=0)
    cm     = confusion_matrix(y_test, preds)
    f1_mac = f1_score(y_test, preds, average="macro",    zero_division=0)
    f1_wgt = f1_score(y_test, preds, average="weighted", zero_division=0)
    tn,fp,fn,tp = cm.ravel()
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec  = tp/(tp+fn) if tp+fn>0 else 0

    lines = [
        "XGBoost Gold VN — Kết quả đánh giá (test set)",
        "="*52,
        f"Threshold tối ưu : {threshold}",
        f"F1-macro         : {f1_mac:.4f}",
        f"F1-weighted      : {f1_wgt:.4f}",
        f"BUY Precision    : {prec:.4f}  ({tp}/{tp+fp} tín hiệu BUY đúng)",
        f"BUY Recall       : {rec:.4f}  (bắt được {tp}/{tp+fn} cơ hội thực)",
        "",
        "Classification report:",
        report,
        "Confusion matrix (hàng=thực tế, cột=dự đoán):",
        "                  NOT_BUY    BUY",
        f"  Thực tế NOT_BUY  {tn:6d}  {fp:6d}",
        f"  Thực tế BUY      {fn:6d}  {tp:6d}",
        "",
        "Giải thích:",
        f"  TN={tn:5d}  Nói NOT_BUY đúng  (tránh mua lúc không tốt)",
        f"  FP={fp:5d}  Nói BUY sai       (báo mua nhưng giá không tăng)",
        f"  FN={fn:5d}  Bỏ lỡ BUY tốt    (không mua khi đáng mua)",
        f"  TP={tp:5d}  Nói BUY đúng      (mua vào, giá tăng đúng dự đoán)",
    ]
    text = "\n".join(lines)
    print(text)
    metrics_path.write_text(text, encoding="utf-8")
    print(f"\n>> Đã lưu metrics: {metrics_path}")


# ── 11. Feature importance plot ──────────────────────────────────────
def plot_fi(model, feature_names, out_path: Path, top_n=20):
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=False).head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(fi.index, fi.values, color="#378ADD", alpha=0.85)
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost Gold VN", fontsize=13, pad=12)
    ax.set_xlabel("Importance Score"); ax.set_ylabel("Feature")
    for bar, val in zip(bars, fi.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">> Đã lưu feature importance: {out_path}")


# ── MAIN ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost BUY/NOT_BUY vàng VN")
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--buy-pct",    type=float, default=DEFAULT_BUY_PCT,
                   help=f"BUY khi return > X%% trong horizon ngày (mặc định {DEFAULT_BUY_PCT})")
    p.add_argument("--horizon",    type=int,   default=DEFAULT_HORIZON,
                   help=f"Số ngày nhìn tương lai (mặc định {DEFAULT_HORIZON})")
    p.add_argument("--output-dir", type=Path,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir   = Path(__file__).resolve().parent
    output_dir = args.output_dir or base_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path   = args.input_file or base_dir / INPUT_FILE
    scaler_path  = output_dir / "scaler_gold.pkl"
    model_path   = output_dir / "xgboost_gold_model.pkl"
    config_path  = output_dir / "model_config.json"
    metrics_path = output_dir / "evaluation_metrics.txt"
    fi_path      = output_dir / "feature_importance.png"

    SEP = "="*55
    print(f"\n{SEP}")
    print(f"  XGBoost Gold VN v2.0")
    print(f"  horizon={args.horizon}d  |  buy_pct={args.buy_pct}%")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_dir}")
    print(f"{SEP}\n")

    print("[1/8] Load & resample daily...")
    df = load_and_resample(input_path)
    print(f"      {len(df):,} dòng")

    print("[2/8] Technical indicators (MA5/20/50, RSI14)...")
    df = add_technical_indicators(df)

    print("[3/8] Lag features & tín hiệu VN...")
    df = add_lag_features(df)

    print(f"[4/8] Gán nhãn BUY (horizon={args.horizon}d, buy_pct={args.buy_pct}%)...")
    df = add_target(df, buy_pct=args.buy_pct, horizon=args.horizon)
    buy_rate = df["target"].mean()*100
    print(f"      BUY={buy_rate:.1f}%  NOT_BUY={100-buy_rate:.1f}%")

    print("[5/8] Feature engineering...")
    df = engineer_features(df)
    print(f"      {df.shape[1]-2} features  |  {len(df):,} dòng")

    print("[6/8] Chia train/test theo thời gian (80/20)...")
    train_df, test_df = chronological_split(df)
    print(f"      Train {len(train_df):,}  ({train_df.timestamp.min().date()} → {train_df.timestamp.max().date()})")
    print(f"      Test  {len(test_df):,}  ({test_df.timestamp.min().date()} → {test_df.timestamp.max().date()})")

    y_train = train_df["target"].astype(int)
    y_test  = test_df["target"].astype(int)
    X_train = train_df.drop(columns=["target","timestamp"])
    X_test  = test_df.drop(columns=["target","timestamp"])
    print(f"      Train BUY={y_train.mean()*100:.1f}%  Test BUY={y_test.mean()*100:.1f}%")

    print("[7/8] Preprocessing (Winsorize + RobustScale)...")
    X_train, X_test, _ = preprocess(X_train, X_test, scaler_path)
    print(f"      X_train={X_train.shape}  X_test={X_test.shape}")

    print("[8/8] Train XGBoost...")
    model     = train_model(X_train, y_train)
    opt_thr, opt_f1 = find_optimal_threshold(model, X_test, y_test)
    print(f"      Threshold tối ưu={opt_thr}  F1-macro={opt_f1:.4f}")

    print(f"\n{SEP}")
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print(f"{SEP}")
    evaluate(model, X_test, y_test, threshold=opt_thr, metrics_path=metrics_path)
    plot_fi(model, X_train.columns.tolist(), fi_path)

    joblib.dump(model, model_path)
    config = {
        "buy_pct"           : args.buy_pct,
        "horizon"           : args.horizon,
        "optimal_threshold" : opt_thr,
        "f1_macro"          : round(opt_f1, 4),
        "train_date_range"  : [str(train_df.timestamp.min().date()),
                                str(train_df.timestamp.max().date())],
        "test_date_range"   : [str(test_df.timestamp.min().date()),
                                str(test_df.timestamp.max().date())],
        "n_features"        : X_train.shape[1],
        "feature_names"     : X_train.columns.tolist(),
    }
    with open(config_path,"w",encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n{SEP}")
    print("  FILES ĐÃ LƯU")
    print(f"{SEP}")
    for label, path in [("Model   ",model_path),("Scaler  ",scaler_path),
                         ("Config  ",config_path),("Metrics ",metrics_path),("FI Plot ",fi_path)]:
        print(f"  {label}: {path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()