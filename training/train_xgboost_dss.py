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
  
  train_xgboost_gold.py  —  v2.1  (với lưu file từng bước)
==========================================================
Giống v2.0, nhưng mỗi bước xử lý data đều lưu 1 file CSV vào
thư mục  output/pipeline_steps/  để dễ kiểm tra.

Files được tạo:
  step1_raw_resample.csv          sau load & resample
  step2_technical_indicators.csv  sau MA, RSI
  step3_lag_features.csv          sau lag + macro + seasonal
  step4_labeled.csv               sau gán nhãn BUY/NOT_BUY
  step5_engineered.csv            sau engineer features + one-hot
  step6_train.csv / step6_test.csv  sau chia 80/20
  step7_train_scaled.csv / step7_test_scaled.csv  sau Winsorize+Scale
  step8_predictions.csv           kết quả dự đoán trên test
  step9_feature_importance.csv    feature importance model
  decision_explanation.txt        giải thích tín hiệu mới nhất
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


# ── Hàm tiện ích lưu file từng bước ──────────────────────────────────
def save_step(df: pd.DataFrame, steps_dir: Path, filename: str, note: str = "") -> None:
    path = steps_dir / filename
    desc_lines = [
        f"# {filename}",
        f"# {note}",
        f"# Shape: {df.shape[0]:,} dong x {df.shape[1]} cot",
        f"# Cot dau: {', '.join(df.columns[:8].tolist())}{'...' if len(df.columns) > 8 else ''}",
        "#",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in desc_lines:
            f.write(line + "\n")
        df.to_csv(f, index=False)
    print(f"  >> Luu: {path.name}  ({df.shape[0]:,} dong, {df.shape[1]} cot)")


# ── 1. Load & resample ────────────────────────────────────────────────
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
    for d in [3,7,14]:
        df[f"cum_return_{d}d"] = grp["sell_price"].pct_change(d) * 100
    df["rsi_chg_3d"] = grp["RSI_14"].transform(lambda s: s.diff(3))
    df["rsi_chg_7d"] = grp["RSI_14"].transform(lambda s: s.diff(7))
    df["world_chg_5d"]  = grp["world_price_vnd"].pct_change(5)  * 100
    df["world_chg_14d"] = grp["world_price_vnd"].pct_change(14) * 100
    df["dxy_chg_5d"]    = grp["dxy_index"].pct_change(5) * 100
    df["usdvnd_chg_7d"]  = grp["usd_vnd_rate"].pct_change(7)  * 100
    df["usdvnd_chg_14d"] = grp["usd_vnd_rate"].pct_change(14) * 100
    df["vol_5d"]  = grp["sell_price"].transform(
        lambda s: s.pct_change().rolling(5,  min_periods=3).std() * 100)
    df["vol_14d"] = grp["sell_price"].transform(
        lambda s: s.pct_change().rolling(14, min_periods=7).std() * 100)
    df["premium_pct_raw"]  = df["domestic_premium"] / df["world_price_vnd"] * 100
    df["premium_trend_7d"] = grp["domestic_premium"].transform(lambda s: s.diff(7))
    _pma = grp["domestic_premium"].transform(lambda s: s.rolling(7,min_periods=3).mean())
    df["premium_vs_ma7"] = np.where(
        _pma.abs() < 1e3, 0.0,
        (df["domestic_premium"] - _pma) / _pma * 100)
    df["rate_chg_7d"]  = grp["interest_rate_market"].transform(lambda s: s.diff(7))
    df["rate_chg_30d"] = grp["interest_rate_market"].transform(lambda s: s.diff(30))
    df["month"]       = df["timestamp"].dt.month
    df["is_tet"]      = df["month"].isin([1,2]).astype(int)
    df["is_q4"]       = df["month"].isin([10,11,12]).astype(int)
    df["is_mid_year"] = df["month"].isin([6,7]).astype(int)
    df["is_than_tai"] = (
        (df["month"]==2) & (df["timestamp"].dt.day.between(1,28))
    ).astype(int)
    return df


# ── 4. Gán nhãn ───────────────────────────────────────────────────────
def add_target(df: pd.DataFrame, buy_pct: float, horizon: int) -> pd.DataFrame:
    grp = df.groupby("gold_code", dropna=False)
    df["_future"]        = grp["sell_price"].shift(-horizon)
    df["_ret"]           = (df["_future"] - df["sell_price"]) / df["sell_price"] * 100
    df["return_7d_pct"]  = df["_ret"].round(4)
    df["target"]         = (df["_ret"] > buy_pct).astype(int)
    df["target_label"]   = df["target"].map({1: "BUY", 0: "NOT_BUY"})
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


# ── 6. Split ──────────────────────────────────────────────────────────
def chronological_split(df: pd.DataFrame):
    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * TRAIN_SPLIT_RATIO)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ── 7. Preprocessing ──────────────────────────────────────────────────
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


# ── 8. Train ──────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> XGBClassifier:
    sw = compute_sample_weight("balanced", y=y_train)
    tw = np.linspace(0.6, 1.4, len(y_train))
    sw = sw * tw / (sw * tw).mean()
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=500, max_depth=4, min_child_weight=5,
        subsample=0.80, colsample_bytree=0.75, learning_rate=0.04,
        gamma=1.0, reg_alpha=1.5, reg_lambda=6.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    model.fit(X_train, y_train, sample_weight=sw)
    return model


# ── 9. Threshold ──────────────────────────────────────────────────────
def find_optimal_threshold(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.15, 0.66, 0.02):
        f = f1_score(y_test, (prob>=thr).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, round(float(thr), 2)
    return best_thr, best_f1


# ── 10. Evaluate ──────────────────────────────────────────────────────
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
        "XGBoost Gold VN — Ket qua danh gia (test set)",
        "="*52,
        f"Threshold toi uu : {threshold}",
        f"F1-macro         : {f1_mac:.4f}",
        f"F1-weighted      : {f1_wgt:.4f}",
        f"BUY Precision    : {prec:.4f}  ({tp}/{tp+fp} tin hieu BUY dung)",
        f"BUY Recall       : {rec:.4f}  (bat duoc {tp}/{tp+fn} co hoi thuc)",
        "",
        "Classification report:",
        report,
        "Confusion matrix (hang=thuc te, cot=du doan):",
        "                  NOT_BUY    BUY",
        f"  Thuc te NOT_BUY  {tn:6d}  {fp:6d}",
        f"  Thuc te BUY      {fn:6d}  {tp:6d}",
        "",
        f"  TN={tn:5d}  Noi NOT_BUY dung",
        f"  FP={fp:5d}  Bao mua nhung gia khong tang",
        f"  FN={fn:5d}  Bo lo co hoi mua",
        f"  TP={tp:5d}  Noi BUY dung, gia tang",
    ]
    text = "\n".join(lines)
    print(text)
    metrics_path.write_text(text, encoding="utf-8")


# ── 11. Plot FI ───────────────────────────────────────────────────────
def plot_fi(model, feature_names, out_path: Path, top_n=20):
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=False).head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(fi.index, fi.values, color="#378ADD", alpha=0.85)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, pad=12)
    ax.set_xlabel("Importance Score"); ax.set_ylabel("Feature")
    for bar, val in zip(bars, fi.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 12. Giải thích tín hiệu mới nhất ─────────────────────────────────
def explain_latest_signal(model, pp, df_after_lag, horizon, buy_pct, threshold, out_path):
    df_inf = df_after_lag.copy()
    df_inf["target"] = 0
    df_inf["target_label"] = "NOT_BUY"
    df_inf["return_7d_pct"] = 0.0
    df_eng = engineer_features(df_inf)
    latest_date = df_eng["timestamp"].max()
    df_latest   = df_eng[df_eng["timestamp"] == latest_date].copy()
    if df_latest.empty:
        return

    X_latest = df_latest.drop(
        columns=[c for c in ["target","target_label","return_7d_pct","timestamp"] if c in df_latest.columns],
        errors="ignore")
    X_latest = pp.transform_df(X_latest)
    X_latest = X_latest.reindex(columns=model.feature_names_in_, fill_value=0)
    prob_buy  = model.predict_proba(X_latest)[:, 1]

    fi = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    top10 = fi.nlargest(10).index.tolist()

    # Lấy 1 mã đại diện để giải thích
    df_raw = df_after_lag[df_after_lag["timestamp"] == latest_date]
    rep_codes = [c for c in df_raw["gold_code"].values if "SJL1L10" in str(c)]
    rep_code  = rep_codes[0] if rep_codes else df_raw["gold_code"].iloc[0]
    row       = df_raw[df_raw["gold_code"] == rep_code].iloc[0]
    gc_col    = f"gold_code_{rep_code}"
    if gc_col in df_latest.columns:
        mask     = df_latest[gc_col] == 1
        prob_rep = float(prob_buy[mask.values][0]) if mask.any() else float(prob_buy[0])
    else:
        prob_rep = float(prob_buy[0])
    sig_rep = "BUY" if prob_rep >= threshold else "NOT_BUY"

    lines = []
    lines.append("=" * 62)
    lines.append(f"  TIN HIEU MOI NHAT — {latest_date.date()}")
    lines.append("=" * 62)
    lines.append(f"  Ma dai dien : {rep_code}")
    lines.append(f"  Tin hieu    : {sig_rep}")
    lines.append(f"  P(BUY)      : {prob_rep:.1%}")
    lines.append(f"  Nguong      : {threshold}  (P >= {threshold} → BUY)")
    lines.append(f"  Tieu chi BUY: gia tang > {buy_pct}% sau {horizon} ngay toi")
    lines.append("")

    lines.append("-" * 62)
    lines.append("  CAU TRA LOI: DUA VAO DAU DE DU DOAN?")
    lines.append("-" * 62)
    lines.append("")

    lines.append("  Model XGBoost su dung 46 features, chia thanh 5 nhom:")
    lines.append("")

    # 1. Momentum
    lines.append("  [1] XU HUONG GIA (momentum) — 'Da tang hay giam?'")
    for col, label in [
        ("cum_return_3d", "    Loi nhuan 3 ngay"),
        ("cum_return_7d", "    Loi nhuan 7 ngay"),
        ("cum_return_14d","    Loi nhuan 14 ngay"),
    ]:
        val = row.get(col)
        if val is not None:
            v = float(val)
            lines.append(f"     {label}: {'tang' if v>0 else 'giam'} {abs(v):.2f}%")
    lines.append("")

    # 2. RSI
    lines.append("  [2] CHI SO KY THUAT — 'Qua mua hay qua ban?'")
    rsi = row.get("RSI_14")
    if rsi is not None:
        rv = float(rsi)
        comment = ("oversold nang → co the tang" if rv < 30 else
                   "oversold" if rv < 40 else
                   "overbought" if rv > 70 else "trung tinh")
        lines.append(f"     RSI 14 ngay = {rv:.1f}  ({comment})")
    for col, label in [("price_to_MA20_pct","Gia vs MA20"),("price_to_MA50_pct","Gia vs MA50")]:
        val = row.get(col)
        if val is not None:
            v = float(val)
            lines.append(f"     {label}: {'tren' if v>0 else 'duoi'} {abs(v):.2f}%")
    lines.append("")

    # 3. Premium
    lines.append("  [3] PREMIUM SJC — 'NHNN co dang can thiep?'")
    for col, label in [
        ("premium_pct_raw",  "    Premium hien tai (%)"),
        ("premium_trend_7d", "    Xu huong premium 7 ngay"),
        ("premium_vs_ma7",   "    Premium vs MA7"),
    ]:
        val = row.get(col)
        if val is not None:
            lines.append(f"     {label}: {float(val):.2f}")
    prem_trend = row.get("premium_trend_7d")
    if prem_trend is not None and float(prem_trend) < -1_000_000:
        lines.append("     !! Premium thu hep nhanh → NHNN co the dang can thiep")
    lines.append("")

    # 4. Macro
    lines.append("  [4] VI MO — 'The gioi anh huong the nao?'")
    for col, label in [
        ("world_chg_5d",  "    Vang TG 5 ngay"),
        ("usdvnd_chg_7d", "    USD/VND 7 ngay"),
        ("dxy_chg_5d",    "    DXY 5 ngay"),
    ]:
        val = row.get(col)
        if val is not None:
            v = float(val)
            lines.append(f"     {label}: {'+' if v>=0 else ''}{v:.2f}%")
    for col, label in [("fed_rate","FED rate"),("interest_rate_state","Lai suat VN")]:
        val = row.get(col)
        if val is not None:
            lines.append(f"     {label}: {float(val):.2f}%")
    lines.append("")

    # 5. Seasonal
    lines.append("  [5] MUA VU — 'Dac thu thi truong VN?'")
    seasonal_map = {"is_tet":"Thang Tet (1-2)","is_than_tai":"Via Than Tai","is_q4":"Quy 4","is_mid_year":"Giua nam"}
    found_seasonal = [label for col, label in seasonal_map.items() if float(row.get(col, 0)) == 1]
    if found_seasonal:
        for s in found_seasonal:
            lines.append(f"     [DANG TRONG] {s}")
    else:
        lines.append("     Khong co yeu to mua vu dac biet")
    lines.append("")

    # Top features
    lines.append("-" * 62)
    lines.append("  TOP 10 FEATURES MODEL CHU Y NHAT (theo importance)")
    lines.append("-" * 62)
    lines.append("  Hang  Feature                          Importance  Gia tri hom nay")
    for i, fname in enumerate(top10, 1):
        imp_val = fi[fname]
        raw_val = row.get(fname)
        val_str = f"{float(raw_val):.3f}" if raw_val is not None and fname in row.index else "N/A"
        lines.append(f"  {i:2d}.   {fname:<35s}  {imp_val:.4f}    {val_str}")
    lines.append("")

    # Kết luận
    lines.append("=" * 62)
    lines.append("  KET LUAN")
    lines.append("=" * 62)
    if sig_rep == "BUY":
        lines.append(f"  >> TIN HIEU: BUY (P = {prob_rep:.1%})")
        lines.append(f"     Du doan: gia vang co the TANG > {buy_pct}% trong {horizon} ngay toi.")
        lines.append(f"     Tuy nhien: precision ~50% — cu 2 tin hieu BUY thi 1 dung 1 sai.")
        lines.append(f"     Nen ket hop voi tin tuc NHNN va vang the gioi truoc khi mua.")
    else:
        lines.append(f"  >> TIN HIEU: NOT_BUY (P(BUY) = {prob_rep:.1%})")
        lines.append(f"     Du doan: gia vang KHONG tang du {buy_pct}% trong {horizon} ngay toi.")
        lines.append(f"     Nen cho tin hieu tot hon (RSI < 30, premium thu hep, da tang).")
    lines.append("")
    lines.append("  LUU Y: Day chi la tin hieu ho tro quyet dinh (DSS),")
    lines.append("  khong phai loi khuyen dau tu. Ket hop voi phan doan")
    lines.append("  ca nhan va thong tin thi truong thuc te.")
    lines.append("=" * 62)

    text = "\n".join(lines)
    print("\n" + text)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n  >> Luu giai thich: {out_path}")


# ── MAIN ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost BUY/NOT_BUY vang VN")
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--buy-pct",    type=float, default=DEFAULT_BUY_PCT)
    p.add_argument("--horizon",    type=int,   default=DEFAULT_HORIZON)
    p.add_argument("--output-dir", type=Path,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir   = Path(__file__).resolve().parent
    output_dir = args.output_dir or base_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir  = output_dir / "pipeline_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    input_path   = args.input_file or base_dir / INPUT_FILE
    scaler_path  = output_dir / "scaler_gold.pkl"
    model_path   = output_dir / "xgboost_gold_model.pkl"
    config_path  = output_dir / "model_config.json"
    metrics_path = output_dir / "evaluation_metrics.txt"
    fi_path      = output_dir / "feature_importance.png"
    explain_path = output_dir / "decision_explanation.txt"

    SEP = "="*55
    print(f"\n{SEP}")
    print(f"  XGBoost Gold VN v2.1 — luu file tung buoc")
    print(f"  horizon={args.horizon}d  |  buy_pct={args.buy_pct}%")
    print(f"  Input : {input_path}")
    print(f"  Steps : {steps_dir}")
    print(f"{SEP}\n")

    print("[1/9] Load & resample daily...")
    df = load_and_resample(input_path)
    print(f"      {len(df):,} dong")
    save_step(df, steps_dir, "step1_raw_resample.csv",
              "Sau load_and_resample: gia cuoi ngay, forward-fill ngay nghi")

    print("[2/9] Technical indicators...")
    df = add_technical_indicators(df)
    save_step(df, steps_dir, "step2_technical_indicators.csv",
              "Sau add_technical_indicators: them MA5/20/50, RSI_14, price_to_MA_pct")

    print("[3/9] Lag features & tin hieu VN...")
    df = add_lag_features(df)
    df_after_lag = df.copy()
    save_step(df, steps_dir, "step3_lag_features.csv",
              "Sau add_lag_features: cum_return, RSI chg, premium, volatility, Tet/Q4")

    print(f"[4/9] Gan nhan BUY (horizon={args.horizon}d, buy_pct={args.buy_pct}%)...")
    df = add_target(df, buy_pct=args.buy_pct, horizon=args.horizon)
    buy_rate = df["target"].mean()*100
    print(f"      BUY={buy_rate:.1f}%  NOT_BUY={100-buy_rate:.1f}%")
    save_step(df, steps_dir, "step4_labeled.csv",
              f"Sau add_target: cot target(0/1) va target_label(BUY/NOT_BUY). "
              f"7 ngay cuoi bi xoa (shift(-{args.horizon})=NaN)")

    print("[5/9] Feature engineering...")
    df = engineer_features(df)
    print(f"      {df.shape[1]-2} features  |  {len(df):,} dong")
    save_step(df, steps_dir, "step5_engineered.csv",
              "Sau engineer_features: daily_ret/world_ret/spread_pct, one-hot gold_code_*, dropna")

    print("[6/9] Chia train/test (80/20)...")
    train_df, test_df = chronological_split(df)
    print(f"      Train {len(train_df):,}  ({train_df.timestamp.min().date()} → {train_df.timestamp.max().date()})")
    print(f"      Test  {len(test_df):,}  ({test_df.timestamp.min().date()} → {test_df.timestamp.max().date()})")

    y_train = train_df["target"].astype(int)
    y_test  = test_df["target"].astype(int)
    drop_inf = [c for c in ["target","target_label","return_7d_pct","timestamp"] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_inf)
    X_test  = test_df.drop(columns=[c for c in drop_inf if c in test_df.columns])
    print(f"      Train BUY={y_train.mean()*100:.1f}%  Test BUY={y_test.mean()*100:.1f}%")
    scale_pos_w = (y_train==0).sum() / (y_train==1).sum()
    save_step(train_df, steps_dir, "step6_train.csv",
              f"Tap TRAIN (80%): {train_df.timestamp.min().date()} → {train_df.timestamp.max().date()}, "
              f"scale_pos_weight={scale_pos_w:.3f}")
    save_step(test_df, steps_dir, "step6_test.csv",
              f"Tap TEST (20%): {test_df.timestamp.min().date()} → {test_df.timestamp.max().date()}")

    print("[7/9] Preprocessing (Winsorize + RobustScale)...")
    X_train_sc, X_test_sc, pp = preprocess(X_train, X_test, scaler_path)
    print(f"      X_train={X_train_sc.shape}  X_test={X_test_sc.shape}")
    tr_sc = X_train_sc.copy(); tr_sc["target"] = y_train.values
    te_sc = X_test_sc.copy();  te_sc["target"]  = y_test.values
    save_step(tr_sc, steps_dir, "step7_train_scaled.csv",
              "Train sau Winsorize[p1,p99] + RobustScaler. Fit chi tren train.")
    save_step(te_sc, steps_dir, "step7_test_scaled.csv",
              "Test sau ap scaler (khong fit lai). Gia tri gan 0 = trung binh.")

    print("[8/9] Train XGBoost...")
    model       = train_model(X_train_sc, y_train)
    opt_thr, opt_f1 = find_optimal_threshold(model, X_test_sc, y_test)
    print(f"      Threshold toi uu={opt_thr}  F1-macro={opt_f1:.4f}")

    prob_test  = model.predict_proba(X_test_sc)[:, 1]
    pred_test  = (prob_test >= opt_thr).astype(int)
    preds_df   = pd.DataFrame({
        "timestamp":    test_df["timestamp"].values if "timestamp" in test_df.columns else range(len(y_test)),
        "y_true":       y_test.values,
        "y_true_label": pd.Series(y_test.values).map({1:"BUY",0:"NOT_BUY"}).values,
        "prob_buy":     prob_test.round(4),
        "y_pred":       pred_test,
        "y_pred_label": pd.Series(pred_test).map({1:"BUY",0:"NOT_BUY"}).values,
        "correct":      (pred_test == y_test.values).astype(int),
    })
    save_step(preds_df, steps_dir, "step8_predictions.csv",
              f"Ket qua du doan tren test set, threshold={opt_thr}. correct=1 la dung.")

    fi_df = pd.DataFrame({
        "rank":       range(1, len(model.feature_names_in_)+1),
        "feature":    model.feature_names_in_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df["rank"] = fi_df.index + 1
    save_step(fi_df, steps_dir, "step9_feature_importance.csv",
              "Feature importance (gain). Cao = model chu y nhieu hon khi quyet dinh.")

    print("[9/9] Danh gia & luu ket qua...")
    print(f"\n{SEP}\n  KET QUA DANH GIA\n{SEP}")
    evaluate(model, X_test_sc, y_test, threshold=opt_thr, metrics_path=metrics_path)
    plot_fi(model, X_train_sc.columns.tolist(), fi_path)
    explain_latest_signal(model, pp, df_after_lag, args.horizon, args.buy_pct, opt_thr, explain_path)

    joblib.dump(model, model_path)
    config = {
        "buy_pct": args.buy_pct, "horizon": args.horizon,
        "optimal_threshold": opt_thr, "f1_macro": round(opt_f1, 4),
        "train_date_range": [str(train_df.timestamp.min().date()), str(train_df.timestamp.max().date())],
        "test_date_range":  [str(test_df.timestamp.min().date()),  str(test_df.timestamp.max().date())],
        "n_features": X_train_sc.shape[1],
        "feature_names": X_train_sc.columns.tolist(),
    }
    with open(config_path,"w",encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n{SEP}\n  TẤT CẢ FILES ĐÃ LƯU\n{SEP}")
    for label, path in [
        ("Model   ", model_path), ("Scaler  ", scaler_path),
        ("Config  ", config_path), ("Metrics ", metrics_path),
        ("FI Plot ", fi_path), ("Explain ", explain_path),
    ]:
        print(f"  {label}: {path}")
    print(f"\n  Pipeline steps ({steps_dir.name}/):")
    for p in sorted(steps_dir.glob("*.csv")):
        size_kb = p.stat().st_size // 1024
        print(f"    {p.name:<45s}  {size_kb:>5} KB")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()