"""
Backend Flask: load model XGBoost, chạy pipeline trên master, trả API cho frontend visualize.
Chạy từ thư mục gốc: python frontend/app.py (hoặc cd frontend && python app.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Thêm New folder để import train_xgboost_dss, root để import news/sentiment
ROOT = Path(__file__).resolve().parent.parent
NEW_FOLDER = ROOT / "New folder"
if str(NEW_FOLDER) not in sys.path:
    sys.path.insert(0, str(NEW_FOLDER))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time as _time

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder=Path(__file__).resolve().parent / "templates", static_folder="static")

# Đường dẫn mặc định: model và data
MODEL_DIR = NEW_FOLDER / "output"
MODEL_PATH = MODEL_DIR / "xgboost_dss_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler_xgboost_dss.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder_dss.pkl"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
_DEFAULT_LABELS = ["BUY", "HOLD", "SELL"]


def _load_model_config() -> dict:
    """Load model_config.json saved at training time. Returns empty dict if missing."""
    try:
        if MODEL_CONFIG_PATH.exists():
            import json as _json
            with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
                return _json.load(f)
    except Exception:
        pass
    return {}


def _get_labels() -> list[str]:
    """Load labels từ label_encoder đã lưu, fallback về 3-class mặc định."""
    try:
        if LABEL_ENCODER_PATH.exists():
            le = joblib.load(LABEL_ENCODER_PATH)
            return [str(c) for c in le.classes_]
    except Exception:
        pass
    return list(_DEFAULT_LABELS)
MAX_POINTS = 90  # Số điểm trả về cho chart (90 ngày gần nhất)

# Master CSV: ưu tiên file có nhiều ngày (thư mục gốc thường có từ tháng 1) để chart đủ 90 ngày
_master_path_cache: Path | None = None
_gold_codes_cache: list | None = None


def _normalize_master_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột master gốc (World_Price_VND...) sang tên pipeline cần (world_price_vnd...)."""
    rename = {"World_Price_VND": "world_price_vnd", "Domestic_Premium": "domestic_premium"}
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _get_model_gold_codes() -> set[str]:
    """Trả về set gold_code mà model đã train (từ feature_names_in_)."""
    try:
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "feature_names_in_"):
            return {c.replace("gold_code_", "") for c in model.feature_names_in_ if c.startswith("gold_code_")}
    except Exception:
        pass
    return set()


def _get_master_path() -> Path:
    """Ưu tiên file master có gold_code khớp model; nếu bằng nhau thì chọn file nhiều ngày hơn."""
    global _master_path_cache
    if _master_path_cache is not None:
        return _master_path_cache

    model_codes = _get_model_gold_codes()
    root_master = ROOT / "master_dss_dataset.csv"
    new_master = NEW_FOLDER / "master_dss_dataset.csv"

    candidates = []
    for path in (new_master, root_master):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, nrows=200000)
            df = _normalize_master_columns(df)
            ndays = pd.to_datetime(df["timestamp"], errors="coerce").dt.date.nunique()
            file_codes = set(df["gold_code"].dropna().unique()) if "gold_code" in df.columns else set()
            overlap = len(model_codes & file_codes) if model_codes else 0
            candidates.append((overlap, ndays, path, df))
        except Exception:
            pass

    if not candidates:
        _master_path_cache = new_master if new_master.exists() else root_master
        return _master_path_cache

    # Chọn file có overlap gold_code cao nhất; tie-break bằng số ngày
    best = max(candidates, key=lambda t: (t[0], t[1]))
    best_path, best_df = best[2], best[3]

    # Nếu cần normalize (root master), cache lại file đã normalize
    if best_path == root_master and any(c in best_df.columns for c in ("World_Price_VND", "Domestic_Premium")):
        cache_path = ROOT / ".frontend_master_cache.csv"
        best_df.to_csv(cache_path, index=False)
        best_path = cache_path

    _master_path_cache = best_path
    return best_path


def _get_gold_codes():
    global _gold_codes_cache
    if _gold_codes_cache is None:
        try:
            path = _get_master_path()
            df = pd.read_csv(path, usecols=["gold_code"], nrows=200000)
            _gold_codes_cache = df["gold_code"].dropna().unique().tolist()
        except Exception:
            _gold_codes_cache = []
    return _gold_codes_cache



def _run_pipeline_and_predict(gold_code: str | None = None):
    from train_xgboost_dss import (
        add_lag_features,
        add_technical_indicators,
        add_t3_target,
        engineer_features,
        load_and_resample_daily,
    )

    LABELS = _get_labels()
    cfg = _load_model_config()
    binary_mode = cfg.get("binary", LABELS == ["BUY", "NOT_BUY"])
    buy_pct = cfg.get("buy_pct", None)
    hold_band = cfg.get("hold_band_ratio", 0.15)
    buy_ratio = cfg.get("buy_ratio", 1.0)
    sell_ratio = cfg.get("sell_ratio", 0.3)
    optimal_threshold = cfg.get("optimal_threshold", 0.5)
    horizon = int(cfg.get("horizon", 3))

    df = load_and_resample_daily(_get_master_path())
    # Nếu chọn 1 mã vàng: lọc để mỗi ngày = 1 dòng = 1 lần dự đoán
    if gold_code and gold_code.strip():
        allowed = _get_gold_codes()
        if gold_code not in allowed:
            return None
        df = df[df["gold_code"] == gold_code].copy()
        if df.empty:
            return None

    df = add_technical_indicators(df)
    # Lag features phải tính trên FULL dataset (trước khi lọc 3 ngày cuối)
    df = add_lag_features(df)

    # Giá theo ngày lấy TRƯỚC add_t3_target để giữ đủ ngày
    daily_price = df.groupby("timestamp")["sell_price"].mean().reset_index()
    daily_price.columns = ["date", "price"]

    le = LabelEncoder()
    le.fit(LABELS)
    # Dùng đúng labeling config đã train
    df_with_target, _ = add_t3_target(
        df.copy(),
        hold_band_ratio=hold_band,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        buy_pct=buy_pct,
        binary=binary_mode,
        horizon=horizon,
    )
    df_with_target = engineer_features(df_with_target)
    if df_with_target.empty:
        return None

    target_col = "target_encoded"
    if target_col not in df_with_target.columns:
        return None

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(SCALER_PATH)

    def _predict_block(df_block: pd.DataFrame) -> pd.Series:
        """Reindex → scale (column-safe) → predict with optimal threshold."""
        X = df_block.drop(columns=[target_col], errors="ignore").copy()
        X = X.drop(columns=["timestamp"], errors="ignore")
        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        if hasattr(preprocessor, "transform_df"):
            X = preprocessor.transform_df(X)
        # Use optimal threshold for binary BUY/NOT_BUY
        if binary_mode and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            buy_col = classes.index(
                list(LABELS).index("BUY") if "BUY" in LABELS else 0
            )
            prob_buy = proba[:, buy_col]
            thr = optimal_threshold if optimal_threshold != 0.5 else 0.5
            pred_encoded = np.where(prob_buy >= thr, classes[buy_col], classes[1 - buy_col])
            return pd.Series(pred_encoded, index=df_block.index), pd.Series(prob_buy, index=df_block.index)
        preds = pd.Series(model.predict(X), index=df_block.index)
        return preds, pd.Series(np.nan, index=df_block.index)

    df_with_target["prediction"], df_with_target["prob_buy"] = _predict_block(df_with_target)

    # Inference cho 3 ngày cuối (không cần data 3 ngày sau):
    # df đã có lag features từ add_lag_features trên toàn bộ dataset
    last_dates = sorted(df["timestamp"].dropna().unique())[-3:]
    df_last = df[df["timestamp"].isin(last_dates)].copy()
    if not df_last.empty:
        df_last["future_sell_price_3d"] = df_last["sell_price"].values  # placeholder
        df_last["expected_profit"] = 0.0
        df_last["current_spread"] = (df_last["sell_price"] - df_last["buy_price"]).values
        df_last["target_trend"] = "NOT_BUY" if binary_mode else "HOLD"
        df_last["target_encoded"] = 1
        try:
            df_last = engineer_features(df_last)
            if not df_last.empty and target_col in df_last.columns:
                df_last["prediction"], df_last["prob_buy"] = _predict_block(df_last)
                df_with_target = pd.concat([df_with_target, df_last], ignore_index=True)
        except Exception:
            pass

    df = df_with_target
    if gold_code and gold_code.strip():
        pred_per_day = df[["timestamp", "prediction"]].drop_duplicates("timestamp")
        pred_per_day.columns = ["date", "prediction"]
    else:
        pred_per_day = df.groupby("timestamp")["prediction"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]
        ).reset_index()
        pred_per_day.columns = ["date", "prediction"]

    # Left merge để giữ mọi ngày có giá (kể cả 3 ngày cuối chưa có dự đoán)
    # Also aggregate prob_buy per day
    if "prob_buy" in df.columns:
        prob_per_day = df.groupby("timestamp")["prob_buy"].mean().reset_index()
        prob_per_day.columns = ["date", "prob_buy"]
        pred_per_day = pred_per_day.merge(prob_per_day, on="date", how="left")
    merge = daily_price.merge(pred_per_day, on="date", how="left")
    merge = merge.sort_values("date").tail(MAX_POINTS)
    merge["prediction_label"] = merge["prediction"].apply(
        lambda i: LABELS[int(i)] if pd.notna(i) else None
    )
    merge["prob_buy_label"] = merge.get("prob_buy", None)

    # Dự đoán mới nhất = dòng cuối có prediction (3 ngày cuối không có do shift(-3))
    last_with_pred = merge[merge["prediction"].notna()]
    latest_pred = last_with_pred["prediction_label"].iloc[-1] if len(last_with_pred) else None
    latest_pred_date = last_with_pred["date"].iloc[-1] if len(last_with_pred) else None

    def _safe_prob(val):
        try:
            v = float(val)
            return round(v, 3) if not np.isnan(v) else None
        except Exception:
            return None

    return {
        "gold_code": gold_code.strip() if gold_code and gold_code.strip() else None,
        "gold_codes": _get_gold_codes(),
        "labels": LABELS,
        "dates": merge["date"].astype(str).tolist(),
        "prices": merge["price"].round(0).tolist(),
        "predictions": [p if p is not None else "N/A" for p in merge["prediction_label"]],
        "probabilities": [_safe_prob(p) for p in merge.get("prob_buy_label", [None]*len(merge))],
        "latest": {
            "date": str(merge["date"].iloc[-1]),
            "price": float(merge["price"].iloc[-1]),
            "prediction": latest_pred,
            "prediction_date": str(latest_pred_date) if latest_pred_date is not None else None,
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/gold_codes")
def api_gold_codes():
    return jsonify({"gold_codes": _get_gold_codes()})


@app.route("/api/macro")
def api_macro():
    """Trả dữ liệu lãi suất FED, DXY, tỷ giá USD/VND, giá vàng thế giới để vẽ biểu đồ macro."""
    macro_candidates = [
        ROOT / "MACRO_FEATURES_1_YEAR.csv",
        ROOT / "input_1year" / "MACRO_FEATURES_1_YEAR.csv",
    ]
    macro_path = next((p for p in macro_candidates if p.exists()), None)
    if macro_path is None:
        return jsonify({"error": "Macro file not found. Run fetch_macro_1_year.py first."}), 404

    try:
        df = pd.read_csv(macro_path)
        date_col = next((c for c in df.columns if c.lower() in ("date", "ngay", "timestamp")), None)
        if date_col is None:
            return jsonify({"error": "No date column in macro file."}), 500

        df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)

        def col_or_none(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        fed_col   = col_or_none(["fed_rate", "Fed_Rate"])
        dxy_col   = col_or_none(["dxy_index", "DXY_Index"])
        vn_col    = col_or_none(["interest_rate_market", "interest_rate_state"])
        usd_col   = col_or_none(["usd_vnd_rate", "USD_VND_Rate"])
        world_col = col_or_none(["World_Price_USD_Ounce", "world_price_usd_ounce"])

        def to_list(col):
            if col and col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").round(4).tolist()
            return []

        return jsonify({
            "dates":       df["_date"].dt.strftime("%Y-%m-%d").tolist(),
            "fed_rate":    to_list(fed_col),
            "dxy_index":   to_list(dxy_col),
            "vn_rate":     to_list(vn_col),
            "usd_vnd":     to_list(usd_col),
            "world_gold":  to_list(world_col),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict")
def api_predict():
    gold_code = request.args.get("gold_code", "").strip() or None
    try:
        out = _run_pipeline_and_predict(gold_code=gold_code)
        if out is None:
            return jsonify({"error": "No data or pipeline failed"}), 500
        return jsonify(out)
    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {e}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_news_cache = None
_news_cache_ts = 0.0
_NEWS_CACHE_TTL = 1800


@app.route("/api/news")
def api_news():
    """Return gold-related news with sentiment analysis, cached for 30 min."""
    global _news_cache, _news_cache_ts

    source_filter = request.args.get("source", "").strip().lower()

    if _news_cache is not None and (_time.time() - _news_cache_ts) < _NEWS_CACHE_TTL:
        articles = _news_cache
    else:
        try:
            from news_scraper import fetch_all_gold_news
            from sentiment_analyzer import analyze_articles
            raw = fetch_all_gold_news(max_per_source=15)
            articles = [r.to_dict() for r in analyze_articles(raw)]
            _news_cache = articles
            _news_cache_ts = _time.time()
        except Exception as e:
            return jsonify({"error": str(e), "articles": []}), 500

    if source_filter and source_filter != "all":
        filtered = [a for a in articles if a["source"].lower() == source_filter]
    else:
        filtered = articles

    scores = [a["sentiment_score"] for a in filtered if a.get("sentiment_score") is not None]
    overall = round(sum(scores) / len(scores), 3) if scores else 0.0
    pos = sum(1 for s in scores if s > 0.15)
    neg = sum(1 for s in scores if s < -0.15)
    neu = len(scores) - pos - neg

    return jsonify({
        "articles": filtered[:20],
        "overall_sentiment": overall,
        "summary": {"positive": pos, "negative": neg, "neutral": neu, "total": len(filtered)},
        "sources": sorted(set(a["source"] for a in articles)),
    })


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("Model not found at", MODEL_PATH, "- train first (New folder/train_xgboost_dss.py)")
    else:
        print("Open http://127.0.0.1:5000 in browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
