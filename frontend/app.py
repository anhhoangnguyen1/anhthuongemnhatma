"""
Backend Flask: load model XGBoost, chạy pipeline trên master, trả API cho frontend visualize.
Chạy từ thư mục gốc: python frontend/app.py (hoặc cd frontend && python app.py)

Env (cho LLM đánh giá tin): đọc từ .env hoặc biến môi trường hệ thống.
Xem .env.example và thiết lập OPENAI_API_KEY, GNEWS_API_KEY.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Load .env nếu có (để OPENAI_API_KEY, GNEWS_API_KEY có sẵn cho llm_adjust)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()  # thử .env ở cwd
except ImportError:
    pass

# Thêm New folder để import train_xgboost_dss
ROOT = Path(__file__).resolve().parent.parent
NEW_FOLDER = ROOT / "New folder"
if str(NEW_FOLDER) not in sys.path:
    sys.path.insert(0, str(NEW_FOLDER))

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
    new_master = NEW_FOLDER / "master_dss_dataset.csv"

    candidates = []
    for path in (new_master,):
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
        _master_path_cache = new_master
        return _master_path_cache

    # Chọn file có overlap gold_code cao nhất; tie-break bằng số ngày
    best = max(candidates, key=lambda t: (t[0], t[1]))
    best_path, best_df = best[2], best[3]

    # Nếu master có cột tên cũ (World_Price_VND, Domestic_Premium), ghi bản đã chuẩn hóa ra cache
    if any(c in best_df.columns for c in ("World_Price_VND", "Domestic_Premium")):
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

    # Daily features cho biểu đồ (aggregate theo ngày từ df trước add_t3_target)
    feat_cols = ["domestic_premium", "RSI_14", "cum_return_3d_pct", "cum_return_7d_pct", "sell_price", "world_price_vnd"]
    available = [c for c in feat_cols if c in df.columns]
    daily_feat = None
    if available:
        daily_feat = df.groupby("timestamp")[available].mean().reset_index()
        daily_feat.columns = ["date"] + available
        if "sell_price" in daily_feat.columns and "world_price_vnd" in daily_feat.columns:
            w = daily_feat["world_price_vnd"].replace(0, np.nan)
            daily_feat["premium_pct"] = (daily_feat["sell_price"] / w) * 100.0

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

    # Gắn feature theo ngày (cùng thứ tự với merge) cho biểu đồ phía dưới
    feature_series = {}
    if daily_feat is not None:
        feat_merge = merge[["date"]].merge(daily_feat, on="date", how="left")
        for col in ["premium_pct", "domestic_premium", "RSI_14", "cum_return_3d_pct", "cum_return_7d_pct", "world_price_vnd"]:
            if col in feat_merge.columns:
                feature_series[col] = [float(x) if pd.notna(x) else None for x in feat_merge[col]]

    # Dự đoán mới nhất = dòng cuối có prediction (3 ngày cuối không có do shift(-3))
    last_with_pred = merge[merge["prediction"].notna()]
    latest_pred = last_with_pred["prediction_label"].iloc[-1] if len(last_with_pred) else None
    latest_pred_date = last_with_pred["date"].iloc[-1] if len(last_with_pred) else None
    # Giá tại ngày dự đoán (để LLM đánh giá cùng tin tức ngày đó)
    latest_pred_date_price = None
    if latest_pred_date is not None and len(merge[merge["date"] == latest_pred_date]["price"]) > 0:
        latest_pred_date_price = float(merge[merge["date"] == latest_pred_date]["price"].iloc[0])

    def _safe_prob(val):
        try:
            v = float(val)
            return round(v, 3) if not np.isnan(v) else None
        except Exception:
            return None

    out = {
        "gold_code": gold_code.strip() if gold_code and gold_code.strip() else None,
        "gold_codes": _get_gold_codes(),
        "labels": LABELS,
        "dates": merge["date"].astype(str).tolist(),
        "prices": merge["price"].round(0).tolist(),
        "predictions": [p if p is not None else "N/A" for p in merge["prediction_label"]],
        "probabilities": [_safe_prob(p) for p in merge.get("prob_buy_label", [None]*len(merge))],
        "features": feature_series,
        "latest": {
            "date": str(merge["date"].iloc[-1]),
            "price": float(merge["price"].iloc[-1]),
            "prediction": latest_pred,
            "prediction_date": str(latest_pred_date) if latest_pred_date is not None else None,
            "prediction_date_price": latest_pred_date_price,
        },
    }

    # Mô hình B: sau khi có dự đoán ML, lấy tin ngày dự đoán → LLM đánh giá và điều chỉnh
    if request.args.get("llm") in ("1", "true", "yes"):
        try:
            try:
                from frontend.llm_adjust import run_llm_adjust_for_latest
            except ImportError:
                from llm_adjust import run_llm_adjust_for_latest
            llm_result = run_llm_adjust_for_latest(
                ROOT,
                str(latest_pred_date) if latest_pred_date is not None else None,
                latest_pred,
                latest_pred_date_price,
            )
            if llm_result is not None:
                out["latest"]["llm_adjusted"] = llm_result
        except Exception as e:
            out["latest"]["llm_adjusted"] = {
                "adjusted_signal": latest_pred,
                "reasoning": f"Lỗi: {e}",
                "confidence": 0.5,
                "updated_price_note": "",
            }
    return out


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


@app.route("/api/date-detail")
def api_date_detail():
    """
    Khi user chọn 1 ngày: trả về tin realtime + dự đoán bổ sung từ LLM (chỉ dựa trên tin).
    Query: date=YYYY-MM-DD, gold_code= (tùy chọn).
    """
    date_str = (request.args.get("date") or "").strip()[:10]
    gold_code = request.args.get("gold_code", "").strip() or None
    if not date_str:
        return jsonify({"error": "Thiếu tham số date (YYYY-MM-DD)"}), 400
    try:
        out = _run_pipeline_and_predict(gold_code=gold_code)
        if out is None:
            return jsonify({"error": "No data or pipeline failed"}), 500
        dates = out.get("dates") or []
        prices = out.get("prices") or []
        predictions = out.get("predictions") or []
        idx = None
        for i, d in enumerate(dates):
            if (d[:10] if isinstance(d, str) else str(d)[:10]) == date_str:
                idx = i
                break
        if idx is None:
            return jsonify({"error": f"Không có dữ liệu cho ngày {date_str}"}), 404
        price = float(prices[idx]) if idx < len(prices) and prices[idx] is not None else 0.0
        ml_pred = predictions[idx] if idx < len(predictions) else "N/A"
        if ml_pred is None:
            ml_pred = "N/A"

        try:
            try:
                from frontend.llm_adjust import get_news_and_llm_supplement_for_date
            except ImportError:
                from llm_adjust import get_news_and_llm_supplement_for_date
        except ImportError:
            return jsonify({"error": "Module llm_adjust không tìm thấy"}), 500

        import os
        api_key = os.getenv("OPENAI_API_KEY")
        gnews_key = os.getenv("GNEWS_API_KEY")
        if not api_key or not gnews_key:
            return jsonify({
                "date": date_str,
                "price": price,
                "ml_prediction": ml_pred,
                "news": [],
                "llm_supplement": None,
                "error": "Thiếu OPENAI_API_KEY hoặc GNEWS_API_KEY trong .env",
            }), 200

        from datetime import date as date_type
        try:
            d = date_type.fromisoformat(date_str)
        except ValueError:
            return jsonify({"error": f"Ngày không hợp lệ: {date_str}"}), 400
        detail = get_news_and_llm_supplement_for_date(
            prediction_date=d,
            price_vnd=price,
            api_key=api_key,
            gnews_key=gnews_key,
        )
        return jsonify({
            "date": date_str,
            "price": price,
            "ml_prediction": ml_pred,
            "news": detail.get("news") or [],
            "llm_supplement": detail.get("llm_supplement"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Đồng bộ dữ liệu mới từ Google Sheet vào master (nếu có GOOGLE_SHEET_ID + credentials)
    try:
        from frontend.sheet_sync import sync_master_from_google_sheet
    except ImportError:
        from sheet_sync import sync_master_from_google_sheet
    master_path = NEW_FOLDER / "master_dss_dataset.csv"
    synced = sync_master_from_google_sheet(master_path)
    if synced is None:
        print("[Sheet sync] Sync bỏ qua (xem log trên để biết lý do).")
    elif synced == 0:
        print("[Sheet sync] Không có dòng mới từ Sheet so với master.")
    elif synced > 0:
        g = globals()
        g["_master_path_cache"] = None
        g["_gold_codes_cache"] = None
    # Nếu synced is None: không cấu hình Sheet hoặc lỗi, bỏ qua

    if not MODEL_PATH.exists():
        print("Model not found at", MODEL_PATH, "- train first (New folder/train_xgboost_dss.py)")
    else:
        print("Open http://127.0.0.1:5000 in browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
