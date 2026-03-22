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

# Luôn dùng training/ làm nguồn model + script train thực tế.
ROOT = Path(__file__).resolve().parent.parent
NEW_FOLDER = ROOT / "training"
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
MODEL_PATH = MODEL_DIR / "xgboost_gold_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler_gold.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder_dss.pkl"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
# QUAN TRỌNG: thứ tự khớp với model.classes_ = [0, 1] => [NOT_BUY, BUY]
# LABELS[0]="NOT_BUY", LABELS[1]="BUY" — dùng LABELS[prediction_int] mới đúng
_DEFAULT_LABELS = ["NOT_BUY", "BUY"]
GOLD_GROUP_MAP: dict[str, set[str]] = {
    "SJC_BAR": {"SJL1L10", "BTSJC", "VNGSJC", "VIETTINMSJC"},
    "JEWELRY_9999": {"BT9999NTT", "PQHN24NTT"},
    "DOJI_PLAIN_RING": {"DOJINHTV"},
}


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
            # Chỉ expose 3 nhóm cố định cho UI (cộng thêm "trung bình tất cả" ở frontend)
            _gold_codes_cache = list(GOLD_GROUP_MAP.keys())
        except Exception:
            _gold_codes_cache = list(GOLD_GROUP_MAP.keys())
    return _gold_codes_cache



def _run_pipeline_and_predict(gold_code: str | None = None):
    # Fix 3: import từ train_xgboost_gold (mới), fallback train_xgboost_dss (cũ)
    try:
        from train_xgboost_gold import (
            add_target,
            add_lag_features,
            add_technical_indicators,
            engineer_features,
            load_and_resample,
        )
    except ImportError:
        from train_xgboost_dss import (
            add_target,
            add_lag_features,
            add_technical_indicators,
            engineer_features,
            load_and_resample,
        )

    LABELS = _get_labels()
    cfg = _load_model_config()

    # Fix 3: parse config mới (train_xgboost_gold) lẫn config cũ (train_xgboost_dss)
    buy_pct           = cfg.get("buy_pct",            cfg.get("buy_pct", 0.5))
    horizon           = int(cfg.get("horizon",         cfg.get("horizon", 7)))
    optimal_threshold = cfg.get("optimal_threshold",   0.5)
    binary_mode       = True  # model mới luôn binary BUY/NOT_BUY

    df = load_and_resample(_get_master_path())
    if gold_code and gold_code.strip():
        selected = GOLD_GROUP_MAP.get(gold_code.strip())
        if selected is None:
            return None
        df = df[df["gold_code"].isin(selected)].copy()
        if df.empty:
            return None

    df = add_technical_indicators(df)
    df = add_lag_features(df)

    daily_price = df.groupby("timestamp")["sell_price"].mean().reset_index()
    daily_price.columns = ["date", "price"]

    feat_cols = ["domestic_premium", "RSI_14", "cum_return_3d_pct", "cum_return_7d_pct",
                 "cum_return_3d", "cum_return_7d", "sell_price", "world_price_vnd"]
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

    # Fix 4: gọi add_target với signature mới, target col là "target" (không phải "target_encoded")
    df_with_target = add_target(df.copy(), buy_pct=float(buy_pct), horizon=horizon)

    # Hỗ trợ cả script mới (target) và cũ (target_encoded)
    target_col = "target" if "target" in df_with_target.columns else "target_encoded"
    if target_col not in df_with_target.columns:
        return None

    df_with_target = engineer_features(df_with_target)
    if df_with_target.empty:
        return None

    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(SCALER_PATH)

    def _predict_block(df_block: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Reindex → scale → predict với optimal threshold."""
        X = df_block.drop(columns=[target_col, "timestamp"], errors="ignore").copy()
        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        if hasattr(preprocessor, "transform_df"):
            X = preprocessor.transform_df(X)

        if binary_mode and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Fix 5: binary:logistic → classes=[0,1], proba[:,1] = P(BUY=1)
            classes    = list(model.classes_)
            buy_idx    = classes.index(1) if 1 in classes else 1   # luôn là 1
            prob_buy   = proba[:, buy_idx]
            thr        = float(optimal_threshold) if optimal_threshold else 0.5
            pred_int   = np.where(prob_buy >= thr, 1, 0).astype(int)
            return (pd.Series(pred_int, index=df_block.index),
                    pd.Series(prob_buy, index=df_block.index))

        preds = pd.Series(model.predict(X), index=df_block.index)
        return preds, pd.Series(np.nan, index=df_block.index)

    def _engineer_features_for_inference_loose(df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features cho inference nhưng không dropna toàn bộ hàng.
        Mục tiêu: vẫn có dự đoán cho ngày mới nhất khi macro feature bị thiếu tạm thời.
        """
        dfx = df_raw.copy()
        grp = dfx.groupby("gold_code", dropna=False)
        dfx["daily_ret"] = grp["sell_price"].pct_change() * 100
        dfx["world_ret"] = grp["world_price_vnd"].pct_change() * 100
        dfx["premium_pct"] = dfx["sell_price"] / dfx["world_price_vnd"].replace(0, np.nan) * 100
        dfx["spread_pct"] = (dfx["sell_price"] - dfx["buy_price"]) / dfx["buy_price"].replace(0, np.nan) * 100
        drop = ["buy_price", "sell_price", "world_price_vnd", "domestic_premium", "MA5", "MA20", "MA50"]
        dfx = dfx.drop(columns=[c for c in drop if c in dfx.columns])
        dfx = pd.get_dummies(dfx, columns=["gold_code"], drop_first=False)
        dum = [c for c in dfx.columns if c.startswith("gold_code_")]
        if dum:
            dfx[dum] = dfx[dum].astype("int8")
        dfx = dfx.replace([np.inf, -np.inf], np.nan)
        dfx = dfx.sort_values("timestamp").reset_index(drop=True)
        dfx = dfx.ffill().bfill().fillna(0)
        return dfx

    df_with_target["prediction"], df_with_target["prob_buy"] = _predict_block(df_with_target)

    # Inference cho `horizon` ngày cuối (không có future target).
    # Quan trọng: phải engineer trên FULL dataset để pct_change/rolling không bị NaN hàng loạt.
    try:
        cutoff_date = df_with_target["timestamp"].max() if not df_with_target.empty else None
        infer_dates = sorted(df["timestamp"].dropna().unique())
        if cutoff_date is not None:
            infer_dates = [d for d in infer_dates if d > cutoff_date]
        else:
            infer_dates = infer_dates[-horizon:]
        infer_full = df.copy()
        infer_full["target"] = 0
        infer_full["target_encoded"] = 0
        infer_full = engineer_features(infer_full)
        if not infer_full.empty:
            infer_last = infer_full[infer_full["timestamp"].isin(infer_dates)].copy()
            if infer_last.empty:
                # Fallback: nếu engineer_features(dropna) loại hết ngày mới,
                # dùng bản "loose" để vẫn phát tín hiệu realtime.
                infer_full = _engineer_features_for_inference_loose(df.copy())
                if target_col not in infer_full.columns:
                    infer_full[target_col] = 0
                infer_last = infer_full[infer_full["timestamp"].isin(infer_dates)].copy()
            if not infer_last.empty and target_col in infer_last.columns:
                infer_last["prediction"], infer_last["prob_buy"] = _predict_block(infer_last)
                df_with_target = pd.concat([df_with_target, infer_last], ignore_index=True)
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
    def _int_to_label(val):
        if pd.isna(val):
            return None
        try:
            idx = int(val)
            return LABELS[idx] if 0 <= idx < len(LABELS) else str(val)
        except (ValueError, TypeError):
            return str(val) if str(val) in LABELS else None
    merge["prediction_label"] = merge["prediction"].apply(_int_to_label)
    merge["prob_buy_label"] = merge.get("prob_buy", None)

    # Gắn feature theo ngày (cùng thứ tự với merge) cho biểu đồ phía dưới
    feature_series = {}
    if daily_feat is not None:
        feat_merge = merge[["date"]].merge(daily_feat, on="date", how="left")
        # Hỗ trợ cả tên cột script cũ (_pct suffix) và mới (không có _pct)
        _col_map = {"cum_return_3d_pct": "cum_return_3d", "cum_return_7d_pct": "cum_return_7d"}
        for col in ["premium_pct", "domestic_premium", "RSI_14",
                    "cum_return_3d_pct", "cum_return_7d_pct",
                    "cum_return_3d", "cum_return_7d", "world_price_vnd"]:
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


def _fallback_payload_without_model(gold_code: str | None = None) -> dict | None:
    """
    Fallback khi thiếu model/scaler: vẫn trả dữ liệu giá để frontend render,
    predictions để N/A để tránh dashboard bị kẹt "Đang tải...".
    """
    df = pd.read_csv(_get_master_path())
    df = _normalize_master_columns(df)
    if "timestamp" not in df.columns or "sell_price" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if "gold_code" not in df.columns:
        df["gold_code"] = None
    if gold_code and gold_code.strip():
        selected = GOLD_GROUP_MAP.get(gold_code.strip())
        if selected is None:
            return None
        df = df[df["gold_code"].isin(selected)].copy()
        if df.empty:
            return None

    daily_price = (
        df.groupby(df["timestamp"].dt.normalize())["sell_price"]
        .mean()
        .reset_index()
    )
    daily_price.columns = ["date", "price"]
    daily_price = daily_price.sort_values("date").tail(MAX_POINTS)
    predictions = ["N/A"] * len(daily_price)

    latest_price = float(daily_price["price"].iloc[-1]) if len(daily_price) else None
    latest_date = str(daily_price["date"].iloc[-1]) if len(daily_price) else None
    return {
        "gold_code": gold_code.strip() if gold_code and gold_code.strip() else None,
        "gold_codes": _get_gold_codes(),
        "labels": _get_labels(),
        "dates": daily_price["date"].astype(str).tolist(),
        "prices": daily_price["price"].round(0).tolist(),
        "predictions": predictions,
        "probabilities": [None] * len(daily_price),
        "features": {},
        "latest": {
            "date": latest_date,
            "price": latest_price,
            "prediction": None,
            "prediction_date": None,
            "prediction_date_price": None,
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
        fallback = _fallback_payload_without_model(gold_code=gold_code)
        if fallback is not None:
            fallback["warning"] = f"Model chưa sẵn sàng ({e}). Đang hiển thị dữ liệu giá, tín hiệu tạm thời là N/A."
            return jsonify(fallback), 200
        return jsonify({"error": f"File not found: {e}"}), 404
    except Exception as e:
        fallback = _fallback_payload_without_model(gold_code=gold_code)
        if fallback is not None:
            fallback["warning"] = f"Pipeline tạm thời lỗi ({e}). Đang hiển thị dữ liệu giá, tín hiệu tạm thời là N/A."
            return jsonify(fallback), 200
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
    try:
        synced = sync_master_from_google_sheet(master_path)
        if synced is None:
            print("[Sheet sync] Sync bo qua (xem log tren de biet ly do).")
        elif synced == 0:
            print("[Sheet sync] Khong co dong moi tu Sheet so voi master.")
        elif synced > 0:
            g = globals()
            g["_master_path_cache"] = None
            g["_gold_codes_cache"] = None
    except Exception as e:
        # On some Windows terminals, non-ASCII log lines can raise UnicodeEncodeError.
        # Do not stop the server because of sync logging problems.
        print(f"[Sheet sync] Skip sync due to error: {e}")
    # Nếu synced is None: không cấu hình Sheet hoặc lỗi, bỏ qua

    if not MODEL_PATH.exists():
        print("Model not found. Train first with train_xgboost_gold.py")
    else:
        print("Open http://127.0.0.1:5000 in browser")
    app.run(host="0.0.0.0", port=5000, debug=False)