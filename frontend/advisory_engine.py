"""
advisory_engine.py  —  v1.0
============================
Module tổng hợp XGBoost signal + LLM analysis → lời khuyên đầu tư cuối cùng.

Trả lời 3 câu hỏi:
  1. Có nên mua không?  → STRONG_BUY / BUY / WATCH / AVOID
  2. Tại sao?           → list lý do cụ thể từ model + tin tức
  3. Xu hướng 7 ngày?   → tăng / giảm / sideway

Cách dùng:
  from advisory_engine import generate_advisory
  result = generate_advisory(gold_code="SJC_BAR", ...)
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

# ── In-memory cache ─────────────────────────────────────────────────────────────
_advisory_cache: dict = {}   # key = gold_code, value = (timestamp_float, result_dict)
CACHE_TTL_SECONDS = 1800     # 30 phút


def _cache_get(gold_code: str) -> dict | None:
    """Trả về cached result nếu còn hiệu lực, None nếu hết hạn hoặc chưa có."""
    key = gold_code or "__ALL__"
    entry = _advisory_cache.get(key)
    if entry is None:
        return None
    cached_ts, cached_result = entry
    if time.time() - cached_ts > CACHE_TTL_SECONDS:
        del _advisory_cache[key]
        return None
    return cached_result


def _cache_set(gold_code: str, result: dict) -> None:
    key = gold_code or "__ALL__"
    _advisory_cache[key] = (time.time(), result)


# ── Feature interpretation (Vietnamese) ─────────────────────────────────────────
_FEATURE_INTERPRETERS: dict = {
    "cum_return_7d": lambda v: (
        f"tăng {v:.2f}% trong 7 ngày" if v > 0 else f"giảm {abs(v):.2f}% trong 7 ngày"
    ),
    "cum_return_3d": lambda v: (
        f"tăng {v:.2f}% trong 3 ngày" if v > 0 else f"giảm {abs(v):.2f}% trong 3 ngày"
    ),
    "cum_return_14d": lambda v: (
        f"tăng {v:.2f}% trong 14 ngày" if v > 0 else f"giảm {abs(v):.2f}% trong 14 ngày"
    ),
    "RSI_14": lambda v: (
        "oversold nặng, có thể sắp bật tăng" if v < 30
        else "oversold, vùng tích lũy" if v < 40
        else "overbought, cẩn thận" if v > 70
        else f"trung tính ({v:.1f})"
    ),
    "premium_trend_7d": lambda v: (
        f"premium thu hẹp {abs(v/1e6):.1f}tr/tuần — NHNN can thiệp" if v < -1_000_000
        else f"premium mở rộng {v/1e6:.1f}tr/tuần" if v > 1_000_000
        else "premium ổn định"
    ),
    "premium_pct_raw": lambda v: f"premium nội địa {v:.1f}%",
    "premium_vs_ma7": lambda v: (
        f"premium cao hơn MA7 {v:.1f}%" if v > 0 else f"premium thấp hơn MA7 {abs(v):.1f}%"
    ),
    "daily_ret": lambda v: (
        f"tăng {v:.2f}% hôm nay" if v > 0 else f"giảm {abs(v):.2f}% hôm nay"
    ),
    "world_ret": lambda v: (
        f"vàng TG tăng {v:.2f}%" if v > 0 else f"vàng TG giảm {abs(v):.2f}%"
    ),
    "world_chg_5d": lambda v: (
        f"vàng TG tăng {v:.2f}% trong 5 ngày" if v > 0
        else f"vàng TG giảm {abs(v):.2f}% trong 5 ngày"
    ),
    "world_chg_14d": lambda v: (
        f"vàng TG tăng {v:.2f}% trong 14 ngày" if v > 0
        else f"vàng TG giảm {abs(v):.2f}% trong 14 ngày"
    ),
    "usdvnd_chg_7d": lambda v: (
        f"USD/VND tăng {v:.2f}% tuần qua" if v > 0
        else f"USD/VND giảm {abs(v):.2f}% tuần qua"
    ),
    "dxy_chg_5d": lambda v: (
        f"DXY tăng {v:.2f}% — USD mạnh lên" if v > 0
        else f"DXY giảm {abs(v):.2f}% — USD yếu đi"
    ),
    "vol_5d": lambda v: (
        "biến động cao (rủi ro lớn)" if v > 3 else "biến động bình thường"
    ),
    "vol_14d": lambda v: (
        "biến động 14 ngày cao" if v > 2.5 else "biến động 14 ngày bình thường"
    ),
    "spread_pct": lambda v: f"spread mua-bán {v:.2f}%",
    "price_to_MA5_pct": lambda v: (
        f"giá trên MA5 {v:.2f}%" if v > 0 else f"giá dưới MA5 {abs(v):.2f}%"
    ),
    "price_to_MA20_pct": lambda v: (
        f"giá trên MA20 {v:.2f}%" if v > 0 else f"giá dưới MA20 {abs(v):.2f}%"
    ),
    "price_to_MA50_pct": lambda v: (
        f"giá trên MA50 {v:.2f}%" if v > 0 else f"giá dưới MA50 {abs(v):.2f}%"
    ),
    "rsi_chg_3d": lambda v: (
        f"RSI tăng {v:.1f} điểm (3 ngày) — momentum tăng" if v > 0
        else f"RSI giảm {abs(v):.1f} điểm (3 ngày) — momentum giảm"
    ),
    "rsi_chg_7d": lambda v: (
        f"RSI tăng {v:.1f} điểm (7 ngày)" if v > 0
        else f"RSI giảm {abs(v):.1f} điểm (7 ngày)"
    ),
    "fed_rate": lambda v: f"lãi suất FED {v:.2f}%",
    "interest_rate_state": lambda v: f"lãi suất NN Việt Nam {v:.2f}%",
}


def _interpret_feature(name: str, value: float) -> str:
    """Chuyển feature name + giá trị → câu giải thích tiếng Việt."""
    interpreter = _FEATURE_INTERPRETERS.get(name)
    if interpreter:
        try:
            return interpreter(value)
        except Exception:
            pass
    return f"{name} = {value:.3f}"


# ── Decision matrix ─────────────────────────────────────────────────────────────
_RECOMMENDATION_LABELS = {
    "STRONG_BUY": "MUA MẠNH",
    "BUY":        "NÊN MUA",
    "WATCH":      "THEO DÕI",
    "AVOID":      "CHƯA NÊN MUA",
}

_RECOMMENDATION_THRESHOLD = 0.17  # fallback nếu config không có


def _determine_recommendation(
    p_buy: float,
    llm_signal: str | None,
    llm_confidence: float | None,
    threshold: float = _RECOMMENDATION_THRESHOLD,
) -> str:
    """
    Decision matrix:
      STRONG_BUY : P_buy >= 0.5  AND llm=BUY  AND confidence >= 0.7
      BUY        : P_buy >= thr  AND (llm=BUY  OR  confidence >= 0.6)
      WATCH      : (P_buy >= thr AND llm=NOT_BUY)
                OR (P_buy <  thr AND llm=BUY)
                OR (confidence < 0.5)
      AVOID      : P_buy <  thr  AND llm=NOT_BUY
    """
    sig = (llm_signal or "").upper()
    conf = llm_confidence if llm_confidence is not None else 0.5

    # STRONG_BUY
    if p_buy >= 0.5 and sig == "BUY" and conf >= 0.7:
        return "STRONG_BUY"

    # BUY
    if p_buy >= threshold and (sig == "BUY" or conf >= 0.6):
        return "BUY"

    # AVOID
    if p_buy < threshold and sig == "NOT_BUY":
        return "AVOID"

    # WATCH — tín hiệu mâu thuẫn hoặc confidence thấp
    # (P_buy >= thr AND llm=NOT_BUY) OR (P_buy < thr AND llm=BUY) OR (conf < 0.5)
    return "WATCH"


def _determine_recommendation_xgb_only(p_buy: float, threshold: float) -> str:
    """Recommendation khi không có LLM (fallback)."""
    if p_buy >= 0.5:
        return "BUY"
    if p_buy >= threshold:
        return "WATCH"
    return "AVOID"


def _determine_price_outlook(
    xgb_signal: str,
    llm_outlook: str | None,
    final_rec: str,
) -> str:
    """
    Xu hướng giá 7 ngày (định tính):
      XGBoost=BUY  + LLM=tăng  → "tăng"
      XGBoost=AVOID + LLM=giảm → "giảm"
      Còn lại                   → "sideway"
    """
    llm_dir = (llm_outlook or "").lower().strip()
    if xgb_signal == "BUY" and llm_dir == "tăng":
        return "tăng"
    if final_rec == "AVOID" and llm_dir == "giảm":
        return "giảm"
    return "sideway"


# ── Build reasons list ──────────────────────────────────────────────────────────
def _build_reasons(
    xgb_top_features: list[dict],
    llm_result: dict | None,
    llm_available: bool,
) -> list[str]:
    """Tổng hợp 4-5 lý do từ XGBoost features + LLM analysis."""
    reasons = []

    # Lý do từ XGBoost (top 2-3 features)
    for feat in xgb_top_features[:3]:
        interp = feat.get("interpretation", "")
        if interp:
            reasons.append(interp.capitalize() if interp[0].islower() else interp)

    # Lý do từ LLM
    if llm_available and llm_result:
        # Từ key_factors
        factors = llm_result.get("key_factors") or []
        for f in factors[:2]:
            factor_text = f.get("factor", "")
            impact = f.get("impact", "")
            if factor_text:
                direction = "↑" if impact == "positive" else "↓" if impact == "negative" else "→"
                reasons.append(f"{factor_text} {direction}")

        # Từ news_summary
        news_summary = llm_result.get("news_summary", "")
        if news_summary and len(reasons) < 5:
            reasons.append(f"Tin tức: {news_summary}")
    elif not llm_available:
        reasons.append("Phân tích tin tức tạm thời không khả dụng")

    return reasons[:5]


# ── Main function ───────────────────────────────────────────────────────────────
def generate_advisory(
    gold_code: str | None = None,
    root_path: "Path | str | None" = None,
    predict_result: dict | None = None,
) -> dict:
    """
    Tổng hợp XGBoost + LLM → lời khuyên đầu tư.

    Parameters:
        gold_code:      Nhóm vàng (SJC_BAR, JEWELRY_9999, ...) hoặc None = trung bình
        root_path:      Đường dẫn ROOT project
        predict_result: Kết quả từ _run_pipeline_and_predict() — nếu đã chạy sẵn

    Returns:
        dict — advisory response đầy đủ 3 layer
    """
    import numpy as np

    gold_code_str = (gold_code or "").strip() or None

    # ── Check cache ──────────────────────────────────────────────────────
    cached = _cache_get(gold_code_str)
    if cached is not None:
        return cached

    # ── ROOT path ────────────────────────────────────────────────────────
    if root_path is None:
        root_path = Path(__file__).resolve().parent.parent
    else:
        root_path = Path(root_path)

    # ── Layer 1: XGBoost — lấy từ predict_result ────────────────────────
    if predict_result is None:
        # Import lazy để tránh circular
        import sys
        training_dir = root_path / "training"
        if str(training_dir) not in sys.path:
            sys.path.insert(0, str(training_dir))

        # Đọc model config
        config_path = training_dir / "output" / "model_config.json"
        try:
            import json
            with open(config_path, encoding="utf-8") as f:
                model_cfg = json.load(f)
        except Exception:
            model_cfg = {}

        threshold = model_cfg.get("optimal_threshold", _RECOMMENDATION_THRESHOLD)
        # Không có predict_result → trả về advisory tối thiểu
        result = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "gold_code": gold_code_str,
            "current_price": None,
            "xgb_signal": "N/A",
            "xgb_prob_buy": None,
            "xgb_threshold": threshold,
            "xgb_top_features": [],
            "llm_signal": None,
            "llm_confidence": None,
            "llm_available": False,
            "news_count": 0,
            "key_news_factors": [],
            "price_outlook_7d": "sideway",
            "final_recommendation": "WATCH",
            "recommendation_label": _RECOMMENDATION_LABELS["WATCH"],
            "reasons": ["Không có đủ dữ liệu để phân tích"],
            "suggested_action": "Chờ hệ thống cập nhật dữ liệu",
            "risk_level": "medium",
            "disclaimer": "Đây là hỗ trợ quyết định, không phải lời khuyên đầu tư",
        }
        _cache_set(gold_code_str, result)
        return result

    # ── Parse XGBoost results ────────────────────────────────────────────
    latest = predict_result.get("latest", {})
    current_price = latest.get("price")
    latest_date = latest.get("date", datetime.now().strftime("%Y-%m-%d"))
    xgb_prediction = latest.get("prediction")  # "BUY" or "NOT_BUY" or None

    # Lấy xác suất BUY từ probabilities (ngày cuối cùng có prediction)
    probabilities = predict_result.get("probabilities", [])
    predictions = predict_result.get("predictions", [])
    dates = predict_result.get("dates", [])

    # Tìm prob_buy cho prediction mới nhất
    prob_buy = None
    for i in range(len(predictions) - 1, -1, -1):
        if predictions[i] and predictions[i] != "N/A":
            prob_buy = probabilities[i] if i < len(probabilities) else None
            break

    if prob_buy is None:
        prob_buy = 0.0

    # Đọc model config
    config_path = root_path / "training" / "output" / "model_config.json"
    try:
        import json
        with open(config_path, encoding="utf-8") as f:
            model_cfg = json.load(f)
    except Exception:
        model_cfg = {}

    threshold = model_cfg.get("optimal_threshold", _RECOMMENDATION_THRESHOLD)
    xgb_signal = "BUY" if prob_buy >= threshold else "NOT_BUY"

    # ── Top features từ XGBoost ──────────────────────────────────────────
    xgb_top_features = _extract_top_features(predict_result, root_path)

    # ── Layer 2: LLM + News ──────────────────────────────────────────────
    llm_result = None
    llm_available = False
    news_count = 0

    try:
        try:
            from frontend.llm_adjust import run_llm_adjust_for_advisory
        except ImportError:
            from llm_adjust import run_llm_adjust_for_advisory

        pred_date_str = latest.get("prediction_date") or latest_date
        llm_result = run_llm_adjust_for_advisory(
            root_path=root_path,
            pred_date_str=str(pred_date_str)[:10] if pred_date_str else None,
            ml_signal=xgb_signal,
            price_vnd=current_price,
            prob_buy=prob_buy,
        )
        if llm_result is not None:
            llm_available = True
            news_count = llm_result.get("news_count", 0)
    except ImportError:
        # run_llm_adjust_for_advisory chưa tồn tại → fallback
        try:
            try:
                from frontend.llm_adjust import run_llm_adjust_for_latest
            except ImportError:
                from llm_adjust import run_llm_adjust_for_latest

            llm_raw = run_llm_adjust_for_latest(
                root_path=root_path,
                pred_date_str=str(latest.get("prediction_date", latest_date))[:10],
                ml_signal=xgb_signal,
                price_vnd=current_price,
            )
            if llm_raw:
                llm_result = {
                    "signal": llm_raw.get("adjusted_signal", "NOT_BUY"),
                    "confidence": llm_raw.get("confidence", 0.5),
                    "reasoning": llm_raw.get("reasoning", ""),
                    "key_factors": [],
                    "risk_level": "medium",
                    "suggested_action": "",
                    "price_outlook_7d": "sideway",
                    "key_risk": llm_raw.get("key_risk", ""),
                    "news_summary": llm_raw.get("updated_price_note", ""),
                    "news_count": 0,
                }
                llm_available = True
        except Exception:
            pass
    except Exception:
        pass

    # ── Layer 3: Tổng hợp → lời khuyên cuối ─────────────────────────────
    llm_signal = None
    llm_confidence = None

    if llm_available and llm_result:
        llm_signal = (llm_result.get("signal") or "").upper()
        if llm_signal not in ("BUY", "NOT_BUY", "WATCH"):
            llm_signal = None
        llm_confidence = llm_result.get("confidence")

    # Recommendation
    if llm_available and llm_signal:
        final_rec = _determine_recommendation(
            p_buy=prob_buy,
            llm_signal=llm_signal,
            llm_confidence=llm_confidence,
            threshold=threshold,
        )
    else:
        final_rec = _determine_recommendation_xgb_only(prob_buy, threshold)

    # Price outlook
    llm_outlook = llm_result.get("price_outlook_7d") if llm_result else None
    price_outlook = _determine_price_outlook(xgb_signal, llm_outlook, final_rec)

    # Risk level
    risk_level = "medium"
    if llm_result:
        risk_level = llm_result.get("risk_level", "medium")
    if not llm_available:
        risk_level = "medium"

    # Suggested action
    suggested_action = ""
    if llm_result and llm_result.get("suggested_action"):
        suggested_action = llm_result["suggested_action"]
    else:
        if final_rec == "STRONG_BUY":
            suggested_action = "Tín hiệu tốt, cân nhắc mua nếu phù hợp kế hoạch đầu tư"
        elif final_rec == "BUY":
            suggested_action = "Có thể mua với khối lượng vừa phải, đặt stop-loss"
        elif final_rec == "WATCH":
            suggested_action = "Theo dõi thêm 2-3 ngày, chờ tín hiệu rõ ràng hơn"
        else:
            suggested_action = "Không nên mua, chờ giá ổn định và tín hiệu cải thiện"

    # Reasons
    reasons = _build_reasons(xgb_top_features, llm_result, llm_available)

    # ── Build response ───────────────────────────────────────────────────
    result = {
        "date": str(latest_date)[:10],
        "gold_code": gold_code_str,
        "current_price": current_price,

        # Layer 1: XGBoost
        "xgb_signal": xgb_signal,
        "xgb_prob_buy": round(prob_buy, 4) if prob_buy is not None else None,
        "xgb_threshold": threshold,
        "xgb_top_features": xgb_top_features[:5],

        # Layer 2: LLM + News
        "llm_signal": llm_signal,
        "llm_confidence": round(llm_confidence, 2) if llm_confidence is not None else None,
        "llm_available": llm_available,
        "news_count": news_count,
        "key_news_factors": (llm_result.get("key_factors") or []) if llm_result else [],
        "price_outlook_7d": price_outlook,

        # Layer 3: Tổng hợp
        "final_recommendation": final_rec,
        "recommendation_label": _RECOMMENDATION_LABELS.get(final_rec, final_rec),
        "reasons": reasons,
        "suggested_action": suggested_action,
        "risk_level": risk_level,
        "disclaimer": "Đây là hỗ trợ quyết định, không phải lời khuyên đầu tư",
    }

    # LLM extras
    if llm_result:
        result["llm_reasoning"] = llm_result.get("reasoning", "")
        result["llm_key_risk"] = llm_result.get("key_risk", "")
        result["llm_news_summary"] = llm_result.get("news_summary", "")

    _cache_set(gold_code_str, result)
    return result


def _extract_top_features(predict_result: dict, root_path: Path) -> list[dict]:
    """Trích xuất top features quan trọng nhất với interpretation tiếng Việt."""
    import joblib
    import pandas as pd

    model_path = root_path / "training" / "output" / "xgboost_gold_model.pkl"
    top_features = []

    try:
        model = joblib.load(model_path)
        if not hasattr(model, "feature_importances_") or not hasattr(model, "feature_names_in_"):
            return top_features

        fi = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        # Lọc bỏ gold_code_ dummies
        fi = fi[[c for c in fi.index if not c.startswith("gold_code_")]]
        top_names = fi.nlargest(5).index.tolist()

        # Lấy giá trị feature từ features trong predict_result
        features_data = predict_result.get("features", {})

        for name in top_names:
            # Tìm giá trị mới nhất của feature
            value = None
            if name in features_data:
                vals = features_data[name]
                # Lấy giá trị cuối cùng (mới nhất)
                for v in reversed(vals):
                    if v is not None:
                        value = v
                        break

            interpretation = ""
            if value is not None:
                interpretation = _interpret_feature(name, value)

            top_features.append({
                "name": name,
                "value": round(value, 4) if value is not None else None,
                "interpretation": interpretation,
            })
    except Exception:
        pass

    return top_features
