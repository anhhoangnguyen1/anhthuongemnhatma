# -*- coding: utf-8 -*-
"""
Mô hình B: Sau khi ML (A) dự đoán xong, lấy tin tức realtime (GNews API)
24–48h gần đây, đưa vào LLM đánh giá và điều chỉnh — không dùng CSV.

Luồng: Prediction(ML) → GNews realtime (48h qua) → LLM → điều chỉnh.
Cần: OPENAI_API_KEY, GNEWS_API_KEY (gnews.io).
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"
GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"


def fetch_news_realtime_gnews(
    api_key: str,
    query: str = "giá vàng SJC thị trường vàng",
    hours_back: int = 48,
    lang: str = "vi",
    max_articles: int = 15,
) -> list[dict[str, str]]:
    """
    GNews API (gnews.io): tin realtime, lang=vi.
    Tham số đúng theo docs: apikey (không phải token).
    Lọc theo hours_back nếu có publishedAt; không có ngày thì vẫn giữ bài.
    """
    try:
        key = (api_key or "").strip()
        if not key:
            print("[GNEWS] Missing api_key")
            return []
        params = {
            "apikey": key,
            "q": query,
            "lang": lang,
            "max": max_articles,
        }
        print(f"[GNEWS] Request: {GNEWS_SEARCH_URL} params={params}")
        resp = requests.get(GNEWS_SEARCH_URL, params=params, timeout=15)
        status = resp.status_code
        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {}
        total = data.get("totalArticles", "?")
        print(f"[GNEWS] Status={status}, keys={list(data.keys())}, totalArticles={total}")
        if not resp.ok:
            print(f"[GNEWS] Error body={data}")
            return []
        articles = data.get("articles") or []
        print(f"[GNEWS] articles_count={len(articles)}")
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        out = []
        for a in articles:
            pub = a.get("publishedAt") or a.get("published")
            if pub:
                try:
                    ts = pd.to_datetime(pub, utc=True)
                    if ts.to_pydatetime() < cutoff:
                        continue
                except Exception:
                    pass
            title = (a.get("title") or "").strip()
            if not title:
                continue
            snippet = (a.get("description") or a.get("content") or "")[:400]
            out.append({"title": title, "snippet": snippet})
        if not out and articles:
            print("[GNEWS] No articles after cutoff filter, falling back to all articles.")
            for a in articles:
                title = (a.get("title") or "").strip()
                if not title:
                    continue
                snippet = (a.get("description") or a.get("content") or "")[:400]
                out.append({"title": title, "snippet": snippet})
        if not out:
            fallback_query = "gold price Vietnam"
            fallback_lang = "en"
            print(f"[GNEWS] 0 articles for q={query!r} lang={lang}, retry q={fallback_query!r} lang={fallback_lang}")
            params2 = {"apikey": key, "q": fallback_query, "lang": fallback_lang, "max": max_articles}
            try:
                resp2 = requests.get(GNEWS_SEARCH_URL, params=params2, timeout=15)
                data2 = resp2.json() if resp2.content else {}
                articles2 = data2.get("articles") or []
                print(f"[GNEWS] fallback totalArticles={data2.get('totalArticles')}, articles_count={len(articles2)}")
                cutoff = datetime.utcnow() - timedelta(hours=hours_back)
                for a in articles2:
                    pub = a.get("publishedAt") or a.get("published")
                    if pub:
                        try:
                            ts = pd.to_datetime(pub, utc=True)
                            if ts.to_pydatetime() < cutoff:
                                continue
                        except Exception:
                            pass
                    title = (a.get("title") or "").strip()
                    if not title:
                        continue
                    snippet = (a.get("description") or a.get("content") or "")[:400]
                    out.append({"title": title, "snippet": snippet})
                if not out and articles2:
                    for a in articles2:
                        title = (a.get("title") or "").strip()
                        if not title:
                            continue
                        snippet = (a.get("description") or a.get("content") or "")[:400]
                        out.append({"title": title, "snippet": snippet})
            except Exception as e2:
                print(f"[GNEWS] fallback Exception: {e2}")
        print(f"[GNEWS] returned_articles={len(out)}")
        return out
    except Exception as e:
        print(f"[GNEWS] Exception: {e}")
        return []


def fetch_news_gnews_for_window(
    api_key: str,
    query: str,
    start_date: date,
    end_date: date,
    lang: str = "en",
    country: str | None = None,
    max_articles: int = 20,
) -> list[dict[str, str]]:
    """
    Lấy tin GNews cho 1 cửa sổ thời gian cố định [start_date, end_date] (không phụ thuộc thời điểm hiện tại).
    Dùng cho luồng click vào ngày: ví dụ lấy tin 24–48h TRƯỚC ngày đó.
    """
    try:
        key = (api_key or "").strip()
        if not key:
            print("[GNEWS/window] Missing api_key")
            return []
        params: dict[str, Any] = {
            "apikey": key,
            "q": query,
            "lang": lang,
            "max": max_articles,
            "in": "title,description,content",
            "nullable": "description,content,image",
            "sortby": "publishedAt",
            "from": f"{start_date.isoformat()}T00:00:00.000Z",
            "to": f"{end_date.isoformat()}T23:59:59.999Z",
        }
        if country:
            params["country"] = country
        print(f"[GNEWS/window] Request: {GNEWS_SEARCH_URL} params={params}")
        resp = requests.get(GNEWS_SEARCH_URL, params=params, timeout=15)
        status = resp.status_code
        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {}
        total = data.get("totalArticles", "?")
        print(f"[GNEWS/window] Status={status}, totalArticles={total}")
        if not resp.ok:
            print(f"[GNEWS/window] Error body={data}")
            return []
        articles = data.get("articles") or []
        print(f"[GNEWS/window] articles_count={len(articles)}")
        out: list[dict[str, str]] = []
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title:
                continue
            snippet = (a.get("description") or a.get("content") or "")[:400]
            out.append({"title": title, "snippet": snippet})
        print(f"[GNEWS/window] returned_articles={len(out)}")
        return out
    except Exception as e:
        print(f"[GNEWS/window] Exception: {e}")
        return []


def _call_openai(api_key: str, prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(OPENAI_CHAT_URL, json=payload, headers=headers, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def _parse_llm_response(text: str) -> dict[str, Any]:
    out = {
        "adjusted_signal": None,
        "reasoning": "",
        "confidence": 0.5,
        "updated_price_note": "",
    }
    raw = re.sub(r"^```\w*\n?", "", text.strip()).strip()
    raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            out["adjusted_signal"] = obj.get("adjusted_signal") or obj.get("updated_signal")
            out["reasoning"] = str(obj.get("reasoning") or obj.get("reason") or "")
            c = obj.get("confidence")
            if c is not None:
                out["confidence"] = max(0.0, min(1.0, float(c)))
            out["updated_price_note"] = str(obj.get("updated_price_note") or obj.get("price_note") or "")
    except (json.JSONDecodeError, TypeError):
        out["reasoning"] = raw[:500]
    return out


PROMPT_TEMPLATE = """Bạn là chuyên gia tài chính. Cho ngày {date}, mô hình ML dự đoán tín hiệu vàng: **{prediction}**, giá tham chiếu: {price} VND.

Tin tức 24–48h gần đây (realtime):
{headlines}

Nhiệm vụ: Đánh giá độ tin cậy của dự đoán dựa trên tin này; nếu tin cho thấy cần điều chỉnh thì đưa ra tín hiệu điều chỉnh và ghi chú về giá.

Trả lời ĐÚNG 1 khối JSON sau, không thêm text khác:
{{
  "adjusted_signal": "BUY hoặc NOT_BUY (hoặc giữ nguyên nếu đáng tin)",
  "reasoning": "Lý do ngắn gọn",
  "confidence": 0.0-1.0 (độ tin cậy sau khi xem tin),
  "updated_price_note": "Ghi chú về xu hướng giá nếu có (ví dụ: có thể giảm nhẹ do tin Fed)"
}}
"""

# Prompt cho luồng "chọn ngày": chỉ dựa trên tin tức, đưa ra dự đoán bổ sung (không đánh giá ML).
PROMPT_SUPPLEMENT = """Bạn là chuyên gia tài chính. Cho ngày {date}, giá vàng tham chiếu: {price} VND.

Tin tức 24–48h gần đây (realtime):
{headlines}

Nhiệm vụ: Chỉ dựa trên tin tức trên, đưa ra **dự đoán bổ sung** (tín hiệu giao dịch) cho ngày này. Không cần so với mô hình ML.

Trả lời ĐÚNG 1 khối JSON sau, không thêm text khác:
{{
  "adjusted_signal": "BUY hoặc NOT_BUY",
  "reasoning": "Lý do ngắn gọn dựa trên tin",
  "confidence": 0.0-1.0 (độ tin cậy),
  "updated_price_note": "Ghi chú về xu hướng giá nếu có"
}}
"""


def llm_adjust_prediction(
    prediction_date: date,
    ml_prediction: str,
    price_vnd: float,
    news_list: list[dict[str, str]],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Đưa dự đoán ML + tin realtime vào LLM, trả về đánh giá và điều chỉnh."""
    if not news_list:
        return {
            "adjusted_signal": ml_prediction,
            "reasoning": "Không lấy được tin realtime; giữ nguyên dự đoán ML.",
            "confidence": 0.5,
            "updated_price_note": "",
        }

    headlines = "\n".join(
        f"- {n.get('title', '')}" + (f" | {n.get('snippet', '')[:150]}" if n.get("snippet") else "")
        for n in news_list[:15]
    )
    prompt = PROMPT_TEMPLATE.format(
        date=prediction_date.isoformat(),
        prediction=ml_prediction or "N/A",
        price=f"{price_vnd:,.0f}" if price_vnd else "N/A",
        headlines=headlines or "(Không có tiêu đề)",
    )
    try:
        text = _call_openai(api_key, prompt, model=model)
        result = _parse_llm_response(text)
        if result["adjusted_signal"] is None:
            result["adjusted_signal"] = ml_prediction
        return result
    except Exception as e:
        return {
            "adjusted_signal": ml_prediction,
            "reasoning": f"LLM lỗi: {e}",
            "confidence": 0.5,
            "updated_price_note": "",
        }


def run_llm_adjust_for_latest(
    root: Path,
    latest_date: str | None,
    latest_prediction: str | None,
    latest_price: float | None,
) -> dict[str, Any] | None:
    """
    Chỉ dùng tin realtime (GNews). Lấy tin 48h gần đây → gọi LLM → trả về llm_adjusted.
    Cần OPENAI_API_KEY và GNEWS_API_KEY.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    gnews_key = os.getenv("GNEWS_API_KEY")
    if not api_key or not gnews_key or not latest_date or not latest_prediction:
        return None
    try:
        d = pd.to_datetime(latest_date, errors="coerce").date()
    except Exception:
        return None
    if pd.isna(d):
        return None

    news_list = fetch_news_realtime_gnews(
        gnews_key,
        query="gold price",
        hours_back=48,
        lang="en",
    )
    price = float(latest_price) if latest_price is not None else 0.0
    result = llm_adjust_prediction(
        prediction_date=d,
        ml_prediction=latest_prediction,
        price_vnd=price,
        news_list=news_list,
        api_key=api_key,
    )
    return result


def get_news_and_llm_supplement_for_date(
    prediction_date: date,
    price_vnd: float,
    api_key: str,
    gnews_key: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Lấy tin realtime (GNews) + gọi LLM đưa ra **dự đoán bổ sung** chỉ dựa trên tin (cho 1 ngày).
    Trả về: {"news": [{title, snippet}, ...], "llm_supplement": {adjusted_signal, reasoning, confidence, updated_price_note}}.
    """
    # Lấy tin 24–48h TRƯỚC ngày được chọn: [prediction_date - 2, prediction_date - 1]
    start = prediction_date - timedelta(days=2)
    end = prediction_date - timedelta(days=1)
    news_list = fetch_news_gnews_for_window(
        gnews_key,
        query="gold price",
        start_date=start,
        end_date=end,
        lang="en",
        country=None,  # nếu muốn lọc theo quốc gia, đặt mã 2 chữ cái, ví dụ \"us\", \"gb\"...
    )
    out = {"news": news_list, "llm_supplement": None}
    if not news_list:
        out["llm_supplement"] = {
            "adjusted_signal": "N/A",
            "reasoning": "Không lấy được tin realtime.",
            "confidence": 0.5,
            "updated_price_note": "",
        }
        return out
    headlines = "\n".join(
        f"- {n.get('title', '')}" + (f" | {n.get('snippet', '')[:150]}" if n.get("snippet") else "")
        for n in news_list[:15]
    )
    prompt = PROMPT_SUPPLEMENT.format(
        date=prediction_date.isoformat(),
        price=f"{price_vnd:,.0f}" if price_vnd else "N/A",
        headlines=headlines or "(Không có tiêu đề)",
    )
    try:
        text = _call_openai(api_key, prompt, model=model)
        result = _parse_llm_response(text)
        if result["adjusted_signal"] is None:
            result["adjusted_signal"] = "N/A"
        out["llm_supplement"] = result
    except Exception as e:
        out["llm_supplement"] = {
            "adjusted_signal": "N/A",
            "reasoning": f"LLM lỗi: {e}",
            "confidence": 0.5,
            "updated_price_note": "",
        }
    return out
