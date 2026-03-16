# -*- coding: utf-8 -*-
"""
Đánh giá MỨC ĐỘ ẢNH HƯỞNG của tin tức lên giá vàng bằng LLM (OpenAI).

Ý tưởng: không nhồi số lượng tin hay sentiment thô vào mô hình ML, mà dùng LLM
để đánh giá từng tin → impact (1-5) và relevance (high/medium/low) → aggregate
theo ngày thành 1 chỉ số "news_impact" mạnh. Chỉ số này có thể:
  - Dùng làm 1 feature duy nhất đưa vào model, hoặc
  - Dùng làm overlay: hiển thị cảnh báo / điều chỉnh tín hiệu khi impact cao.

Cần: OPENAI_API_KEY (hoặc --api-key). Có thể dùng model rẻ (gpt-4o-mini).

Input: CSV tin tức có cột date, title, (description/snippet).
  - Có thể tạo từ Marketaux: chạy fetch_news_sentiment_marketaux.py trước, sau đó
    cần export raw articles (hiện script chỉ lưu aggregate). Hoặc dùng --from-marketaux
    để script tự gọi Marketaux rồi đưa từng tin vào LLM.

Output:
  - NEWS_IMPACT_DAILY.csv: date, news_impact (0–5), article_count — dùng làm 1 feature
    hoặc overlay (hiển thị "Mức ảnh hưởng tin tức hôm nay: cao" mà không nhồi vào ML).
  - NEWS_IMPACT_ARTICLES.csv: từng tin + impact_on_gold + relevance (chi tiết).

Cách dùng kết quả:
  - Overlay: Frontend đọc NEWS_IMPACT_DAILY, hiển thị cảnh báo khi news_impact >= 4.
  - 1 feature: Đưa news_impact vào prepare_gold_dss_pipeline (merge NEWS_IMPACT_DAILY.csv)
    rồi train lại; chỉ 1 cột impact mạnh thay vì nhiều cột tin thô.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DAILY_OUTPUT = "NEWS_IMPACT_DAILY.csv"
ARTICLES_OUTPUT = "NEWS_IMPACT_ARTICLES.csv"

PROMPT_TEMPLATE = """You are a financial analyst. For the following news headline (and optional snippet), rate its IMPACT on GOLD price in the short term (1-2 weeks). Consider: Fed/rates, USD/dollar, oil/commodities, geopolitics, inflation.

Headline: {title}
Snippet: {snippet}

Reply with ONLY a valid JSON object, no other text:
{{"impact_on_gold": <1-5>, "relevance": "<high|medium|low>"}}
- impact_on_gold: 1=negligible, 2=low, 3=moderate, 4=high, 5=very high (can move gold significantly)
- relevance: how directly the news relates to gold/dollar/rates/commodities"""


def _get_openai_key(cli_key: str | None) -> str:
    key = cli_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "Missing OpenAI API key. Set OPENAI_API_KEY or use --api-key."
        )
    return key


def _call_openai(api_key: str, prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(OPENAI_CHAT_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return text.strip()


def _parse_impact_response(text: str) -> dict[str, Any]:
    """Parse LLM response to impact (1-5) and relevance (high/medium/low)."""
    out = {"impact_on_gold": 3, "relevance": "medium"}
    # Tìm JSON trong response (có thể có markdown code block)
    raw = re.sub(r"^```\w*\n?", "", text).strip()
    raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj.get("impact_on_gold"), (int, float)):
            out["impact_on_gold"] = max(1, min(5, int(obj["impact_on_gold"])))
        r = str(obj.get("relevance", "")).lower()
        if r in ("high", "medium", "low"):
            out["relevance"] = r
    except (json.JSONDecodeError, TypeError):
        pass
    return out


def assess_article_impact(
    api_key: str,
    title: str,
    snippet: str = "",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Gọi OpenAI đánh giá 1 tin → impact_on_gold (1-5), relevance."""
    snippet = (snippet or "")[:500]
    prompt = PROMPT_TEMPLATE.format(title=title or "(no title)", snippet=snippet or "(none)")
    try:
        text = _call_openai(api_key, prompt, model=model)
        return _parse_impact_response(text)
    except Exception as e:
        return {"impact_on_gold": 3, "relevance": "medium", "_error": str(e)}


def load_articles_from_csv(path: Path) -> pd.DataFrame:
    """CSV cần có: date (hoặc timestamp), title. Tùy chọn: description, snippet."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    date_col = "date" if "date" in df.columns else "timestamp"
    if date_col not in df.columns:
        raise ValueError(f"CSV must have 'date' or 'timestamp'. Columns: {list(df.columns)}")
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=["_date"])
    title_col = next((c for c in ("title", "headline", "name") if c in df.columns), None)
    if not title_col:
        raise ValueError(f"CSV must have 'title' or 'headline'. Columns: {list(df.columns)}")
    df["_title"] = df[title_col].astype(str).str.strip()
    df["_snippet"] = df.get("description", df.get("snippet", pd.Series([""] * len(df)))).astype(str).str.strip()
    return df[["_date", "_title", "_snippet"]].drop_duplicates(subset=["_date", "_title"]).reset_index(drop=True)


def run_assessment(
    api_key: str,
    articles_df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    delay_seconds: float = 0.5,
) -> pd.DataFrame:
    """Đánh giá từng dòng tin qua LLM, trả về DataFrame có _date, _title, impact_on_gold, relevance."""
    rows = []
    for i, row in articles_df.iterrows():
        date_val = row["_date"]
        title_val = row["_title"]
        snippet_val = row["_snippet"]
        if not title_val or title_val == "nan":
            continue
        result = assess_article_impact(api_key, title_val, snippet_val, model=model)
        rows.append({
            "date": date_val,
            "title": title_val[:200],
            "impact_on_gold": result["impact_on_gold"],
            "relevance": result["relevance"],
        })
        time.sleep(delay_seconds)
    return pd.DataFrame(rows)


def aggregate_daily_impact(articles_with_impact: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp theo ngày: news_impact = trung bình impact có trọng số relevance
    (high=1, medium=0.6, low=0.3) hoặc đơn giản là max impact trong ngày.
    """
    if articles_with_impact.empty:
        return pd.DataFrame(columns=["date", "news_impact", "article_count"])

    df = articles_with_impact.copy()
    weight = df["relevance"].map({"high": 1.0, "medium": 0.6, "low": 0.3}).fillna(0.5)
    df["weighted_impact"] = df["impact_on_gold"] * weight
    daily = df.groupby("date", as_index=False).agg(
        news_impact=("weighted_impact", "mean"),
        article_count=("impact_on_gold", "count"),
    )
    daily["news_impact"] = daily["news_impact"].round(3)
    return daily.sort_values("date").reset_index(drop=True)


def fetch_marketaux_articles(
    marketaux_key: str,
    start_date: str,
    end_date: str,
    countries: str = "us",
    search: str = "gold",
) -> pd.DataFrame:
    """Lấy tin từ Marketaux (free), trả về DataFrame có _date, _title, _snippet."""
    import datetime as dt
    start = dt.datetime.fromisoformat(start_date).date()
    end = dt.datetime.fromisoformat(end_date).date()
    all_items = []
    step = dt.timedelta(days=7)
    cur = start
    while cur <= end:
        win_end = min(end, cur + step - dt.timedelta(days=1))
        win_start_dt = dt.datetime.combine(cur, dt.time.min)
        win_end_dt = dt.datetime.combine(win_end, dt.time.max)
        params = {
            "api_token": marketaux_key,
            "language": "en",
            "filter_entities": "true",
            "published_after": win_start_dt.isoformat(timespec="minutes"),
            "published_before": win_end_dt.isoformat(timespec="minutes"),
            "limit": 100,
            "countries": countries,
            "search": search,
        }
        for attempt in range(3):
            try:
                r = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=60)
                r.raise_for_status()
                data = r.json().get("data") or []
                for art in data:
                    pub = art.get("published_at") or art.get("published")
                    if not pub:
                        continue
                    try:
                        ts = pd.to_datetime(pub, utc=True, errors="coerce")
                    except Exception:
                        continue
                    if pd.isna(ts):
                        continue
                    all_items.append({
                        "_date": ts.date(),
                        "_title": (art.get("title") or art.get("name") or "").strip() or "(no title)",
                        "_snippet": (art.get("description") or art.get("snippet") or "").strip()[:500],
                    })
                break
            except (requests.exceptions.RequestException, ConnectionError) as e:
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise e from e
        cur = win_end + dt.timedelta(days=1)
    if not all_items:
        return pd.DataFrame(columns=["_date", "_title", "_snippet"])
    df = pd.DataFrame(all_items).drop_duplicates(subset=["_date", "_title"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá mức độ ảnh hưởng tin tức lên giá vàng bằng LLM (OpenAI)."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (hoặc OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--from-marketaux",
        action="store_true",
        help="Lấy tin trực tiếp từ Marketaux rồi đưa vào LLM (cần --marketaux-api-key, --start, --end).",
    )
    parser.add_argument(
        "--marketaux-api-key",
        type=str,
        default=None,
        help="Marketaux API key (khi dùng --from-marketaux). Hoặc MARKETAUX_API_KEY.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Ngày bắt đầu (YYYY-MM-DD), dùng với --from-marketaux.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=pd.Timestamp.now().strftime("%Y-%m-%d"),
        help="Ngày kết thúc (YYYY-MM-DD), dùng với --from-marketaux.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV tin tức: cột date (hoặc timestamp), title, (description/snippet).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Số giây nghỉ giữa mỗi request để tránh rate limit.",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=Path(DAILY_OUTPUT),
        help=f"File CSV tổng hợp theo ngày (default: {DAILY_OUTPUT}).",
    )
    parser.add_argument(
        "--articles-output",
        type=Path,
        default=Path(ARTICLES_OUTPUT),
        help=f"File CSV chi tiết từng tin (default: {ARTICLES_OUTPUT}).",
    )
    args = parser.parse_args()

    api_key = _get_openai_key(args.api_key)

    if args.from_marketaux:
        mk = args.marketaux_api_key or os.getenv("MARKETAUX_API_KEY")
        if not mk:
            raise SystemExit("--from-marketaux cần --marketaux-api-key hoặc MARKETAUX_API_KEY.")
        print(f"[INFO] Fetching news from Marketaux ({args.start} -> {args.end}) ...")
        articles_df = fetch_marketaux_articles(mk, args.start, args.end)
        print(f"[INFO] Got {len(articles_df)} articles from Marketaux.")
    elif args.input_csv and args.input_csv.exists():
        articles_df = load_articles_from_csv(args.input_csv)
        print(f"[INFO] Loaded {len(articles_df)} articles from {args.input_csv}")
    else:
        raise SystemExit(
            "Cần --from-marketaux (và --marketaux-api-key) hoặc --input-csv <file>."
        )
    if articles_df.empty:
        raise SystemExit("Không có tin nào để đánh giá.")

    result_df = run_assessment(api_key, articles_df, model=args.model, delay_seconds=args.delay)
    if result_df.empty:
        raise SystemExit("No assessed articles.")

    daily_df = aggregate_daily_impact(result_df)

    out_daily = args.daily_output if args.daily_output.is_absolute() else Path.cwd() / args.daily_output
    out_articles = args.articles_output if args.articles_output.is_absolute() else Path.cwd() / args.articles_output
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_daily, index=False, encoding="utf-8-sig")
    result_df.to_csv(out_articles, index=False, encoding="utf-8-sig")
    print(f"[OK] Daily impact: {out_daily}  (rows={len(daily_df)})")
    print(f"[OK] Articles detail: {out_articles}  (rows={len(result_df)})")
    print("\nGợi ý: Dùng news_impact làm 1 feature hoặc overlay, không nhồi nhiều cột tin vào ML.")


if __name__ == "__main__":
    main()
