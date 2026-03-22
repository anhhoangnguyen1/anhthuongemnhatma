"""
llm_adjust.py  —  v2.0
========================
Module B của DSS Vàng — lấy tin tức tiếng Việt và phân tích bằng LLM.

Chiến lược lấy tin (3 nguồn, ưu tiên theo thứ tự):
  1. RSS trực tiếp từ báo VN (VnExpress, CafeF, Tuổi Trẻ...) — MIỄN PHÍ, không cần key
  2. NewsAPI.org            — 100 req/ngày free, tin VN tốt hơn GNews
  3. GNews.io               — fallback

Cài thư viện:
  pip install openai requests feedparser python-dotenv

Biến môi trường trong .env:
  OPENAI_API_KEY=sk-...
  NEWSAPI_KEY=...       (đăng ký tại newsapi.org — miễn phí)
  GNEWS_API_KEY=...     (tuỳ chọn, fallback — gnews.io)
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

# ── Cấu hình ───────────────────────────────────────────────────────────────────
OPENAI_MODEL    = "gpt-4o-mini"
REQUEST_TIMEOUT = 12
MAX_NEWS        = 8

# Các RSS feed tiếng Việt liên quan đến tài chính / vàng
VN_RSS_FEEDS = [
    # ── Nhóm 1: Báo lớn, cập nhật nhanh nhất ──────────────────────────────
    ("VnExpress Kinh doanh",  "https://vnexpress.net/rss/kinh-doanh.rss"),
    ("VnExpress Hàng hoá",    "https://vnexpress.net/rss/kinh-doanh/hang-hoa.rss"),
    ("CafeF",                 "https://cafef.vn/thi-truong-chung-khoan.rss"),
    ("Tuổi Trẻ Kinh tế",      "https://tuoitre.vn/rss/kinh-te.rss"),
    ("Thanh Niên Tài chính",  "https://thanhnien.vn/rss/tai-chinh-kinh-doanh.rss"),
    ("ZingNews Tài chính",    "https://zingnews.vn/tai-chinh.rss"),
    ("ZingNews Kinh doanh",   "https://zingnews.vn/kinh-doanh.rss"),

    # ── Nhóm 2: Nguồn user cung cấp ───────────────────────────────────────
    ("VietnamNet Kinh doanh", "https://vietnamnet.vn/rss/kinh-doanh.rss"),
    ("VietnamNet Tài chính",  "https://vietnamnet.vn/rss/tai-chinh.rss"),
    ("24h Kinh doanh",        "https://www.24h.com.vn/upload/rss/taichinh-chungkhoan.xml"),
    ("24h Vàng",              "https://www.24h.com.vn/upload/rss/vang-tien-te.xml"),
    ("VOV Thị trường",        "https://vov.vn/rss/thi-truong.rss"),
    ("VOV Kinh tế",           "https://vov.vn/rss/kinh-te.rss"),
    ("Dân Việt Kinh tế",      "https://danviet.vn/kinh-te.rss"),
    ("Báo An Giang",          "https://baoangiang.com.vn/rss/kinh-te-xa-hoi.rss"),

    # ── Nhóm 3: Tài chính chuyên sâu ──────────────────────────────────────
    ("VnEconomy Tài chính",   "https://vneconomy.vn/tai-chinh.rss"),
    ("VnEconomy Chứng khoán", "https://vneconomy.vn/chung-khoan.rss"),
    ("Báo Đầu tư",            "https://baodautu.vn/rss/thi-truong.rss"),
    ("VietnamBiz",            "https://vietnambiz.vn/kinh-te.rss"),
    ("NDHMoney",              "https://ndh.vn/rss/thi-truong.rss"),

    # ── Nhóm 4: Báo tổng hợp ──────────────────────────────────────────────
    ("DanTri Kinh doanh",     "https://dantri.com.vn/kinh-doanh.rss"),
    ("PLO Kinh tế",           "https://plo.vn/kinh-te.rss"),
    ("Lao Động Kinh tế",      "https://laodong.vn/rss/kinh-te.rss"),
    ("Người Lao Động KT",     "https://nld.com.vn/kinh-te.rss"),
    ("Tiền Phong Kinh tế",    "https://tienphong.vn/rss/kinh-te.rss"),
]

# Từ khoá lọc tin liên quan đến vàng
# ── Từ khoá MẠNH: cụm từ đặc trưng, chỉ xuất hiện khi nói về giá vàng ──────
_STRONG_KEYWORDS = [
    "giá vàng", "vàng sjc", "vàng miếng", "vàng 9999", "vàng 24k",
    "xau/usd", "xauusd", "đấu thầu vàng", "bán vàng miếng",
    "vàng thế giới", "ounce vàng", "vàng tăng", "vàng giảm",
    "chênh lệch vàng", "vàng nữ trang", "doji vàng", "pnj vàng",
    "bảo tín minh châu", "giá vàng hôm nay", "thị trường vàng",
    "vàng 9999", "vàng sjc", "nhnn bán vàng", "nhnn đấu thầu",
]

# ── Ngữ cảnh tài chính đi kèm từ "vàng" → bài liên quan ─────────────────────
_FINANCIAL_CONTEXT = [
    "tăng", "giảm", "ổn định", "biến động", "đầu tư", "nhà đầu tư",
    "thị trường", "ngân hàng", "nhnn", "tỷ giá", "usd",
    "triệu", "lượng", "chỉ vàng", "đồng/lượng",
]

# ── Từ khoá NHIỄU: loại bỏ nếu tiêu đề chỉ có "vàng" nghĩa bóng ─────────────
_NOISE_TITLE = [
    "huy chương vàng", "huy chương", "giải vàng", "vàng son",
    "thời kỳ vàng", "tuổi vàng", "bí quyết vàng", "cơ hội vàng",
    "gold coast", "golden", "vàng anh", "vàng tâm", "da vàng",
    "nước da vàng", "màu vàng", "đất vàng", "khu đất vàng",
    "rác thải", "kim loại tái chế", "huy chương bạc",
]

NEWSAPI_URL = "https://newsapi.org/v2/everything"
GNEWS_URL   = "https://gnews.io/api/v4/search"
OPENAI_URL  = "https://api.openai.com/v1/chat/completions"

_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


# ── Nguồn 1: RSS trực tiếp báo VN ─────────────────────────────────────────────
def _is_gold_price_news(title: str, snippet: str = "") -> bool:
    """
    Trả về True chỉ khi bài viết thực sự về GIÁ VÀNG.

    Logic 2 tầng:
      1. Tiêu đề có từ khoá MẠNH → giữ ngay
      2. Tiêu đề có "vàng" + ngữ cảnh tài chính, KHÔNG có từ nhiễu → giữ
    """
    t    = title.lower()
    # Tầng 1: từ khoá mạnh trong tiêu đề
    if any(kw in t for kw in _STRONG_KEYWORDS):
        return True
    # Tầng 2: "vàng" + ngữ cảnh tài chính, không phải nghĩa bóng
    if "vàng" in t:
        if any(noise in t for noise in _NOISE_TITLE):
            return False
        if any(ctx in t for ctx in _FINANCIAL_CONTEXT):
            return True
    return False


def _fetch_rss(target_date: date, max_total: int = MAX_NEWS) -> list[dict]:
    """Đọc RSS từ các báo VN, lọc tin liên quan đến vàng trong ±2 ngày."""
    if not _HAS_FEEDPARSER:
        return []

    date_from = datetime.combine(target_date - timedelta(days=2), datetime.min.time())
    date_to   = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
    results: list[dict] = []

    for source_name, feed_url in VN_RSS_FEEDS:
        if len(results) >= max_total:
            break
        try:
            # Fetch raw content với headers để bypass 403
            resp = requests.get(feed_url, headers=_RSS_HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue
            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                if len(results) >= max_total:
                    break

                title   = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                link    = entry.get("link", "")
                # Parse published date
                pub = None
                for attr in ("published_parsed", "updated_parsed"):
                    raw = entry.get(attr)
                    if raw:
                        try:
                            pub = datetime(*raw[:6])
                        except Exception:
                            pass
                        break

                # Lọc theo thời gian
                if pub and not (date_from <= pub <= date_to):
                    continue

                # Lọc chặt: chỉ lấy bài thực sự về GIÁ VÀNG
                if not _is_gold_price_news(title, summary):
                    continue

                results.append({
                    "title":     title,
                    "snippet":   summary[:300].strip(),
                    "url":       link,
                    "published": pub.strftime("%Y-%m-%d %H:%M") if pub else "",
                    "source":    source_name,
                })
        except Exception:
            continue

    return results


# ── Nguồn 2: NewsAPI.org ────────────────────────────────────────────────────────
def _fetch_newsapi(
    target_date: date,
    newsapi_key: str,
    max_results: int = 5,
) -> list[dict]:
    """Tìm kiếm tin tức qua NewsAPI.org — hỗ trợ nguồn VN tốt hơn GNews."""
    date_from = (target_date - timedelta(days=2)).isoformat()
    date_to   = (target_date + timedelta(days=1)).isoformat()

    # Thử với từ khoá tiếng Việt trước
    for q in ["vàng SJC OR giá vàng OR NHNN vàng", "gold price Vietnam OR XAU USD"]:
        try:
            resp = requests.get(
                NEWSAPI_URL,
                params={
                    "q":        q,
                    "from":     date_from,
                    "to":       date_to,
                    "sortBy":   "relevance",
                    "pageSize": max_results,
                    "language": "vi" if "vàng" in q else "en",
                    "apiKey":   newsapi_key,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 426:  # developer plan required for vi
                continue
            resp.raise_for_status()
            data = resp.json()
            arts = data.get("articles") or []
            if arts:
                return [
                    {
                        "title":     a.get("title", ""),
                        "snippet":   (a.get("description") or "")[:300],
                        "url":       a.get("url", ""),
                        "published": (a.get("publishedAt") or "")[:10],
                        "source":    (a.get("source") or {}).get("name", "NewsAPI"),
                    }
                    for a in arts if a.get("title")
                ]
        except Exception:
            continue
    return []


# ── Nguồn 3: GNews.io (fallback) ───────────────────────────────────────────────
def _fetch_gnews(
    target_date: date,
    gnews_key: str,
    lang: str = "vi",
    max_results: int = 4,
) -> list[dict]:
    """GNews API — fallback khi các nguồn khác không trả đủ tin."""
    dt      = datetime.combine(target_date, datetime.min.time())
    q       = ("vàng SJC OR giá vàng OR NHNN vàng"
                if lang == "vi"
                else "gold price Vietnam OR XAU USD")
    params  = {
        "q":       q,
        "lang":    lang,
        "from":    (dt - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to":      (dt + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max":     max_results,
        "sortby":  "relevance",
        "apikey":  gnews_key,
    }
    try:
        resp = requests.get(GNEWS_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        arts = resp.json().get("articles") or []
        return [
            {
                "title":     a.get("title", ""),
                "snippet":   (a.get("description") or "")[:300],
                "url":       a.get("url", ""),
                "published": (a.get("publishedAt") or "")[:10],
                "source":    (a.get("source") or {}).get("name", "GNews"),
            }
            for a in arts if a.get("title")
        ]
    except Exception:
        return []


# ── Hợp nhất tin tức từ tất cả nguồn ─────────────────────────────────────────
def _collect_news(
    target_date: date,
    newsapi_key: str | None = None,
    gnews_key:   str | None = None,
    max_total:   int = MAX_NEWS,
) -> list[dict]:
    """
    Thu thập tin tức từ nhiều nguồn, ưu tiên:
    1. RSS báo VN (không cần key)
    2. NewsAPI.org (nếu có key)
    3. GNews (nếu có key, fallback)
    Loại bỏ tin trùng tiêu đề.
    """
    news: list[dict] = []
    seen_titles: set[str] = set()

    def _add(items: list[dict]) -> None:
        for item in items:
            t = item.get("title", "").strip().lower()[:80]
            if t and t not in seen_titles and len(news) < max_total:
                seen_titles.add(t)
                news.append(item)

    # 1. RSS
    _add(_fetch_rss(target_date, max_total))

    # 2. NewsAPI nếu chưa đủ
    if len(news) < 3 and newsapi_key:
        _add(_fetch_newsapi(target_date, newsapi_key))

    # 3. GNews nếu vẫn chưa đủ
    if len(news) < 3 and gnews_key:
        _add(_fetch_gnews(target_date, gnews_key, lang="vi"))
        if len(news) < 3:
            _add(_fetch_gnews(target_date, gnews_key, lang="en", max_results=3))

    return news[:max_total]


# ── LLM ────────────────────────────────────────────────────────────────────────
_SYSTEM = """Bạn là chuyên gia phân tích thị trường vàng Việt Nam 15 năm kinh nghiệm.
Bạn am hiểu: chính sách NHNN, premium SJC, tỷ giá USD/VND, chu kỳ Tết/Vía Thần Tài.
Trả lời ngắn gọn, thực tế. Không đưa ra lời khuyên đầu tư tuyệt đối."""


def _make_prompt(
    target_date: date,
    price_vnd:   float,
    ml_signal:   str | None,
    news:        list[dict],
) -> str:
    if news:
        news_block = "\n".join(
            f"{i}. [{n['source']} {n['published'][:10]}] {n['title']}"
            + (f"\n   {n['snippet'][:200]}" if n.get("snippet") else "")
            for i, n in enumerate(news[:MAX_NEWS], 1)
        )
    else:
        news_block = "Không tìm thấy tin tức liên quan trong 48h qua."

    return f"""Ngày: {target_date.strftime('%d/%m/%Y')}
Giá vàng SJC: {price_vnd:,.0f} VND/lượng
Tín hiệu XGBoost: {ml_signal or 'N/A'}

TIN TỨC:
{news_block}

Trả về JSON THUẦN TÚY (không markdown, không ```) theo định dạng:
{{
  "adjusted_signal": "BUY" hoặc "NOT_BUY",
  "reasoning": "2-3 câu tiếng Việt: tin tức nào ảnh hưởng và theo hướng nào",
  "confidence": 0.0-1.0,
  "key_risk": "1 câu rủi ro chính nếu mua hôm nay",
  "updated_price_note": "1 câu nhận xét về mức giá hiện tại"
}}

Nguyên tắc:
- NHNN đấu thầu/kiểm soát vàng → giảm premium → NOT_BUY
- USD yếu, bất ổn địa chính trị → hỗ trợ vàng → BUY
- Không đủ tin → giữ nguyên tín hiệu XGBoost, confidence 0.5"""


def _call_llm(api_key: str, prompt: str) -> dict | None:
    try:
        resp = requests.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model":       OPENAI_MODEL,
                "temperature": 0.2,
                "max_tokens":  500,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Bỏ markdown wrapper nếu có
        if raw.startswith("```"):
            raw = "\n".join(
                l for l in raw.splitlines() if not l.strip().startswith("```")
            ).strip()
        data   = json.loads(raw)
        signal = str(data.get("adjusted_signal", "NOT_BUY")).upper()
        if signal not in ("BUY", "NOT_BUY"):
            signal = "NOT_BUY"
        conf = float(data.get("confidence", 0.5))
        return {
            "adjusted_signal":    signal,
            "reasoning":          str(data.get("reasoning", "")),
            "confidence":         round(max(0.0, min(1.0, conf)), 2),
            "key_risk":           str(data.get("key_risk", "")),
            "updated_price_note": str(data.get("updated_price_note", "")),
        }
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 0
        if code == 401:
            print("[OpenAI] API key không hợp lệ")
        elif code == 429:
            print("[OpenAI] Rate limit — thử lại sau")
        else:
            print(f"[OpenAI] HTTP {code}: {e}")
    except json.JSONDecodeError:
        print("[OpenAI] Không parse được JSON từ LLM")
    except Exception as e:
        print(f"[OpenAI] Lỗi: {e}")
    return None


# ── API PUBLIC — được app.py gọi ───────────────────────────────────────────────
def get_news_and_llm_supplement_for_date(
    prediction_date: date,
    price_vnd:       float,
    api_key:         str,
    gnews_key:       str,
    ml_signal:       str | None = None,
) -> dict:
    """
    Dùng cho /api/date-detail — user click vào 1 ngày trên chart.

    Returns:
        {
            "news": [...],
            "llm_supplement": {adjusted_signal, reasoning, confidence,
                               key_risk, updated_price_note} | None
        }
    """
    newsapi_key = os.getenv("NEWSAPI_KEY")
    news        = _collect_news(prediction_date, newsapi_key, gnews_key)
    llm_result  = _call_llm(api_key, _make_prompt(prediction_date, price_vnd, ml_signal, news))
    return {"news": news, "llm_supplement": llm_result}


def run_llm_adjust_for_latest(
    root_path:    "Path | str",
    pred_date_str: str | None,
    ml_signal:    str | None,
    price_vnd:    float | None,
) -> dict | None:
    """
    Dùng cho /api/predict?llm=1 — điều chỉnh tín hiệu mới nhất.

    Returns:
        {adjusted_signal, reasoning, confidence, key_risk, updated_price_note}
        hoặc None nếu thiếu API key.
    """
    api_key   = os.getenv("OPENAI_API_KEY")
    gnews_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        print("[LLM] Thiếu OPENAI_API_KEY")
        return None

    target_date = date.today()
    if pred_date_str:
        try:
            target_date = date.fromisoformat(pred_date_str[:10])
        except ValueError:
            pass

    newsapi_key = os.getenv("NEWSAPI_KEY")
    news        = _collect_news(target_date, newsapi_key, gnews_key, max_total=6)
    return _call_llm(api_key, _make_prompt(target_date, float(price_vnd or 0), ml_signal, news))


# ── Test chạy trực tiếp ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        load_dotenv()
    except ImportError:
        pass

    api_key   = os.getenv("OPENAI_API_KEY", "")
    gnews_key = os.getenv("GNEWS_API_KEY",  "")

    if not _HAS_FEEDPARSER:
        print("Cài feedparser trước: pip install feedparser")
        sys.exit(1)

    test_date = date.today()
    print(f"Test ngày: {test_date}")

    print("\n[1] Thu thập tin tức...")
    news = _collect_news(test_date, os.getenv("NEWSAPI_KEY"), gnews_key or None)
    print(f"    Tổng: {len(news)} bài")
    for n in news:
        print(f"    [{n['source']}] {n['title'][:70]}")

    if not api_key:
        print("\nThiếu OPENAI_API_KEY — bỏ qua bước LLM")
        sys.exit(0)

    print("\n[2] Phân tích LLM...")
    result = get_news_and_llm_supplement_for_date(
        prediction_date = test_date,
        price_vnd       = 172_000_000,
        api_key         = api_key,
        gnews_key       = gnews_key,
        ml_signal       = "BUY",
    )
    sup = result.get("llm_supplement")
    if sup:
        print(f"    Tín hiệu điều chỉnh : {sup['adjusted_signal']}")
        print(f"    Lý do               : {sup['reasoning']}")
        print(f"    Độ tin cậy          : {sup['confidence']}")
        print(f"    Rủi ro chính        : {sup['key_risk']}")
        print(f"    Ghi chú giá         : {sup['updated_price_note']}")
    else:
        print("    Không có kết quả LLM (kiểm tra API key)")