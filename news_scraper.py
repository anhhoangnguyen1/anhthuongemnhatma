"""
Scrape gold-related news from Vietnamese and international RSS feeds.

Sources:
  - VnExpress Kinh doanh (VN)
  - CafeF Hang hoa (VN)
  - Kitco News (EN)
  - Google News gold search (EN)

Usage:
    from news_scraper import fetch_all_gold_news
    articles = fetch_all_gold_news()
"""
from __future__ import annotations

import re
import html
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional

import feedparser

GOLD_KEYWORDS_VN = [
    "vàng", "sjc", "gold", "kim loại quý", "giá vàng",
    "vàng miếng", "vàng nhẫn", "vàng thế giới", "xauusd",
]
GOLD_KEYWORDS_EN = [
    "gold", "xau", "precious metal", "bullion", "gold price",
    "gold futures", "gold market", "central bank gold",
]

RSS_SOURCES = [
    {
        "name": "VnExpress",
        "url": "https://vnexpress.net/rss/kinh-doanh.rss",
        "lang": "vi",
        "keywords": GOLD_KEYWORDS_VN,
    },
    {
        "name": "CafeF",
        "url": "https://cafef.vn/hang-hoa-nguyen-lieu.rss",
        "lang": "vi",
        "keywords": GOLD_KEYWORDS_VN,
    },
    {
        "name": "Kitco",
        "url": "https://www.kitco.com/feed/rss/news/gold",
        "lang": "en",
        "keywords": GOLD_KEYWORDS_EN,
    },
    {
        "name": "Google News",
        "url": "https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en",
        "lang": "en",
        "keywords": GOLD_KEYWORDS_EN,
    },
]


@dataclass
class NewsArticle:
    title: str
    summary: str
    url: str
    source: str
    published: Optional[str]
    language: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_html(raw: str) -> str:
    """Strip HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _parse_date(entry: dict) -> str | None:
    for field in ("published_parsed", "updated_parsed"):
        tp = entry.get(field)
        if tp:
            try:
                return dt.datetime(*tp[:6]).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
    for field in ("published", "updated"):
        val = entry.get(field)
        if val:
            return str(val)[:19]
    return None


def fetch_from_rss(
    url: str,
    source_name: str,
    lang: str,
    keywords: list[str],
    max_articles: int = 30,
) -> list[NewsArticle]:
    """Fetch and filter articles from a single RSS feed."""
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    articles: list[NewsArticle] = []
    for entry in feed.entries[:max_articles * 3]:
        title = _clean_html(entry.get("title", ""))
        summary = _clean_html(entry.get("summary", entry.get("description", "")))
        combined = f"{title} {summary}"

        if not _matches_keywords(combined, keywords):
            continue

        articles.append(
            NewsArticle(
                title=title,
                summary=summary[:300],
                url=entry.get("link", ""),
                source=source_name,
                published=_parse_date(entry),
                language=lang,
            )
        )
        if len(articles) >= max_articles:
            break

    return articles


def fetch_all_gold_news(max_per_source: int = 15) -> list[NewsArticle]:
    """Fetch gold news from all configured RSS sources."""
    all_articles: list[NewsArticle] = []
    for src in RSS_SOURCES:
        try:
            articles = fetch_from_rss(
                url=src["url"],
                source_name=src["name"],
                lang=src["lang"],
                keywords=src["keywords"],
                max_articles=max_per_source,
            )
            all_articles.extend(articles)
        except Exception:
            continue

    all_articles.sort(key=lambda a: a.published or "", reverse=True)
    return all_articles


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    articles = fetch_all_gold_news()
    print(f"Found {len(articles)} gold news articles:")
    for a in articles[:10]:
        print(f"  [{a.source}] [{a.language}] {a.published} - {a.title[:80]}")
