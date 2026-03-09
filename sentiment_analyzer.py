"""
Sentiment analysis for gold-related news articles.

- English: FinVADER (VADER with financial lexicon ~7300 terms)
- Vietnamese: keyword-based scoring with gold/finance-specific lexicon

Usage:
    from sentiment_analyzer import analyze_sentiment, analyze_articles
    score, label = analyze_sentiment("Gold prices surge to record high", "en")
    results = analyze_articles(articles)  # list of NewsArticle
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from news_scraper import NewsArticle

_HAS_FINVADER = False
_vader = None

try:
    from finvader import finvader
    _HAS_FINVADER = True
except ImportError:
    pass

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    pass


# Vietnamese gold/finance sentiment lexicon
# score: positive words > 0, negative words < 0
_VN_LEXICON: list[tuple[str, float]] = [
    # Strong positive
    ("tang manh", 0.8), ("ky luc", 0.9), ("but pha", 0.8),
    ("tang vot", 0.9), ("cao nhat", 0.7), ("lac quan", 0.7),
    ("tang toc", 0.7), ("nhu cau cao", 0.6), ("tang gia", 0.6),
    ("dat dinh", 0.8), ("bung no", 0.7), ("tang phi ma", 0.9),
    ("xu huong tang", 0.7), ("tich cuc", 0.6),
    # Mild positive
    ("tang", 0.4), ("phuc hoi", 0.5), ("on dinh", 0.2),
    ("ho tro", 0.3), ("khoi sac", 0.5), ("tang nhe", 0.3),
    # Strong negative
    ("giam manh", -0.8), ("giam sau", -0.9), ("lao doc", -0.9),
    ("sut giam", -0.7), ("thap nhat", -0.7), ("bi quan", -0.7),
    ("ban thao", -0.8), ("mat gia", -0.6), ("do vo", -0.9),
    ("giam soc", -0.9), ("rui ro", -0.5), ("suy thoai", -0.7),
    ("xu huong giam", -0.7), ("tieu cuc", -0.6),
    # Mild negative
    ("giam", -0.4), ("dieu chinh", -0.3), ("bien dong", -0.2),
    ("giam nhe", -0.3), ("lo ngai", -0.4), ("bat on", -0.4),
    # Neutral
    ("di ngang", 0.0), ("duy tri", 0.1), ("cho doi", 0.0),
]


def _strip_diacritics(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def _analyze_vietnamese(text: str) -> float:
    """Keyword-based sentiment scoring for Vietnamese text."""
    normalized = _strip_diacritics(text)
    score = 0.0
    matches = 0
    for keyword, weight in _VN_LEXICON:
        count = normalized.count(keyword)
        if count > 0:
            score += weight * count
            matches += count
    if matches == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(matches, 1)))


def _analyze_english(text: str) -> float:
    """FinVADER or fallback VADER sentiment for English text."""
    if _HAS_FINVADER:
        try:
            result = finvader(text, use_sentibignomics=True, use_henry=True)
            if isinstance(result, dict) and "compound" in result:
                return float(result["compound"])
        except Exception:
            pass
    if _vader is not None:
        try:
            scores = _vader.polarity_scores(text)
            return float(scores["compound"])
        except Exception:
            pass
    return 0.0


def analyze_sentiment(text: str, language: str = "en") -> tuple[float, str]:
    """Analyze sentiment of a text string.

    Returns (score, label) where:
      score: float in [-1.0, 1.0]
      label: "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    """
    if language == "vi":
        score = _analyze_vietnamese(text)
    else:
        score = _analyze_english(text)

    if score >= 0.15:
        label = "POSITIVE"
    elif score <= -0.15:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return round(score, 4), label


@dataclass
class SentimentResult:
    title: str
    summary: str
    url: str
    source: str
    published: str | None
    language: str
    sentiment_score: float
    sentiment_label: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "source": self.source,
            "published": self.published,
            "language": self.language,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
        }


def analyze_articles(articles: list) -> list[SentimentResult]:
    """Analyze sentiment for a list of NewsArticle objects."""
    results: list[SentimentResult] = []
    for article in articles:
        combined_text = f"{article.title} {article.summary}"
        score, label = analyze_sentiment(combined_text, article.language)
        results.append(
            SentimentResult(
                title=article.title,
                summary=article.summary,
                url=article.url,
                source=article.source,
                published=article.published,
                language=article.language,
                sentiment_score=score,
                sentiment_label=label,
            )
        )
    return results


def compute_daily_sentiment(results: list[SentimentResult]) -> dict[str, float]:
    """Aggregate sentiment scores by date (YYYY-MM-DD)."""
    from collections import defaultdict
    daily: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.published:
            date_str = r.published[:10]
            daily[date_str].append(r.sentiment_score)
    return {d: sum(s) / len(s) for d, s in daily.items() if s}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from news_scraper import fetch_all_gold_news

    articles = fetch_all_gold_news()
    results = analyze_articles(articles)
    print(f"Analyzed {len(results)} articles:\n")
    for r in results[:15]:
        emoji = {"POSITIVE": "+", "NEGATIVE": "-", "NEUTRAL": "~"}[r.sentiment_label]
        print(f"  [{emoji}{r.sentiment_score:+.2f}] [{r.source}] {r.title[:70]}")

    daily = compute_daily_sentiment(results)
    print(f"\nDaily sentiment ({len(daily)} days):")
    for d, s in sorted(daily.items(), reverse=True)[:5]:
        print(f"  {d}: {s:+.3f}")
