"""Pure scoring: themes, source trust, recency, aggregation. No I/O."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.news_fetcher import _score_headline_sentiment

# Source trust: how much a headline from this publisher moves the pulse.
# Wire services report; opinion/social amplifies. Unknown defaults to 0.5.
DEFAULT_TRUST: Dict[str, float] = {
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "wsj.com": 1.0,
    "bloomberg.com": 0.95,
    "ft.com": 0.95,
    "cnbc.com": 0.85,
    "marketwatch.com": 0.8,
    "barrons.com": 0.8,
    "finance.yahoo.com": 0.7,
    "yahoo.com": 0.7,
    "investors.com": 0.7,
    "businessinsider.com": 0.6,
    "fool.com": 0.4,
    "seekingalpha.com": 0.5,
    "benzinga.com": 0.45,
    "stocktwits.com": 0.3,
    "reddit.com": 0.3,
}

# Order matters: more specific themes first ("tariff war" must hit
# trade_tariffs before geopolitics catches "war").
THEMES: Dict[str, List[str]] = {
    "fed_rates": ["fed ", "federal reserve", "fomc", "powell", "rate cut",
                  "rate hike", "interest rate", "monetary policy", "treasury yield"],
    "inflation": ["inflation", "cpi", "ppi", "pce", "price pressures", "deflation"],
    "jobs": ["jobs report", "payrolls", "unemployment", "labor market", "jobless"],
    "trade_tariffs": ["tariff", "trade war", "trade deal", "export controls",
                      "trade talks", "import duties"],
    "geopolitics": ["war", "strike", "missile", "invasion", "conflict",
                    "geopolit", "sanctions", "nato", "middle east", "ukraine",
                    "taiwan", "north korea"],
    "earnings_tech": ["earnings", "guidance", "revenue beat", "profit",
                      "ai ", "chip", "semiconductor", "nvidia", "apple",
                      "microsoft", "big tech"],
    "energy": ["oil", "opec", "crude", "natural gas", "energy prices"],
    "crypto": ["bitcoin", "crypto", "ethereum", "stablecoin"],
}

# Macro-market vocabulary the per-ticker lexicon (and TextBlob's generic
# polarity model) doesn't cover. Each hit nudges ±0.25, clamped at ±1.
_MACRO_NEGATIVE = {
    "plunge", "plunges", "crash", "crashes", "collapse", "sell-off", "selloff",
    "recession", "fears", "tumble", "tumbles", "slump", "slumps", "sinks",
    "escalates", "escalation", "turmoil", "crisis", "default", "stagflation",
    "slides", "slip", "slips", "rout", "panic", "contagion", "downturn",
    "bear market", "falls", "fall ", "drops", "weak", "warns", "war",
}
_MACRO_POSITIVE = {
    "rally", "rallies", "surge", "surges", "soar", "soars", "jumps", "gains",
    "record high", "rebound", "rebounds", "optimism", "beats expectations",
    "strong growth", "expansion", "recovery", "bull market", "rises", "climbs",
    "cools", "easing", "ceasefire", "truce", "resolution", "breakthrough",
}


# For inflation headlines the lexicon's price-direction words invert: prices
# RISING is bad for markets, prices COOLING is good.
_INFLATION_HOT = ("rose", "rise", "rises", "surge", "jump", "accelerat",
                  "faster", "hotter", "more than expected", "fastest pace",
                  "climbs", "picks up")
_INFLATION_COOL = ("cools", "cooling", "eases", "easing", "slows", "slowing",
                   "below expectations", "less than expected", "moderates",
                   "falls", "drops")


def headline_sentiment(title: str, theme: Optional[str] = None) -> float:
    """Per-ticker lexicon + TextBlob base, supplemented with macro vocabulary.
    Inflation-theme headlines get their hot/cool direction enforced explicitly
    (hot prints are market-negative regardless of upbeat verbs)."""
    base = _score_headline_sentiment(title or "")
    t = (title or "").lower()
    neg = sum(1 for k in _MACRO_NEGATIVE if k in t)
    pos = sum(1 for k in _MACRO_POSITIVE if k in t)
    s = base + 0.25 * (pos - neg)
    if theme == "inflation":
        # cool-words first: "cools more than expected" must read as cooling
        if any(k in t for k in _INFLATION_COOL):
            s = max(s, 0.0) + 0.3
        elif any(k in t for k in _INFLATION_HOT):
            s = min(s, 0.0) - 0.3
    return max(-1.0, min(1.0, s))


def classify_theme(title: str) -> str:
    t = f" {(title or '').lower()} "
    for theme, keys in THEMES.items():
        if any(k in t for k in keys):
            return theme
    return "other"


def source_trust(domain: str, overrides: Optional[Dict[str, float]] = None) -> float:
    table = {**DEFAULT_TRUST, **(overrides or {})}
    d = (domain or "").lower().removeprefix("www.")
    return table.get(d, 0.5)


def recency_weight(published: Optional[datetime], now: datetime,
                   half_life_hours: float = 24.0) -> float:
    """exp decay with a 24h half-life; undated items get a conservative 0.5."""
    if published is None:
        return 0.5
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_h = max(0.0, (now - published).total_seconds() / 3600.0)
    return 0.5 ** (age_h / half_life_hours)


def aggregate(items: List[Dict[str, Any]], now: Optional[datetime] = None,
              trust_overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Items [{title, source, published, url}] → trust/recency-weighted pulse.

    Returns {pulse, bull_pct, bear_pct, confidence, n_items, n_sources,
    themes: {name: {score, n}}, top: [...]}.
    """
    now = now or datetime.now(timezone.utc)
    scored = []
    for it in items:
        theme = classify_theme(it.get("title") or "")
        s = headline_sentiment(it.get("title") or "", theme=theme)
        w = (source_trust(it.get("source") or "", trust_overrides)
             * recency_weight(it.get("published"), now))
        scored.append({**it, "sentiment": s, "weight": w, "theme": theme})

    total_w = sum(x["weight"] for x in scored)
    if not scored or total_w <= 0:
        return {"pulse": 0.0, "bull_pct": 0.5, "bear_pct": 0.5, "confidence": 0,
                "n_items": 0, "n_sources": 0, "themes": {}, "top": []}

    pulse = max(-1.0, min(1.0, sum(x["sentiment"] * x["weight"] for x in scored) / total_w))

    directional = [x for x in scored if abs(x["sentiment"]) > 0.1]
    dir_w = sum(x["weight"] for x in directional)
    bull_w = sum(x["weight"] for x in directional if x["sentiment"] > 0)
    bull_pct = (bull_w / dir_w) if dir_w > 0 else 0.5

    n_sources = len({x["source"] for x in scored})
    n_factor = min(1.0, len(scored) / 30.0)
    diversity = min(1.0, n_sources / 6.0)
    agreement = (abs(sum(math.copysign(x["weight"], x["sentiment"])
                         for x in directional)) / dir_w) if dir_w > 0 else 0.0
    confidence = round(100 * (0.4 * n_factor + 0.3 * diversity + 0.3 * agreement))

    themes: Dict[str, Dict[str, Any]] = {}
    for x in scored:
        th = themes.setdefault(x["theme"], {"score": 0.0, "weight": 0.0, "n": 0})
        th["score"] += x["sentiment"] * x["weight"]
        th["weight"] += x["weight"]
        th["n"] += 1
    for th in themes.values():
        th["score"] = th["score"] / th["weight"] if th["weight"] > 0 else 0.0
        del th["weight"]

    top = sorted(scored, key=lambda x: -abs(x["sentiment"]) * x["weight"])[:5]
    return {
        "pulse": pulse,
        "bull_pct": bull_pct,
        "bear_pct": 1.0 - bull_pct,
        "confidence": confidence,
        "n_items": len(scored),
        "n_sources": n_sources,
        "themes": themes,
        "top": [{"title": x["title"], "source": x["source"],
                 "sentiment": x["sentiment"], "theme": x["theme"]} for x in top],
    }
