"""Sponsor name (ClinicalTrials.gov) → ticker (SEC), by exact normalised match.

MEASURED 2026-08-25 over all 599 Ph3 trials in a six-month window: 162 trials
(27.0%) resolved, from 68 of 343 unique sponsors (19.8%). That is the real
ceiling, not a defect. Sampled misses are correct rejections — Akeso is
HK-listed, Actinogen is ASX, ALK-Abello is Danish, Menarini is private, Acerta
and Actelion are subsidiaries of AZN and JNJ. Two apparent false negatives,
APLS and ADVM, are both delisted.

THERE IS NO FUZZY MATCHING HERE, DELIBERATELY. A wrong ticker attaches a real
company to someone else's clinical trial — far worse than a dropped row, and
undetectable downstream. Subsidiaries are mapped by hand, one deliberate entry
at a time, in data/sponsor_aliases.json.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, Optional, Set

from src.paths import repo_path

CACHE = repo_path(os.path.join("data", "edgar_company_names.json"))
CACHE_DAYS = 30

# Curated knowledge, so it lives in tracked `configs/` rather than in `data/`,
# which is gitignored — a fresh clone must not silently lose these mappings.
# The 30-day SEC name cache above is derived state and stays in data/.
ALIASES = repo_path(os.path.join("configs", "sponsor_aliases.json"))

_SUFFIXES = (r"(inc|incorporated|ltd|limited|corp|corporation|plc|llc|lp|co|"
             r"company|gmbh|ag|sa|nv|bv|as|ab|oy|kk|pty|aps|srl|spa|"
             r"holdings|holding|group)")
_SUFFIX_RE = re.compile(rf"\s+{_SUFFIXES}$")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Lowercase, depunctuate, and strip trailing corporate suffixes.

    Suffixes are stripped repeatedly: "Foo Holdings Group Inc." must reach
    "foo", and one pass would leave "foo holdings group"."""
    text = (name or "").lower().replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    prev = None
    while text != prev:
        prev = text
        text = _SUFFIX_RE.sub("", text).strip()
    return text


def build_index(titles_to_tickers: Dict[str, str]) -> Dict[str, str]:
    """Normalised SEC title → ticker.

    A normalised name claimed by two different tickers is DROPPED rather than
    resolved to whichever sorted first — an ambiguous match is not a match."""
    counts: Dict[str, Set[str]] = {}
    for title, ticker in titles_to_tickers.items():
        counts.setdefault(normalize(title), set()).add(ticker)
    return {name: next(iter(t)) for name, t in counts.items() if len(t) == 1}


def _fetch_sec_titles() -> Dict[str, str]:
    from src.insider.edgar import _get
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["title"]: v["ticker"].upper() for v in data.values()}


def name_index(cache_path: Optional[str] = None) -> Dict[str, str]:
    """Normalised-name → ticker, disk-cached 30 days. {} on any failure."""
    path = cache_path or CACHE
    try:
        if (time.time() - os.path.getmtime(path)) / 86400.0 <= CACHE_DAYS:
            with open(path) as f:
                return dict(json.load(f))
    except (OSError, ValueError):
        pass
    try:
        index = build_index(_fetch_sec_titles())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(index, f)
        return index
    except Exception:
        return {}


def load_aliases(path: Optional[str] = None) -> Dict[str, str]:
    """Hand-maintained sponsor → ticker overrides, keys normalised on load."""
    try:
        with open(path or ALIASES) as f:
            raw = json.load(f)
        return {normalize(k): str(v).upper() for k, v in (raw or {}).items()}
    except (OSError, ValueError):
        return {}


def resolve(sponsor: str, index: Dict[str, str],
            aliases: Dict[str, str]) -> Optional[str]:
    """Ticker for a sponsor, or None. Aliases win over the SEC index."""
    key = normalize(sponsor)
    if not key:
        return None
    return aliases.get(key) or index.get(key)
