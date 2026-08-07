"""Squeeze-candidate sourcing: high-short-float optionable names via Finviz.

Two screens, ranked by short interest. The preferred one adds a +10% week to
the short-float floor, because that pair is what ``docs/SQUEEZE_BACKTEST.md``
actually measured: top-5% SI *with* upward momentum hit +20% within 42 days
50.5% of the time against a 22.5% base rate, and it fires on ~16 names per
settlement date. It can legitimately return few names or none on a quiet week,
so the plain short-float screen fills the remaining slots.

Ordering is ``-shortinterestshare``, not average volume: SI deciles are
monotone in that study (top 1%, SI >=32%, hits +20% 41.7% of the time vs 39.0%
for top 5%), while liquidity carries no measured signal — it belongs in the
filters, where >500K avg volume and optionable already sit, not in the ranking.

Tickers are parsed from the row anchors' ``stock?t=`` hrefs, NOT from cell
text: Finviz's ticker cell carries a letter-icon anchor before the ticker
link, so bs4 ``.text`` concatenation duplicates the first letter (ABEO →
"AABEO") — the bug that breaks finvizfinance 1.3.0's own DataFrame. The href
is unaffected by presentation changes.

A failed fetch degrades to a small hardcoded high-SI list instead of raising
(same contract as ``data_fetching.get_dynamic_tickers``), so the SQUEEZE mode
always has a universe to scan.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Names that recur on high-short-float screens; refreshed opportunistically.
# Only a degraded-mode fallback — the live Finviz screen is the real source.
FALLBACK_TICKERS = ["NBIS", "SMCI", "LCID", "CVNA", "UPST", "IONQ", "RKLB", "SOUN"]

# URL filter string: Float Short >20%, optionable, avg vol >500K, USA.
SQUEEZE_FILTERS_F = "sh_short_o20,sh_opt_option,sh_avgvol_o500,geo_usa"

# The same screen plus Finviz "Performance (Week) +10%" — the measured cohort.
SQUEEZE_FILTERS_MOMENTUM_F = SQUEEZE_FILTERS_F + ",ta_perf_1w10o"

# Rank by short interest as a share of float, never by liquidity.
SQUEEZE_ORDER = "-shortinterestshare"

_TICKER_HREF_RE = re.compile(r"[?&]t=([A-Za-z0-9.\-]+)")
_PAGE_SIZE = 20  # finviz screener rows per page


def _extract_tickers(soup) -> List[str]:
    """Tickers from screener-table row hrefs, order preserved, deduped."""
    table = soup.find("table", class_="screener_table")
    if table is None:
        return []
    out: List[str] = []
    for a in table.find_all("a", class_="tab-link"):
        m = _TICKER_HREF_RE.search(a.get("href") or "")
        if m:
            t = m.group(1).upper()
            if t not in out:
                out.append(t)
    return out


def finviz_tickers(f_params: str, order: str = "-averagevolume",
                   limit: int = 25) -> List[str]:
    """Ticker list from the Finviz screener, href-parsed, paginated."""
    from finvizfinance.util import web_scrap

    tickers: List[str] = []
    offset = 1
    while len(tickers) < limit:
        params = {"v": 141, "f": f_params, "o": order, "r": offset}
        soup = web_scrap("https://finviz.com/screener.ashx", params)
        page = _extract_tickers(soup)
        if not page:
            break
        tickers.extend(t for t in page if t not in tickers)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return tickers[:limit]


@dataclass
class SqueezeUniverse:
    """A sourced candidate list plus where each part of it came from."""

    tickers: List[str] = field(default_factory=list)
    momentum: List[str] = field(default_factory=list)  # cleared the +10% week
    source: str = "finviz"                             # "finviz" | "fallback"


def _screen(f_params: str, limit: int, label: str) -> List[str]:
    """One Finviz screen, SI-ranked; a failure is logged, never raised."""
    if limit <= 0:
        return []
    try:
        return finviz_tickers(f_params, order=SQUEEZE_ORDER, limit=limit)
    except Exception as exc:
        logger.warning("Finviz %s screen failed (%s)", label, exc)
        return []


def get_squeeze_universe_detailed(max_tickers: int = 25) -> SqueezeUniverse:
    """High-short-float optionable US names, momentum cohort first.

    Names clearing both the short-float floor and a +10% week lead the list;
    the plain short-float screen fills whatever slots are left, so a quiet
    week degrades to the old universe rather than to an empty scan.
    """
    momentum = _screen(SQUEEZE_FILTERS_MOMENTUM_F, max_tickers, "squeeze momentum")
    tickers = momentum[:max_tickers]

    # The momentum screen is a subset of the base filter set, so the base
    # screen hands back those same names — ask it for the shortfall PLUS the
    # overlap, or dedup silently returns a scan smaller than max_tickers.
    if len(tickers) < max_tickers:
        fill = _screen(SQUEEZE_FILTERS_F, max_tickers + len(tickers), "squeeze")
        for ticker in fill:
            if len(tickers) >= max_tickers:
                break
            if ticker not in tickers:
                tickers.append(ticker)

    if not tickers:
        logger.warning("Finviz squeeze screens returned empty; using fallback list")
        return SqueezeUniverse(tickers=FALLBACK_TICKERS[:max_tickers], source="fallback")
    return SqueezeUniverse(tickers=tickers, momentum=momentum[:len(tickers)])


def get_squeeze_universe(max_tickers: int = 25) -> List[str]:
    """Ticker list only — see ``get_squeeze_universe_detailed`` for provenance."""
    return get_squeeze_universe_detailed(max_tickers).tickers
