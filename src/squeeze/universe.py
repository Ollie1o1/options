"""Squeeze-candidate sourcing: high-short-float optionable names via Finviz.

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
from typing import List

logger = logging.getLogger(__name__)

# Names that recur on high-short-float screens; refreshed opportunistically.
# Only a degraded-mode fallback — the live Finviz screen is the real source.
FALLBACK_TICKERS = ["NBIS", "SMCI", "LCID", "CVNA", "UPST", "IONQ", "RKLB", "SOUN"]

# URL filter string: Float Short >20%, optionable, avg vol >500K, USA.
SQUEEZE_FILTERS_F = "sh_short_o20,sh_opt_option,sh_avgvol_o500,geo_usa"

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


def get_squeeze_universe(max_tickers: int = 25) -> List[str]:
    """High-short-float optionable US names, most-active first."""
    try:
        tickers = finviz_tickers(SQUEEZE_FILTERS_F, order="-averagevolume",
                                 limit=max_tickers)
        if not tickers:
            logger.warning("Finviz squeeze screen returned empty; using fallback list")
            return FALLBACK_TICKERS[:max_tickers]
        return tickers
    except Exception as exc:
        logger.warning("Finviz squeeze screen failed (%s); using fallback list", exc)
        return FALLBACK_TICKERS[:max_tickers]
