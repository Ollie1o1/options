"""Sector Relative Strength analyzer.

Fetches 3 months of daily closes for all 11 SPDR sector ETFs + SPY via a
single yf.download batch call, compares sector and SPY wealth growth over
matching 20-session windows, and returns a SectorContext dataclass.

15-minute in-process TTL cache — no SQLite persistence required.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ALL_SECTOR_ETFS: List[str] = [
    "XLK", "XLF", "XLE", "XLY", "XLP",
    "XLV", "XLI", "XLB", "XLU", "XLRE", "XLC",
]

_CACHE_TTL_SECONDS = 900  # 15 minutes


@dataclass
class SectorContext:
    rs_matrix: Dict[str, float] = field(default_factory=dict)
    top_sectors: List[str] = field(default_factory=list)
    bottom_sectors: List[str] = field(default_factory=list)
    mean_reversion_setups: List[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)


def _roc_20(series: pd.Series) -> Optional[float]:
    """20-day Rate of Change: (price[-1] / price[-21]) - 1."""
    if len(series) < 21:
        return None
    try:
        denom = series.iloc[-21]
        window = series.iloc[-21:]
        if denom <= 0 or window.isna().any() or (window <= 0).any() or not window.map(math.isfinite).all():
            return None
        return float(series.iloc[-1] / denom) - 1.0
    except Exception:
        return None


def _empty_context() -> SectorContext:
    """Safe fallback when data is unavailable."""
    return SectorContext(
        rs_matrix={},
        top_sectors=[],
        bottom_sectors=[],
        mean_reversion_setups=[],
        fetched_at=time.time(),
    )


class SectorAnalyzer:
    """Computes sector Relative Strength vs SPY using 20-day ROC."""

    def __init__(self) -> None:
        self._cache: Optional[SectorContext] = None

    def get_sector_context(self) -> SectorContext:
        """Return cached SectorContext, refreshing if TTL expired."""
        now = time.time()
        if self._cache is not None and (now - self._cache.fetched_at) < _CACHE_TTL_SECONDS:
            return self._cache
        try:
            ctx = self._fetch_and_compute()
        except Exception as exc:
            logger.warning("SectorAnalyzer._fetch_and_compute failed: %s", exc)
            ctx = _empty_context()
        self._cache = ctx
        return ctx

    def _fetch_and_compute(self) -> SectorContext:
        """Download sector + SPY data and compute RS ratios."""
        import yfinance as yf

        symbols = ALL_SECTOR_ETFS + ["SPY"]
        raw = yf.download(
            symbols,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        # yf.download returns MultiIndex columns (field, ticker) when >1 symbol
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]]

        if closes.empty:
            return _empty_context()

        # Drop columns that are entirely NaN
        closes = closes.dropna(axis=1, how="all")

        if "SPY" not in closes.columns:
            logger.warning("SectorAnalyzer: SPY data missing; returning empty context")
            return _empty_context()

        spy_series = closes["SPY"].dropna()
        spy_roc = _roc_20(spy_series)
        if spy_roc is None or spy_roc <= -1:
            return _empty_context()

        rs_matrix: Dict[str, float] = {}
        sma50: Dict[str, float] = {}

        for etf in ALL_SECTOR_ETFS:
            if etf not in closes.columns:
                continue
            # Same session endpoints, with no dropping a hole and quietly
            # turning 20 sessions into a different holding window.
            series = closes[etf].reindex(spy_series.index[-21:])
            if len(series) < 21:
                continue
            etf_roc = _roc_20(series)
            if etf_roc is None:
                continue
            # Ratio of gross returns, not ratio of signed returns. Dividing
            # by a negative SPY ROC used to call the worst decliners leaders;
            # dividing by a near-zero ROC made the result explode or vanish.
            rs_matrix[etf] = round((1.0 + etf_roc) / (1.0 + spy_roc), 4)

            # 50-day SMA for mean-reversion detection
            full_series = closes[etf].reindex(spy_series.index)
            if len(full_series) >= 50 and full_series.iloc[-50:].notna().all():
                sma50[etf] = float(full_series.iloc[-50:].mean())

        if not rs_matrix:
            return _empty_context()

        # Sort by RS ratio descending
        sorted_etfs = sorted(rs_matrix, key=lambda e: rs_matrix[e], reverse=True)
        top_sectors = sorted_etfs[:3]
        bottom_sectors = sorted_etfs[-3:]

        # Mean-reversion setups: RS > 1.0 AND price within 2% of 50-day SMA
        mean_reversion: List[str] = []
        for etf in rs_matrix:
            if rs_matrix[etf] <= 1.0:
                continue
            if etf not in sma50 or etf not in closes.columns:
                continue
            last_price = float(closes[etf].dropna().iloc[-1])
            sma = sma50[etf]
            if sma > 0 and abs(last_price / sma - 1.0) <= 0.02:
                mean_reversion.append(etf)

        return SectorContext(
            rs_matrix=rs_matrix,
            top_sectors=top_sectors,
            bottom_sectors=bottom_sectors,
            mean_reversion_setups=mean_reversion,
            fetched_at=time.time(),
        )
