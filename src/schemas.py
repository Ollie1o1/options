"""Shared schema definitions for the options screener pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import pandas as pd


@dataclass
class ScanResult:
    """Container for a scan's output frames.

    ``picks`` carries the per-contract data-quality / provenance columns added
    by the pipeline (these are part of the contract even though the dataclass
    only types the frame):
      - ``quote_source``    str  — "yfinance", "yfinance+synthetic_spread", "yahooquery"
      - ``quote_as_of``     str  — UTC ISO timestamp of the contract's last print (or NA)
      - ``quote_age_min``   float — minutes between quote_as_of and the fetch (or NaN)
      - ``quote_freshness`` str  — "fresh" | "delayed" | "stale" | "unknown"
      - ``iv_solved``       float — IV solved from the mid price via Brent (Phase 2)
      - ``iv_residual_pct`` float — (yahoo_iv - solved_iv) / solved_iv (Phase 2)
      - ``iv_verified``     bool/None — |residual| <= 15% (None when unsolvable) (Phase 2)
    """
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)
    spreads: pd.DataFrame = field(default_factory=pd.DataFrame)
    credit_spreads: pd.DataFrame = field(default_factory=pd.DataFrame)
    iron_condors: pd.DataFrame = field(default_factory=pd.DataFrame)
    ticker_contexts: Dict[str, dict] = field(default_factory=dict)
    market_context: Dict[str, Any] = field(default_factory=dict)
    top_pick: Optional[Any] = None
    underlying_price: float = 0.0
    rfr: float = 0.045
    chain_iv_median: float = 0.0
    timestamp: str = ""
