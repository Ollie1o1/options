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
    # What the reader actually SAW: post-gate, and past any per-scan budget.
    # The three frames above stay raw on purpose — the visualizer, the CSV
    # export and the --auto-log path all want the unfiltered scan, and the
    # auto-logger in particular must keep applying CONFIG's cap rather than an
    # operator's session budget.
    #
    # The post-scan [P]/[L] menu must log from THESE. With a budget set the
    # board hides what the ledger would refuse, and offering the hidden row in
    # the menu anyway re-opens the exact board-vs-ledger divergence the budget
    # exists to close — the ledger then refuses it, so nothing bad is written,
    # but the operator is offered a trade they were never shown.
    #
    # None means "this mode never built a board" (Lottery, Squeeze) and the
    # caller should fall back to the raw frame. EMPTY is not None: it means the
    # board was built and nothing survived, and the menu must offer nothing.
    board_picks: Optional[pd.DataFrame] = None
    board_credit_spreads: Optional[pd.DataFrame] = None
    board_iron_condors: Optional[pd.DataFrame] = None
    # Squeeze Hunt only: per-symbol scored calls that the generic |delta|
    # 0.15-0.35 band removes from `picks`. Display-only — the squeeze long
    # side is near-ATM by nature, so it never survives into the ranked picks.
    squeeze_calls: Dict[str, pd.DataFrame] = field(default_factory=dict)
    ticker_contexts: Dict[str, dict] = field(default_factory=dict)
    market_context: Dict[str, Any] = field(default_factory=dict)
    top_pick: Optional[Any] = None
    underlying_price: float = 0.0
    rfr: float = 0.045
    chain_iv_median: float = 0.0
    timestamp: str = ""
