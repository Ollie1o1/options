"""Shared type definitions for the options screener pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import pandas as pd


@dataclass
class ScanResult:
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
