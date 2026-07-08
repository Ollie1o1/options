"""The tradeable crypto-perp universe and per-asset round-trip cost.

One source of truth for which perps we trade and what each costs to round-trip
(taker fee + slippage), so validation never uses a single flat cost for assets
with very different liquidity. Cost is stored in basis points; callers that
need a fraction use `cost_frac`. Overridable via config['leverage_universe'].
"""
from __future__ import annotations
from typing import Dict, List, Optional

# key -> {perp symbol, round-trip cost in bps}. BTC/ETH tightest; SOL wider.
_DEFAULT: Dict[str, dict] = {
    "BTC": {"symbol": "BTCUSDT", "cost_bps": 13.0},
    "ETH": {"symbol": "ETHUSDT", "cost_bps": 15.0},
    "SOL": {"symbol": "SOLUSDT", "cost_bps": 30.0},
}


def default_universe(config: Optional[dict] = None) -> Dict[str, dict]:
    """Merge config['leverage_universe'] over the built-in defaults."""
    u = {k: dict(v) for k, v in _DEFAULT.items()}
    block = (config or {}).get("leverage_universe") if config else None
    if isinstance(block, dict):
        for k, v in block.items():
            if isinstance(v, dict):
                u.setdefault(k, {}).update(v)
    return u


def symbols(config: Optional[dict] = None) -> List[str]:
    return list(default_universe(config).keys())


def perp_symbol(key: str, config: Optional[dict] = None) -> str:
    return default_universe(config)[key]["symbol"]


def cost_frac(key: str, config: Optional[dict] = None) -> float:
    return float(default_universe(config)[key]["cost_bps"]) / 10_000.0
