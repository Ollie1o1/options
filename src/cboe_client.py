"""Free CBOE delayed-quote options chains — a second source beside Yahoo.

CBOE publishes full delayed chains (bid/ask with sizes, exchange-computed IV,
all Greeks, OI, volume, last trade time, spot) as unauthenticated JSON:

    https://cdn.cboe.com/api/global/delayed_quotes/options/<SYMBOL>.json

That field coverage is *better* than yfinance (which has no reliable Greeks and
unreliable IV on illiquid strikes), which makes this the cheapest possible
cross-source verification: $0 and no API key. Used by ``src.cross_check`` (on-
demand per-ticker comparison) and ``src.chain_archive`` (daily snapshots).

Quotes are delayed ~15 minutes — same class of freshness as Yahoo; treat
agreement between the two as confirmation, not as a realtime quote.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"

# OCC option symbol: ROOT (1-6 alnum) + YYMMDD + C/P + strike*1000 (8 digits)
_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def parse_occ_symbol(occ: str) -> Optional[Tuple[str, str, str, float]]:
    """'AAPL261016C00335000' -> ('AAPL', '2026-10-16', 'call', 335.0)."""
    m = _OCC_RE.match(occ or "")
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    expiration = f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}"
    return root, expiration, ("call" if cp == "C" else "put"), int(strike) / 1000.0


def parse_chain(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize a CBOE payload into archive-schema rows. Never raises;
    malformed contracts are skipped."""
    data = (payload or {}).get("data") or {}
    spot = data.get("current_price")
    snap = (payload or {}).get("timestamp")
    rows: List[Dict[str, Any]] = []
    for o in data.get("options") or []:
        parsed = parse_occ_symbol(o.get("option") or "")
        if not parsed:
            continue
        symbol, expiration, opt_type, strike = parsed
        rows.append({
            "symbol": symbol,
            "contract": o.get("option"),
            "type": opt_type,
            "strike": strike,
            "expiration": expiration,
            "bid": o.get("bid"),
            "ask": o.get("ask"),
            "bid_size": o.get("bid_size"),
            "ask_size": o.get("ask_size"),
            "iv": o.get("iv"),
            "delta": o.get("delta"),
            "gamma": o.get("gamma"),
            "theta": o.get("theta"),
            "vega": o.get("vega"),
            "rho": o.get("rho"),
            "open_interest": o.get("open_interest"),
            "volume": o.get("volume"),
            "last_trade_time": o.get("last_trade_time"),
            "spot": spot,
            "snapshot_ts": snap,
            "source": "cboe",
        })
    return rows


def _http_get_json(url: str, timeout: int) -> Dict[str, Any]:
    import requests
    resp = requests.get(url, timeout=timeout,
                        headers={"User-Agent": "options-screener/1.0"})
    resp.raise_for_status()
    return resp.json()


def fetch_chain(symbol: str, timeout: int = 20,
                getter: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Fetch + parse the delayed chain for `symbol`. In-session cached.
    Returns [] on any failure — callers treat CBOE as best-effort."""
    sym = (symbol or "").upper()
    if sym in _CACHE:
        return _CACHE[sym]
    getter = getter or _http_get_json
    try:
        payload = getter(BASE_URL.format(symbol=sym), timeout)
        rows = parse_chain(payload)
    except Exception:
        return []
    _CACHE[sym] = rows
    return rows


def clear_cache() -> None:
    _CACHE.clear()


if __name__ == "__main__":
    import sys
    rows = fetch_chain(sys.argv[1] if len(sys.argv) > 1 else "SPY")
    print(json.dumps({"contracts": len(rows), "sample": rows[:2]}, indent=1))
