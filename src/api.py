"""
FastAPI server exposing the options screener as a local HTTP API.

Start with:
    uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

import os
import sys
import math
import time
import logging
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

# ── Resolve project root so load_config("config.json") finds the file ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# ── Internal imports (after chdir so relative paths resolve) ───────────────
from .options_screener import run_scan, load_config, setup_logging
from .data_fetching import get_market_context, get_vix_level, determine_vix_regime

logger = logging.getLogger("api")
app = FastAPI(title="Options Screener API", version="1.0.0")
_executor = ThreadPoolExecutor(max_workers=2)

# ── Market context TTL cache (60s) ─────────────────────────────────────────
_market_cache: Dict[str, Any] = {}
_market_cache_ts: float = 0.0
_MARKET_TTL = 60.0


# ── Serialization helpers ──────────────────────────────────────────────────

def _clean_value(v: Any) -> Any:
    """Recursively convert numpy scalars, NaN/inf, and Timestamps to JSON-safe types."""
    # Lazy import numpy to avoid hard dep at module load
    try:
        import numpy as np
        import pandas as pd
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                return None
            return fv
        if isinstance(v, np.ndarray):
            return [_clean_value(x) for x in v.tolist()]
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
    except ImportError:
        pass
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _clean_value(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_clean_value(x) for x in v]
    return v


def _serialize_picks(picks_df, n: int = 10) -> List[Dict]:
    """Return top-N picks as a list of clean dicts, sorted by quality_score desc."""
    if picks_df is None or picks_df.empty:
        return []

    df = picks_df.copy()

    # Ensure dte column exists
    if "dte" not in df.columns:
        if "T_years" in df.columns:
            df["dte"] = (df["T_years"] * 365).round(0).astype(int)
        else:
            df["dte"] = None

    # Sort and take top N
    if "quality_score" in df.columns:
        df = df.sort_values("quality_score", ascending=False)
    df = df.head(n)

    wanted = [
        "symbol", "type", "strike", "expiration", "dte",
        "premium", "prob_profit", "quality_score", "delta",
        "ev_per_contract", "premium_bucket",
    ]
    cols = [c for c in wanted if c in df.columns]

    rows = []
    for _, row in df[cols].iterrows():
        rows.append({c: _clean_value(row[c]) for c in cols})
    return rows


def _build_market_response(trend: str, regime: str, macro_risk: bool, tnx_change: float) -> Dict:
    vix = None
    vix_regime = None
    try:
        vix = get_vix_level()
        vix_regime = determine_vix_regime(vix)
    except Exception:
        pass

    return _clean_value({
        "market_trend": trend,
        "volatility_regime": regime,
        "macro_risk_active": macro_risk,
        "tnx_change_pct": tnx_change,
        "vix_level": vix,
        "vix_regime": vix_regime,
    })


def _default_scan_args(config: Dict) -> Dict:
    filters = config.get("filters", {})
    return dict(
        mode="Standard scan",
        budget=config.get("default_budget", 750),
        max_expiries=config.get("max_expirations", 4),
        min_dte=filters.get("min_days_to_expiration", 7),
        max_dte=filters.get("max_days_to_expiration", 45),
        trader_profile="neutral",
        verbose=False,
        compact=True,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/market")
async def market_endpoint():
    """Return current market context: VIX level/regime, trend, macro risk."""
    global _market_cache, _market_cache_ts
    now = time.monotonic()
    if now - _market_cache_ts < _MARKET_TTL and _market_cache:
        return JSONResponse(_market_cache)

    import asyncio
    loop = asyncio.get_event_loop()
    trend, regime, macro_risk, tnx_change = await loop.run_in_executor(
        _executor, get_market_context
    )
    result = _build_market_response(trend, regime, macro_risk, tnx_change)
    _market_cache = result
    _market_cache_ts = now
    return JSONResponse(result)


@app.get("/top")
async def top_endpoint(n: int = Query(default=10, ge=1, le=50)):
    """Return top N picks from the liquid_large_cap watchlist."""
    import asyncio

    config = load_config("config.json")
    tickers = config.get("watchlists", {}).get("liquid_large_cap", [])
    if not tickers:
        raise HTTPException(status_code=500, detail="liquid_large_cap watchlist is empty")

    args = _default_scan_args(config)
    scan_logger = setup_logging()

    loop = asyncio.get_event_loop()
    trend, regime, macro_risk, tnx_change = await loop.run_in_executor(
        _executor, get_market_context
    )

    def _run():
        return run_scan(
            tickers=tickers,
            logger=scan_logger,
            market_trend=trend,
            volatility_regime=regime,
            macro_risk_active=macro_risk,
            tnx_change_pct=tnx_change,
            **args,
        )

    result = await loop.run_in_executor(_executor, _run)
    picks = _serialize_picks(result.picks, n=n)
    return JSONResponse({
        "watchlist": "liquid_large_cap",
        "count": len(picks),
        "picks": picks,
    })


@app.get("/scan/{symbol}")
async def scan_symbol(symbol: str):
    """Scan a single ticker and return its top picks."""
    import asyncio

    symbol = symbol.upper().strip()
    config = load_config("config.json")
    args = _default_scan_args(config)
    scan_logger = setup_logging()

    loop = asyncio.get_event_loop()
    trend, regime, macro_risk, tnx_change = await loop.run_in_executor(
        _executor, get_market_context
    )

    def _run():
        return run_scan(
            tickers=[symbol],
            logger=scan_logger,
            market_trend=trend,
            volatility_regime=regime,
            macro_risk_active=macro_risk,
            tnx_change_pct=tnx_change,
            **args,
        )

    try:
        result = await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    picks = _serialize_picks(result.picks, n=10)
    return JSONResponse({
        "symbol": symbol,
        "count": len(picks),
        "picks": picks,
        "market_context": _clean_value({
            k: v for k, v in result.market_context.items()
            if k != "sector_ctx"
        }),
    })


@app.get("/watchlist/{name}")
async def scan_watchlist(name: str, n: int = Query(default=10, ge=1, le=50)):
    """Scan any named watchlist from config.json."""
    import asyncio

    config = load_config("config.json")
    watchlists = config.get("watchlists", {})
    if name not in watchlists:
        raise HTTPException(
            status_code=404,
            detail=f"Watchlist '{name}' not found. Available: {list(watchlists.keys())}",
        )

    tickers = watchlists[name]
    args = _default_scan_args(config)
    scan_logger = setup_logging()

    loop = asyncio.get_event_loop()
    trend, regime, macro_risk, tnx_change = await loop.run_in_executor(
        _executor, get_market_context
    )

    def _run():
        return run_scan(
            tickers=tickers,
            logger=scan_logger,
            market_trend=trend,
            volatility_regime=regime,
            macro_risk_active=macro_risk,
            tnx_change_pct=tnx_change,
            **args,
        )

    try:
        result = await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    picks = _serialize_picks(result.picks, n=n)
    return JSONResponse({
        "watchlist": name,
        "tickers_scanned": len(tickers),
        "count": len(picks),
        "picks": picks,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
