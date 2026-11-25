"""Curated liquid equity + ETF universe builder for `data/equity_ohlcv.db` expansion.

Expands the default 62-symbol OHLCV archive to ~220+ highly liquid US equities and ETFs
(DV20 >> $50M) plus all symbols in `config.json` watchlists, `longterm_plan.json`, and
`src/data_fetching.py::SECTOR_MAP` so `equity_pullback`, `vol_managed_outlook`, and
`holdings_buy_zone` scan a broad, clean, cost-surviving market without illiquid junk.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

from src.paths import repo_path

CORE_ETFS: Sequence[str] = (
    # Broad indices & factors
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "RSP", "MDY",
    # SPDR Sectors & Semis
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
    "SMH", "SOXX", "IGV", "XBI", "IBB", "KRE", "XHB", "XRT", "ITA", "XME", "XOP",
    # Commodities, Bonds & International
    "GLD", "SLV", "GDX", "USO", "UNG", "DBC", "TLT", "IEF", "SHY", "SGOV", "HYG", "LQD",
    "EFA", "EEM", "FXI", "EWJ", "EWZ", "INDA",
)

LIQUID_MEGA_LARGE_CAPS: Sequence[str] = (
    # Tech & Semis
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "CRM", "ADBE", "CSCO", "ACN",
    "IBM", "INTU", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC", "ADI", "SNPS",
    "CDNS", "NOW", "PANW", "CRWD", "FTNT", "ANET", "PLTR", "APP", "DELL", "HPE",
    "INTC", "MRVL", "NXPI", "ON", "MCHP", "SMCI", "SNOW", "DDOG", "ZS", "NET",
    "SHOP", "TEAM", "WDAY", "HUBS", "ARM", "TSM", "ASML", "RMBS", "RGTI",
    # Comm Services & Consumer Discretionary
    "GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "T", "CMCSA", "SPOT", "RBLX",
    "AMZN", "TSLA", "HD", "LOW", "MCD", "SBUX", "NKE", "BKNG", "TJX", "ROST",
    "CMG", "MAR", "HLT", "ABNB", "UBER", "DASH", "RCL", "CCL", "GM", "F",
    "LULU", "DECK", "ORLY", "AZO", "TGT", "DG", "DLTR", "RIVN", "DKNG", "CVNA",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    "V", "MA", "PYPL", "COF", "USB", "PNC", "TFC", "SPGI", "ICE", "CME",
    "MCO", "CB", "PGR", "TRV", "AIG", "MET", "PRU", "MMC", "AON", "KKR",
    "APO", "BX", "COIN", "HOOD", "SOFI", "XYZ", "FI", "FIS",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN",
    "ISRG", "SYK", "BSX", "MDT", "GILD", "VRTX", "REGN", "BMY", "CVS", "CI",
    "ELV", "HUM", "HCA", "MCK", "COR", "ZTS", "BDX", "EW", "DXCM", "IDXX",
    "MRNA", "NVO",
    # Industrials & Defense
    "GE", "CAT", "RTX", "HON", "UNP", "ETN", "DE", "BA", "LMT", "NOC",
    "GD", "UPS", "FDX", "CSX", "NSC", "WM", "RSG", "PH", "ITW", "EMR",
    "TT", "CARR", "URI", "PWR", "TDG", "GWW", "FAST", "DAL", "UAL", "AAL",
    "LUV", "GEV", "VRT",
    # Consumer Staples
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "KR", "SYY", "STZ", "KHC", "MNST", "TGT",
    # Energy, Materials, Utilities & REITs
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "OKE", "HAL", "BKR", "DVN", "FANG", "CTRA", "TRGP",
    "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "DD", "DOW", "VMC", "MLM",
    "NEE", "SO", "DUK", "CEG", "VST", "AEP", "SRE", "D", "EXC", "XEL", "PCG",
    "PLD", "AMT", "EQIX", "WELL", "SPG", "O", "PSA", "DLR", "CCI", "VICI",
)


# Known acquired/delisted tickers that may still exist in legacy SECTOR_MAP
_DELISTED_OR_ACQUIRED = frozenset({
    "MRO", "PXD", "HES", "DFS", "FISV", "ANSS", "K", "PARA", "WBA", "ATVI", "SQ", "WRK", "CTLT", "GPS",
})


def build_expanded_universe(
    *,
    config_path: str = "config.json",
    plan_path: str = "longterm_plan.json",
    extra_symbols: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return a clean, deduplicated, sorted list of ~250+ liquid stocks & ETFs."""
    syms = set(CORE_ETFS) | set(LIQUID_MEGA_LARGE_CAPS)

    # Include all watchlists & archive symbols from config.json
    full_cfg = repo_path(config_path)
    if os.path.exists(full_cfg):
        try:
            with open(full_cfg, encoding="utf-8") as fh:
                cfg = json.load(fh)
            for wl in (cfg.get("watchlists") or {}).values():
                if isinstance(wl, list):
                    syms.update(str(s).upper().strip() for s in wl if s)
            for s in (cfg.get("data_archive") or {}).get("symbols") or []:
                syms.add(str(s).upper().strip())
            for s in (cfg.get("breakout") or {}).get("universe") or []:
                syms.add(str(s).upper().strip())
        except Exception:
            pass

    # Include all tickers from longterm_plan.json so buy zones always have OHLCV coverage
    full_plan = repo_path(plan_path)
    if os.path.exists(full_plan):
        try:
            with open(full_plan, encoding="utf-8") as fh:
                plan = json.load(fh)
            for item in plan.get("names") or []:
                tk = str(item.get("ticker") or "").upper().strip()
                if tk:
                    syms.add(tk)
        except Exception:
            pass

    # Include full SECTOR_MAP tickers
    try:
        from src.data_fetching import SECTOR_MAP

        syms.update(str(k).upper().strip() for k in SECTOR_MAP.keys() if k)
    except Exception:
        pass

    if extra_symbols:
        syms.update(str(s).upper().strip() for s in extra_symbols if s)

    # Filter out obviously invalid/empty or known delisted tokens
    clean = sorted(
        s for s in syms
        if s and s not in _DELISTED_OR_ACQUIRED and all(ch.isalnum() or ch in ("-", ".") for ch in s)
    )
    return clean
