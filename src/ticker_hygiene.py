"""Ticker hygiene — find and wipe dead / renamed / non-tradeable symbols.

Validates each ticker against the live data provider (a symbol is LIVE if it
returns recent price rows). Catches:
  - non-tradeable junk (e.g. SPACEX — private, no ticker; 404s)
  - renamed/old symbols that now fail (e.g. SQ after Block became XYZ)
  - delisted names

Scope = WATCHLISTS (the forward-looking ticker lists in config.json). It does
NOT touch paper_trades.db: closed trades are real P&L history and stay even if
the symbol was later renamed (the XYZ trade is valid history). Open positions
are audited and reported, never auto-deleted — closing them is a trading
decision, not a data cleanup.

CLI:
    python -m src.ticker_hygiene --audit          # dry-run: list LIVE vs DEAD
    python -m src.ticker_hygiene --clean           # remove DEAD from config watchlists
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


def is_live(ticker: str, period: str = "5d") -> bool:
    """True if the symbol returns recent price rows from the data provider.
    A dead/renamed/non-existent symbol (SQ, SPACEX, …) returns 0 rows → False."""
    import warnings
    try:
        import yfinance as yf
    except ImportError:
        return True   # can't validate without the provider; don't falsely flag
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = yf.Ticker(ticker).history(period=period)
        return h is not None and len(h) > 0
    except Exception:
        return False


def audit_tickers(tickers: Iterable[str], live_fn=is_live) -> Dict[str, List[str]]:
    """Partition tickers into live/dead, de-duplicated and upper-cased."""
    seen, live, dead = set(), [], []
    for t in tickers:
        u = str(t or "").upper().strip()
        if not u or u in seen:
            continue
        seen.add(u)
        (live if live_fn(u) else dead).append(u)
    return {"live": sorted(live), "dead": sorted(dead)}


def watchlist_tickers(config: Dict[str, Any]) -> List[str]:
    """All tickers across every config watchlist, de-duplicated."""
    out: set = set()
    for lst in (config.get("watchlists") or {}).values():
        for t in lst or []:
            out.add(str(t).upper().strip())
    return sorted(out)


def clean_watchlists(config_path: str = "config.json", dry_run: bool = True,
                     live_fn=is_live) -> Dict[str, Any]:
    """Validate every watchlist ticker; drop the dead ones. dry_run=True only
    reports. Returns {dead, removed_by_list, audited}. Writes config only when
    dry_run is False AND dead tickers were found (never rewrites needlessly)."""
    with open(config_path) as f:
        config = json.load(f)
    all_tk = watchlist_tickers(config)
    audit = audit_tickers(all_tk, live_fn=live_fn)
    dead = set(audit["dead"])
    removed: Dict[str, List[str]] = {}
    if dead and not dry_run:
        for name, lst in (config.get("watchlists") or {}).items():
            kept = [t for t in (lst or []) if str(t).upper().strip() not in dead]
            dropped = [t for t in (lst or []) if str(t).upper().strip() in dead]
            if dropped:
                removed[name] = dropped
                config["watchlists"][name] = kept
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    return {"audited": len(all_tk), "live": audit["live"], "dead": audit["dead"],
            "removed_by_list": removed, "dry_run": dry_run}


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Find/wipe dead, renamed, non-tradeable tickers")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--clean", action="store_true", help="Remove DEAD tickers from config watchlists")
    ap.add_argument("--extra", default="", help="Comma-separated extra tickers to audit (e.g. open positions)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    tickers = watchlist_tickers(config)
    if args.extra:
        tickers = sorted(set(tickers) | {t.strip().upper() for t in args.extra.split(",") if t.strip()})
    print(f"Auditing {len(tickers)} tickers against the live data provider...")
    audit = audit_tickers(tickers)
    print(f"\nLIVE ({len(audit['live'])}): {', '.join(audit['live'])}")
    if audit["dead"]:
        print(f"\nDEAD / RENAMED / NON-TRADEABLE ({len(audit['dead'])}): {', '.join(audit['dead'])}")
    else:
        print("\nDEAD: none — every ticker is currently tradeable.")

    if args.clean:
        res = clean_watchlists(args.config, dry_run=False)
        if res["removed_by_list"]:
            print("\nRemoved from config watchlists:")
            for name, dropped in res["removed_by_list"].items():
                print(f"  {name}: {dropped}")
        else:
            print("\nNothing to remove from watchlists.")
    elif audit["dead"]:
        print("\n(dry-run) Re-run with --clean to remove the DEAD tickers from config watchlists.")


if __name__ == "__main__":
    _cli()
