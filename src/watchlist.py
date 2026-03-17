"""Watchlist persistence — load/save ticker watchlist to/from JSON."""

import json

try:
    from . import formatting as fmt
    HAS_ENHANCED_CLI = True
except ImportError:
    HAS_ENHANCED_CLI = False

_WATCHLIST_PATH = "watchlist.json"


def load_watchlist() -> list:
    """Load personal watchlist from JSON file."""
    try:
        with open(_WATCHLIST_PATH, "r") as f:
            data = json.load(f)
            return [t.upper() for t in data if isinstance(t, str)]
    except Exception:
        return []


def save_watchlist(tickers: list) -> None:
    """Save personal watchlist to JSON file."""
    with open(_WATCHLIST_PATH, "w") as f:
        json.dump(tickers, f)


def add_to_watchlist(ticker: str) -> None:
    """Add ticker to watchlist (deduplicates)."""
    wl = load_watchlist()
    ticker = ticker.upper()
    if ticker not in wl:
        wl.append(ticker)
        save_watchlist(wl)
        msg = f"Added {ticker} to watchlist ({len(wl)} ticker(s) total)."
        print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
    else:
        msg = f"{ticker} is already in your watchlist."
        print(fmt.colorize(f"  {msg}", fmt.Colors.DIM) if HAS_ENHANCED_CLI else f"  {msg}")


def remove_from_watchlist(ticker: str) -> None:
    """Remove ticker from watchlist."""
    wl = load_watchlist()
    ticker = ticker.upper()
    if ticker in wl:
        wl.remove(ticker)
        save_watchlist(wl)
        msg = f"Removed {ticker} from watchlist ({len(wl)} ticker(s) remaining)."
        print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
    else:
        msg = f"{ticker} not found in watchlist."
        print(fmt.colorize(f"  {msg}", fmt.Colors.DIM) if HAS_ENHANCED_CLI else f"  {msg}")
