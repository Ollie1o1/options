"""Crypto screener — interactive entry point for mode [2].

Sub-menu:
  [1] BTC OPTIONS DISCOVER       — scan BTC chain, rank by quality
  [2] ETH OPTIONS DISCOVER       — same for ETH
  [3] CREDIT SPREADS (high IV)   — verticals with elevated IV rank
  [4] FUNDING / BASIS MONITOR    — perp signals only, no options
  [5] PORTFOLIO                  — view paper_trades_crypto.db positions
  [6] CALIBRATION STATUS         — IC analysis on closed crypto trades
  [Q] BACK                       — return to top-level menu
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
from typing import Optional

import pandas as pd

from . import data_fetching as _df
from . import regime as _regime
from . import scoring as _scoring

# Reuse the equity formatter for visual consistency. Falls back gracefully if
# the optional rich/colorama deps are missing.
try:
    from src import formatting as fmt
    HAS_FMT = True
except ImportError:
    fmt = None  # type: ignore
    HAS_FMT = False


_CRYPTO_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper_trades_crypto.db",
)


def _banner(text: str) -> None:
    bar = "═" * 80
    if HAS_FMT and fmt:
        print(fmt.colorize(bar, fmt.Colors.BRIGHT_CYAN))
        print(fmt.colorize(f"  {text}", fmt.Colors.BRIGHT_CYAN, bold=True))
        print(fmt.colorize(bar, fmt.Colors.BRIGHT_CYAN))
    else:
        print(bar)
        print(f"  {text}")
        print(bar)


def _color(text: str, color_attr: str, bold: bool = False) -> str:
    if HAS_FMT and fmt:
        return fmt.colorize(text, getattr(fmt.Colors, color_attr), bold=bold)
    return text


def _print_regime(regime: Optional[_regime.Regime]) -> None:
    if regime is None:
        print("  Regime: UNKNOWN — insufficient history")
        return
    color_map = {"bull": "BRIGHT_GREEN", "chop": "BRIGHT_YELLOW", "bear": "BRIGHT_RED"}
    label_str = _color(f"REGIME: {regime}", color_map.get(regime.label, "WHITE"), bold=True)
    print(f"  {label_str}")


def _scan_currency(currency: str, top_n: int = 15) -> Optional[pd.DataFrame]:
    """Fetch + score chain for BTC or ETH. Returns ranked top_n rows or None."""
    print(f"\n  Fetching {currency} chain from Deribit...")
    chain = _df.get_options_chain(currency)
    if chain is None or chain.empty:
        print(f"  {_color('ERROR', 'BRIGHT_RED', bold=True)}: failed to fetch {currency} chain")
        return None
    print(f"  → {len(chain)} contracts loaded")

    print(f"  Fetching {currency} spot history (yfinance)...")
    history = _df.get_spot_history(currency, days=365)
    if history is None or history.empty:
        print(f"  {_color('WARN', 'BRIGHT_YELLOW')}: no spot history — VRP/IV-rank will fall back to neutral")

    print("  Fetching perp funding (Binance)...")
    fsym = f"{currency.upper()}USDT"
    funding = _df.get_funding_rate(fsym)
    funding_hist = _df.get_funding_history(fsym, limit=120)

    # Regime classification (BTC drives the regime even when scoring ETH —
    # ETH usually trends with BTC).
    btc_history = history if currency.upper() == "BTC" else _df.get_spot_history("BTC", days=365)
    regime = _regime.classify_btc(btc_history) if btc_history is not None and not btc_history.empty else None
    _print_regime(regime)

    regime_mults = _regime.REGIME_WEIGHT_MULTIPLIERS.get(
        regime.label if regime else "chop"
    )

    print("  Scoring contracts...")
    scored = _scoring.score_chain(
        chain, history, funding, funding_hist,
        regime_multipliers=regime_mults,
    )

    # Friendly filters for the discover view: limit DTE 7-60, OI > 0
    filtered = scored[(scored["dte"].between(5, 60)) & (scored["open_interest"] > 0)].copy()
    if filtered.empty:
        filtered = scored.copy()  # fallback if filters wipe everything
    return filtered.sort_values("quality_score", ascending=False).head(top_n)


def _print_chain_table(df: pd.DataFrame, currency: str) -> None:
    if df is None or df.empty:
        print("\n  No contracts to display.")
        return
    print()
    header = (
        f"  {'#':<3} {'Instrument':<28} {'Type':<5} {'Strike':>10} "
        f"{'DTE':>5} {'IV':>7} {'Bid':>10} {'Ask':>10} {'OI':>7} {'Score':>7}"
    )
    print(_color(header, "BOLD", bold=True))
    print("  " + "─" * (len(header) - 2))
    for i, (_, row) in enumerate(df.iterrows(), 1):
        score_color = "BRIGHT_GREEN" if row["quality_score"] >= 0.65 else (
            "BRIGHT_YELLOW" if row["quality_score"] >= 0.55 else "WHITE"
        )
        line = (
            f"  {i:<3} {row['instrument_name']:<28} {row['type']:<5} "
            f"{row['strike']:>10,.0f} {int(row['dte']):>5} {row['mark_iv']:>6.1%} "
            f"${row['bid_price']:>9,.0f} ${row['ask_price']:>9,.0f} "
            f"{int(row['open_interest']):>7} "
            f"{_color(f'{row.quality_score:>6.3f}', score_color, bold=True)}"
        )
        print(line)


def _funding_basis_dashboard() -> None:
    _banner("FUNDING / BASIS MONITOR")
    for sym, label in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")):
        print()
        f = _df.get_funding_rate(sym)
        if f is None:
            print(f"  {label}: funding fetch failed")
            continue
        annualized = f["funding_rate"] * 3 * 365  # 8h cycles per day × 365
        basis_bps = f["basis_pct"] * 10000
        sign = "+" if annualized > 0 else ""
        f_color = "BRIGHT_GREEN" if abs(annualized) > 0.10 else "WHITE"
        print(f"  {_color(label, 'BRIGHT_CYAN', bold=True)}")
        print(f"    Mark:   ${f['mark_price']:,.2f}")
        print(f"    Index:  ${f['index_price']:,.2f}")
        print(f"    Basis:  {basis_bps:+.1f} bps")
        print(f"    Funding (8h):  {f['funding_rate']*100:+.4f}%   "
              f"→ {_color(f'{sign}{annualized:.1%} annualized', f_color, bold=True)}")
    # Edge note
    print()
    print(_color("  NOTES", "DIM"))
    print("  • |annualized funding| > 10% sustained → cash-and-carry (long spot, short perp)")
    print("  • Negative funding for days → fear-driven crowded shorts; short-vol structures often work")
    print()


def _portfolio_view() -> None:
    """View open / closed positions in paper_trades_crypto.db."""
    _banner("CRYPTO PAPER PORTFOLIO")
    if not os.path.exists(_CRYPTO_DB_PATH):
        print(f"\n  No crypto trades logged yet — DB not created at {_CRYPTO_DB_PATH}")
        return
    try:
        from src.paper_manager import PaperManager
        pm = PaperManager(db_path=_CRYPTO_DB_PATH)
        df = pm.get_all_trades()
    except Exception as e:
        print(f"\n  Could not load portfolio: {type(e).__name__}: {e}")
        return
    if df is None or df.empty:
        print("\n  Crypto ledger is empty.")
        return
    open_n = int((df["status"] == "OPEN").sum())
    closed_n = int((df["status"] == "CLOSED").sum())
    print(f"\n  Open: {open_n}    Closed: {closed_n}")
    if closed_n > 0:
        closed = df[df["status"] == "CLOSED"]
        realized = float(closed["pnl_usd"].fillna(0).sum())
        win_rate = float((closed["pnl_usd"] > 0).mean()) if not closed["pnl_usd"].dropna().empty else 0.0
        print(f"  Realized P&L:  ${realized:+,.2f}")
        print(f"  Win rate:      {win_rate:.0%}")
    print()


def _calibration_status() -> None:
    _banner("CRYPTO CALIBRATION STATUS")
    if not os.path.exists(_CRYPTO_DB_PATH):
        print("\n  No closed trades yet. Build the ledger to ≥30 closes per structure.")
        return
    try:
        from src.paper_manager import PaperManager
        pm = PaperManager(db_path=_CRYPTO_DB_PATH)
        ic = pm.compute_ic()
    except Exception as e:
        print(f"\n  Calibration failed: {type(e).__name__}: {e}")
        return
    n = int(ic.get("n_closed", 0)) if isinstance(ic, dict) else 0
    print(f"\n  Closed trades: {n}")
    print(f"  Apply-gate: needs ≥100 closed crypto trades for IC analysis to be meaningful")
    if n > 0 and isinstance(ic, dict) and "components" in ic:
        print("\n  Per-component IC:")
        for comp, val in (ic["components"] or {}).items():
            print(f"    {comp:<20}  {val:+.3f}")
    print()


def _log_top_pick(top_pick: pd.Series, currency: str) -> None:
    """Log the top-scored contract to paper_trades_crypto.db."""
    try:
        from src.paper_manager import PaperManager
        pm = PaperManager(db_path=_CRYPTO_DB_PATH)
    except Exception as e:
        print(f"  Could not initialize PaperManager: {e}")
        return
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    entry_price = float(top_pick.get("ask_price") or top_pick.get("mark_price") or 0)
    if entry_price <= 0:
        print("  Skipped log — no valid entry price (illiquid contract).")
        return
    trade = {
        "date": today,
        "ticker": currency.upper(),
        "expiration": str(top_pick["expiration"]),
        "strike": float(top_pick["strike"]),
        "type": str(top_pick["type"]).lower(),
        "entry_price": entry_price,
        "quality_score": float(top_pick["quality_score"]),
        "strategy_name": "Long Call" if top_pick["type"] == "call" else "Long Put",
        "entry_iv": float(top_pick.get("mark_iv") or 0),
        "iv_rank_score":        float(top_pick.get("iv_rank_score") or 0),
        "vrp_score":            float(top_pick.get("vrp_score") or 0),
        "term_structure_score": float(top_pick.get("term_structure_score") or 0),
        "skew_align_score":     float(top_pick.get("skew_score") or 0),
        "weight_profile":       "crypto_baseline",
    }
    try:
        if pm.log_trade_if_new(trade):
            print(f"  ✓ Logged {trade['strategy_name']} on {currency} "
                  f"${trade['strike']:,.0f} @ ${entry_price:,.2f} "
                  f"(score {trade['quality_score']:.3f})")
        else:
            print("  Skipped — duplicate of an open paper trade today.")
    except Exception as e:
        print(f"  Log failed: {type(e).__name__}: {e}")


def _prompt(label: str, default: str = "") -> str:
    try:
        return input(f"  {label}: ").strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def main() -> None:
    while True:
        _banner("CRYPTO STRATEGIST  —  BTC / ETH options + perp basis")
        print()
        print("  [1] BTC OPTIONS DISCOVER")
        print("  [2] ETH OPTIONS DISCOVER")
        print("  [3] FUNDING / BASIS MONITOR")
        print("  [4] PORTFOLIO  (paper_trades_crypto.db)")
        print("  [5] CALIBRATION STATUS")
        print("  [Q] BACK")
        print()
        choice = _prompt("Choice", "Q").upper()
        if choice in ("Q", "QUIT", "EXIT", ""):
            return
        if choice == "1":
            ranked = _scan_currency("BTC", top_n=15)
            _print_chain_table(ranked, "BTC")
            if ranked is not None and not ranked.empty:
                if _prompt("Log top pick to crypto paper ledger? [y/N]", "n").lower() == "y":
                    _log_top_pick(ranked.iloc[0], "BTC")
        elif choice == "2":
            ranked = _scan_currency("ETH", top_n=15)
            _print_chain_table(ranked, "ETH")
            if ranked is not None and not ranked.empty:
                if _prompt("Log top pick to crypto paper ledger? [y/N]", "n").lower() == "y":
                    _log_top_pick(ranked.iloc[0], "ETH")
        elif choice == "3":
            _funding_basis_dashboard()
        elif choice == "4":
            _portfolio_view()
        elif choice == "5":
            _calibration_status()
        else:
            print(f"  Unknown choice: {choice!r}")


if __name__ == "__main__":
    main()
