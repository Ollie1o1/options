"""Crypto screener — interactive entry point for mode [2].

Strategy-aware ranking:
  1. Classify regime (bull / chop / bear) from BTC.
  2. Score the chain at chain-level (VRP, IV-rank, term, skew, funding, basis).
  3. For each recommended strategy under this regime, score every contract
     with strategy-specific moneyness/DTE/regime fit.
  4. Display top picks per strategy bucket — long premium gets single-leg picks,
     credit spreads get short/long pairs with net credit + max profit/loss.
  5. Log with correct strategy_name to paper_trades_crypto.db.
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import List, Optional

import pandas as pd

from . import data_fetching as _df
from . import regime as _regime
from . import scoring as _scoring
from . import strategy as _strategy

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


def _chain_quality_score(scored_chain: pd.DataFrame) -> float:
    """Reduce the 7 chain-level scores into a single 0..1 quality number."""
    if scored_chain.empty:
        return 0.5
    cols = ["iv_rank_score", "vrp_score", "term_structure_score", "skew_score",
            "funding_z_score", "basis_score"]
    vals = [float(scored_chain[c].iloc[0]) for c in cols if c in scored_chain.columns]
    return sum(vals) / len(vals) if vals else 0.5


def _print_chain_diagnostics(scored: pd.DataFrame) -> None:
    """One-line readout of the 7 chain-level scores so the user can see what's hot."""
    if scored.empty:
        return
    cols = [
        ("iv_rank_score",        "IV Rank"),
        ("vrp_score",            "VRP"),
        ("term_structure_score", "Term"),
        ("skew_score",           "Skew"),
        ("funding_z_score",      "Funding"),
        ("basis_score",          "Basis"),
    ]
    parts = []
    for col, label in cols:
        if col not in scored.columns:
            continue
        v = float(scored[col].iloc[0])
        col_color = "BRIGHT_GREEN" if v >= 0.65 else ("BRIGHT_YELLOW" if v >= 0.45 else "BRIGHT_RED")
        parts.append(f"{label} {_color(f'{v:.2f}', col_color, bold=True)}")
    print("  Chain signals:  " + "   ".join(parts))


def _scan_currency(currency: str) -> Optional[dict]:
    """Fetch + score chain for BTC or ETH. Returns a dict with picks per strategy."""
    print(f"\n  Fetching {currency} chain from Deribit...")
    chain = _df.get_options_chain(currency)
    if chain is None or chain.empty:
        print(f"  {_color('ERROR', 'BRIGHT_RED', bold=True)}: failed to fetch {currency} chain")
        return None
    print(f"  → {len(chain)} contracts loaded")

    print(f"  Fetching {currency} spot history (yfinance)...")
    history = _df.get_spot_history(currency, days=400)
    if history is None or history.empty:
        print(f"  {_color('WARN', 'BRIGHT_YELLOW')}: no spot history — VRP/IV-rank will fall back to neutral")

    print("  Fetching perp funding (Binance)...")
    fsym = f"{currency.upper()}USDT"
    funding = _df.get_funding_rate(fsym)
    funding_hist = _df.get_funding_history(fsym, limit=120)

    btc_history = history if currency.upper() == "BTC" else _df.get_spot_history("BTC", days=400)
    regime = _regime.classify_btc(btc_history) if btc_history is not None and not btc_history.empty else None
    _print_regime(regime)

    regime_mults = _regime.REGIME_WEIGHT_MULTIPLIERS.get(regime.label if regime else "chop")
    print("  Scoring contracts...")
    scored = _scoring.score_chain(
        chain, history, funding, funding_hist,
        regime_multipliers=regime_mults,
    )
    _print_chain_diagnostics(scored)

    chain_q = _chain_quality_score(scored)
    regime_label = regime.label if regime else "chop"

    # Filter to tradable contracts: DTE 5-60, OI > 0, non-zero spread/IV
    tradable = scored[
        (scored["dte"].between(5, 60))
        & (scored["open_interest"] > 0)
        & (scored["mark_iv"] > 0)
        & (scored["mid_price"] > 0)
    ].copy()
    if tradable.empty:
        tradable = scored.copy()

    # Per-strategy picks. Threshold raised to 0.55 — strategies below that
    # (like Iron Condor in a directional regime) shouldn't surface at all,
    # otherwise the user sees a bucket they shouldn't trade.
    picks_by_strategy: dict = {}
    for strat in _strategy.STRATEGIES:
        regime_fit = float(strat.regime_fit.get(regime_label, 0.0))
        if regime_fit < 0.55:
            continue
        if strat.direction == "spread_credit" and strat.leg_type == "both":
            # Iron Condor — pair a Bear Call and a Bull Put on the same expiry.
            df_picks = _strategy.build_iron_condor_candidates(
                tradable, regime_label, chain_q, top_n=5,
            )
        elif strat.direction == "spread_credit":
            df_picks = _strategy.build_credit_spread_candidates(
                tradable, strat, regime_label, chain_q, top_n=5,
            )
        else:
            df_picks = _strategy.score_for_strategy(
                tradable, strat, regime_label, chain_q,
            ).head(5)
        if df_picks is not None and not df_picks.empty:
            picks_by_strategy[strat.name] = df_picks

    return {
        "currency": currency,
        "regime": regime,
        "chain_quality": chain_q,
        "scored_chain": scored,
        "picks_by_strategy": picks_by_strategy,
    }


def _print_long_premium_table(df: pd.DataFrame, strategy_name: str, currency: str) -> None:
    print()
    title = f"  {_color(strategy_name, 'BRIGHT_CYAN', bold=True)}  ({currency} long premium)"
    print(title)
    header = (
        f"  {'#':<3} {'Instrument':<28} {'Strike':>10} {'DTE':>5} "
        f"{'IV':>7} {'Mid':>10} {'OI':>7} {'M.fit':>6} {'D.fit':>6} {'Score':>7}"
    )
    print(_color(header, "BOLD", bold=True))
    print("  " + "─" * (len(header) - 2))
    for i, (_, row) in enumerate(df.iterrows(), 1):
        sc = float(row["strategy_score"])
        sc_color = "BRIGHT_GREEN" if sc >= 0.55 else ("BRIGHT_YELLOW" if sc >= 0.35 else "WHITE")
        print(
            f"  {i:<3} {row['instrument_name']:<28} {row['strike']:>10,.0f} "
            f"{int(row['dte']):>5} {row['mark_iv']:>6.1%} "
            f"${row['mid_price']:>9,.0f} {int(row['open_interest']):>7} "
            f"{row['moneyness_fit']:>6.2f} {row['dte_fit']:>6.2f} "
            f"{_color(f'{sc:>6.3f}', sc_color, bold=True)}"
        )


def _print_credit_spread_table(df: pd.DataFrame, strategy_name: str, currency: str) -> None:
    print()
    title = f"  {_color(strategy_name, 'BRIGHT_CYAN', bold=True)}  ({currency} credit spread)"
    print(title)
    header = (
        f"  {'#':<3} {'Exp':>10} {'DTE':>4} {'Short':>9} {'Long':>9} "
        f"{'Width':>7} {'Credit':>9} {'MaxLoss':>9} {'R/R':>5} {'Score':>7}"
    )
    print(_color(header, "BOLD", bold=True))
    print("  " + "─" * (len(header) - 2))
    for i, (_, row) in enumerate(df.iterrows(), 1):
        sc = float(row["score"])
        sc_color = "BRIGHT_GREEN" if sc >= 0.55 else ("BRIGHT_YELLOW" if sc >= 0.35 else "WHITE")
        print(
            f"  {i:<3} {str(row['expiration']):>10} {int(row['dte']):>4} "
            f"${row['short_strike']:>8,.0f} ${row['long_strike']:>8,.0f} "
            f"${int(row['width']):>6,} ${row['net_credit']:>8,.0f} "
            f"${row['max_loss']:>8,.0f} {row['risk_reward']:>4.2f}x "
            f"{_color(f'{sc:>6.3f}', sc_color, bold=True)}"
        )


def _print_iron_condor_table(df: pd.DataFrame, currency: str) -> None:
    print()
    title = f"  {_color('Iron Condor', 'BRIGHT_CYAN', bold=True)}  ({currency} 4-leg credit, range-bound)"
    print(title)
    header = (
        f"  {'#':<3} {'Exp':>10} {'DTE':>4} {'Put long':>10} {'Put short':>10} "
        f"{'Call short':>11} {'Call long':>10} {'Credit':>9} {'MaxLoss':>9} {'R/R':>5} {'Score':>7}"
    )
    print(_color(header, "BOLD", bold=True))
    print("  " + "─" * (len(header) - 2))
    for i, (_, row) in enumerate(df.iterrows(), 1):
        sc = float(row["score"])
        sc_color = "BRIGHT_GREEN" if sc >= 0.55 else ("BRIGHT_YELLOW" if sc >= 0.35 else "WHITE")
        print(
            f"  {i:<3} {str(row['expiration']):>10} {int(row['dte']):>4} "
            f"${row['long_put_strike']:>9,.0f} ${row['short_put_strike']:>9,.0f} "
            f"${row['short_call_strike']:>10,.0f} ${row['long_call_strike']:>9,.0f} "
            f"${row['net_credit']:>8,.0f} ${row['max_loss']:>8,.0f} "
            f"{row['risk_reward']:>4.2f}x "
            f"{_color(f'{sc:>6.3f}', sc_color, bold=True)}"
        )


def _print_recommendations_banner(regime_label: str) -> None:
    recs = _strategy.recommended_strategies_for_regime(regime_label)
    if not recs:
        print()
        return
    rec_str = ", ".join(recs)
    print(f"  Recommended for {regime_label.upper()}:  {_color(rec_str, 'BRIGHT_GREEN', bold=True)}")


def _log_long_premium(row: pd.Series, currency: str) -> None:
    try:
        from src.paper_manager import PaperManager
        pm = PaperManager(db_path=_CRYPTO_DB_PATH)
    except Exception as e:
        print(f"  Could not initialize PaperManager: {e}")
        return
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    entry_price = float(row.get("ask_price") or row.get("mark_price") or 0)
    if entry_price <= 0:
        print("  Skipped log — no valid entry price (illiquid contract).")
        return
    trade = {
        "date": today,
        "ticker": currency.upper(),
        "expiration": str(row["expiration"]),
        "strike": float(row["strike"]),
        "type": str(row["type"]).lower(),
        "entry_price": entry_price,
        "quality_score": float(row.get("strategy_score") or 0),
        "strategy_name": str(row["strategy_name"]),
        "entry_iv": float(row.get("mark_iv") or 0),
        "iv_rank_score":        float(row.get("iv_rank_score") or 0),
        "vrp_score":            float(row.get("vrp_score") or 0),
        "term_structure_score": float(row.get("term_structure_score") or 0),
        "skew_align_score":     float(row.get("skew_score") or 0),
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


def _log_iron_condor(row: pd.Series, currency: str) -> None:
    try:
        from src.paper_manager import PaperManager
        pm = PaperManager(db_path=_CRYPTO_DB_PATH)
    except Exception as e:
        print(f"  Could not initialize PaperManager: {e}")
        return
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    condor = {
        "date": today,
        "ticker": currency.upper(),
        "expiration": str(row["expiration"]),
        "short_put_strike":  float(row["short_put_strike"]),
        "long_put_strike":   float(row["long_put_strike"]),
        "short_call_strike": float(row["short_call_strike"]),
        "long_call_strike":  float(row["long_call_strike"]),
        "total_credit": float(row["net_credit"]),
        "max_profit": float(row["max_profit"]),
        "max_risk":   float(row["max_loss"]),
        "quality_score": float(row["score"]),
        "weight_profile": "crypto_baseline",
    }
    try:
        if pm.log_iron_condor_if_new(condor):
            print(f"  ✓ Logged Iron Condor on {currency} "
                  f"{row['long_put_strike']:.0f}/{row['short_put_strike']:.0f}—"
                  f"{row['short_call_strike']:.0f}/{row['long_call_strike']:.0f} "
                  f"credit ${row['net_credit']:,.0f} "
                  f"(score {row['score']:.3f})")
        else:
            print("  Skipped — duplicate of an open paper trade today.")
    except Exception as e:
        print(f"  Log failed: {type(e).__name__}: {e}")


def _log_credit_spread(row: pd.Series, currency: str) -> None:
    try:
        from src.paper_manager import PaperManager
        pm = PaperManager(db_path=_CRYPTO_DB_PATH)
    except Exception as e:
        print(f"  Could not initialize PaperManager: {e}")
        return
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    strat_name = str(row["strategy_name"])
    # log_spread() takes the spread name in `type` per the equity convention
    # ("Bull Put" / "Bear Call") — it derives the underlying option type from
    # the string and writes the spread name into strategy_name.
    spread = {
        "date": today,
        "ticker": currency.upper(),
        "expiration": str(row["expiration"]),
        "short_strike": float(row["short_strike"]),
        "long_strike": float(row["long_strike"]),
        "type": strat_name,
        "net_credit": float(row["net_credit"]),
        "max_profit": float(row["max_profit"]),
        "max_loss": float(row["max_loss"]),
        "quality_score": float(row["score"]),
        "weight_profile": "crypto_baseline",
    }
    try:
        if pm.log_spread_if_new(spread):
            print(f"  ✓ Logged {strat_name} on {currency} "
                  f"${row['short_strike']:,.0f}/${row['long_strike']:,.0f} "
                  f"credit ${row['net_credit']:,.0f} "
                  f"(score {row['score']:.3f})")
        else:
            print("  Skipped — duplicate of an open paper trade today.")
    except Exception as e:
        print(f"  Log failed: {type(e).__name__}: {e}")


def _interactive_log_prompt(scan_result: dict) -> None:
    """After a scan, let the user pick a strategy bucket and log its top pick."""
    if not scan_result or not scan_result.get("picks_by_strategy"):
        return
    picks = scan_result["picks_by_strategy"]
    currency = scan_result["currency"]
    print()
    print("  Log top pick from a strategy?  Enter the strategy name or [Enter] to skip.")
    print(f"  Available: {', '.join(picks.keys())}")
    try:
        choice = input("  Strategy: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not choice:
        return
    # Case-insensitive match
    match = next((k for k in picks if k.lower() == choice.lower()), None)
    if match is None:
        print(f"  Unknown strategy: {choice!r}")
        return
    df = picks[match]
    if df.empty:
        print("  No picks for that strategy.")
        return
    top = df.iloc[0]
    if "short_call_strike" in df.columns:
        _log_iron_condor(top, currency)
    elif "short_strike" in df.columns:
        _log_credit_spread(top, currency)
    else:
        _log_long_premium(top, currency)


def _present_scan(scan: dict) -> None:
    if not scan:
        return
    regime = scan.get("regime")
    if regime:
        _print_recommendations_banner(regime.label)
    chain_q = scan.get("chain_quality", 0.5)
    cq_color = "BRIGHT_GREEN" if chain_q >= 0.65 else ("BRIGHT_YELLOW" if chain_q >= 0.45 else "BRIGHT_RED")
    print(f"  Chain quality: {_color(f'{chain_q:.2f}', cq_color, bold=True)}  "
          f"(0=poor/avoid, 0.5=neutral, 1=high-EV setup)")

    picks = scan.get("picks_by_strategy", {})
    if not picks:
        print()
        print(f"  {_color('No coherent strategies match this regime + chain.', 'BRIGHT_YELLOW')}")
        return
    currency = scan["currency"]
    for strat_name, df in picks.items():
        if "short_call_strike" in df.columns:
            _print_iron_condor_table(df, currency)
        elif "short_strike" in df.columns:
            _print_credit_spread_table(df, strat_name, currency)
        else:
            _print_long_premium_table(df, strat_name, currency)


def _funding_basis_dashboard() -> None:
    _banner("FUNDING / BASIS MONITOR")
    for sym, label in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")):
        print()
        f = _df.get_funding_rate(sym)
        if f is None:
            print(f"  {label}: funding fetch failed")
            continue
        annualized = f["funding_rate"] * 3 * 365
        basis_bps = f["basis_pct"] * 10000
        sign = "+" if annualized > 0 else ""
        f_color = "BRIGHT_GREEN" if abs(annualized) > 0.10 else "WHITE"
        print(f"  {_color(label, 'BRIGHT_CYAN', bold=True)}")
        print(f"    Mark:   ${f['mark_price']:,.2f}")
        print(f"    Index:  ${f['index_price']:,.2f}")
        print(f"    Basis:  {basis_bps:+.1f} bps")
        print(f"    Funding (8h):  {f['funding_rate']*100:+.4f}%   "
              f"→ {_color(f'{sign}{annualized:.1%} annualized', f_color, bold=True)}")
    print()
    print(_color("  NOTES", "DIM"))
    print("  • |annualized funding| > 10% sustained → cash-and-carry (long spot, short perp)")
    print("  • Negative funding for days → fear-driven crowded shorts; short-vol structures often work")
    print()


def _portfolio_view() -> None:
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
    if open_n > 0:
        opens = df[df["status"] == "OPEN"]
        print()
        print(_color("  Open positions:", "BOLD", bold=True))
        for _, r in opens.iterrows():
            print(f"    {r.get('strategy_name','?'):<14} {r['ticker']} "
                  f"${float(r['strike']):>8,.0f} {str(r['type']):<5} "
                  f"exp {r['expiration']}  entry ${float(r.get('entry_price') or 0):,.2f}")
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
            scan = _scan_currency("BTC")
            if scan:
                _present_scan(scan)
                _interactive_log_prompt(scan)
        elif choice == "2":
            scan = _scan_currency("ETH")
            if scan:
                _present_scan(scan)
                _interactive_log_prompt(scan)
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
