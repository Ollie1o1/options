"""Model-based backtest for the lottery-ticket sleeve.

Historical option chains + IV are not available from our data sources, so this
is a *model-based simulation* on real underlying price paths:

  • features (realized vol, momentum, an IV proxy, IV rank) are computed from
    daily closes up to the decision day — no look-ahead;
  • the entry option is priced with Black-Scholes at an IV proxy
    (longer-window realized vol + a volatility-risk-premium markup), struck a
    fixed number of sigmas OTM;
  • the outcome is the option's intrinsic value at expiry along the *actual*
    realized price path, minus entry cost and a slippage markup.

Assumptions (stated honestly — this measures *relative* edge between selection
rules, not a promise of live P&L):
  • IV proxy = 60d realized × (1 + vrp); no real implied surface, no skew;
  • no historical earnings calendar, so the catalyst factor is inert here;
  • hold-to-expiry (no early take-profit) unless --take-profit is given;
  • flat risk-free rate, no dividends, European exercise.

The point: does evidence-based selection (cheap vol + trend) beat a naive
"buy the hot/expensive name" rule and a blind "buy one of everything" baseline?
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils import bs_price
from src.lottery.selector import DEFAULT_LOTTERY_CONFIG, load_lottery_config, select_best

RISK_FREE = 0.04
TRADING_DAYS = 252


def _ann_vol(closes: List[float]) -> float:
    """Annualised vol from a window of closes (log returns)."""
    if closes is None or len(closes) < 3:
        return float("nan")
    arr = np.asarray(closes, dtype=float)
    rets = np.diff(np.log(arr))
    if rets.size < 2:
        return float("nan")
    return float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS))


def build_candidate(
    ticker: str, closes: List[float], t_idx: int, cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Construct a far-OTM lottery candidate as of close on day t_idx.

    Returns None if there is insufficient history. No look-ahead: every input
    uses closes[: t_idx + 1] only.
    """
    if closes is None or t_idx < 63 or t_idx >= len(closes):
        return None

    hist = closes[: t_idx + 1]
    spot = float(hist[-1])
    if spot <= 0:
        return None

    rv_short = _ann_vol(hist[-11:])   # ~10d realized
    rv_long = _ann_vol(hist[-61:])    # ~60d realized
    if not math.isfinite(rv_short) or not math.isfinite(rv_long) or rv_long <= 0:
        return None

    # Implied-vol proxy. Realistic entry pricing matters: IV trades ABOVE realized
    # (the volatility risk premium), and far-OTM strikes carry an extra skew
    # markup. Underpricing entry would make timing irrelevant and inflate payoffs.
    vrp = cfg.get("vrp", 0.12)
    iv_atm = rv_long * (1.0 + vrp)

    momentum_raw = spot / float(hist[-21]) - 1.0  # 20d return
    direction = "call" if momentum_raw >= 0 else "put"

    # IV rank: percentile of current short-vol within its trailing-1y distribution.
    lookback = hist[-min(len(hist), TRADING_DAYS + 11):]
    daily_vol = []
    for j in range(11, len(lookback)):
        daily_vol.append(_ann_vol(lookback[j - 11:j + 1]))
    daily_vol = [v for v in daily_vol if math.isfinite(v)]
    if daily_vol:
        iv_rank = float(sum(1 for v in daily_vol if v <= rv_short) / len(daily_vol))
    else:
        iv_rank = 0.5

    dte = int(cfg.get("dte_target", 14))
    t_years = dte / 365.0
    n_sigma = float(cfg.get("entry_sigma", cfg.get("convexity_sigma_target", 2.0)))
    sigma_move = iv_atm * math.sqrt(t_years)
    if direction == "call":
        strike = spot * math.exp(n_sigma * sigma_move)
    else:
        strike = spot * math.exp(-n_sigma * sigma_move)

    # Skew markup: the further OTM, the richer the implied vol you actually pay.
    skew = cfg.get("skew_per_sigma", 0.08)
    iv_entry = iv_atm * (1.0 + skew * n_sigma)

    premium = float(bs_price(direction, spot, strike, t_years, RISK_FREE, iv_entry))
    if not math.isfinite(premium) or premium <= 0:
        return None

    expiry_offset = max(1, round(dte * TRADING_DAYS / 365.0))

    return {
        "ticker": ticker,
        "direction": direction,
        "spot": spot,
        "strike": strike,
        "premium": premium,
        "iv": iv_entry,
        "realized_vol": rv_short,
        "vol_level": rv_long,
        "iv_rank": iv_rank,
        "momentum": abs(momentum_raw),
        "strike_sigma": n_sigma,
        "has_catalyst": False,
        "spread_pct": None,       # no historical quote; cost applied via slippage
        "open_interest": None,
        "t_idx": t_idx,
        "expiry_offset": expiry_offset,
        "dte": dte,
    }


def simulate_outcome(
    direction: str, strike: float, premium: float, s_exp: float,
    qty: float, cost_mult: float = 1.0,
    s_fav: Optional[float] = None, take_profit: Optional[float] = None,
) -> Dict[str, Any]:
    """Outcome of one lottery ticket.

    Held to expiry by default (payoff = intrinsic at s_exp). If a take-profit
    multiple and the holding-window favorable extreme (s_fav) are supplied, the
    ticket is booked early at take_profit × premium when the underlying's best
    move would have put the option's intrinsic at/above that level — i.e. we
    sell into the spike instead of riding it back down.
    """
    cost_usd = premium * qty * cost_mult
    if take_profit is not None and s_fav is not None:
        fav_intrinsic = (s_fav - strike) if direction == "call" else (strike - s_fav)
        fav_intrinsic = max(fav_intrinsic, 0.0)
        if fav_intrinsic >= take_profit * premium:
            payoff_usd = take_profit * premium * qty
            return {"payoff_usd": payoff_usd, "cost_usd": cost_usd,
                    "pnl_usd": payoff_usd - cost_usd, "win": True}
    if direction == "call":
        intrinsic = max(s_exp - strike, 0.0)
    else:
        intrinsic = max(strike - s_exp, 0.0)
    payoff_usd = intrinsic * qty
    return {
        "payoff_usd": payoff_usd,
        "cost_usd": cost_usd,
        "pnl_usd": payoff_usd - cost_usd,
        "win": payoff_usd > cost_usd,
    }


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sleeve statistics — the win-rate / breakeven the experiment is about."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_cost": 0.0, "total_payoff": 0.0, "net_pnl": 0.0,
                "roi": 0.0, "avg_win_multiple": 0.0, "breakeven_win_rate": None}
    wins = [t for t in trades if t["win"]]
    total_cost = sum(t["cost_usd"] for t in trades)
    total_payoff = sum(t["payoff_usd"] for t in trades)
    net_pnl = total_payoff - total_cost
    win_mults = [t["payoff_usd"] / t["cost_usd"] for t in wins if t["cost_usd"] > 0]
    avg_win_multiple = float(sum(win_mults) / len(win_mults)) if win_mults else 0.0
    breakeven = (1.0 / avg_win_multiple) if avg_win_multiple > 0 else None
    return {
        "n": n,
        "wins": len(wins),
        "losses": n - len(wins),
        "win_rate": len(wins) / n,
        "total_cost": total_cost,
        "total_payoff": total_payoff,
        "net_pnl": net_pnl,
        "roi": net_pnl / total_cost if total_cost > 0 else 0.0,
        "avg_win_multiple": avg_win_multiple,
        "breakeven_win_rate": breakeven,
    }


# ── orchestration (uses real data) ─────────────────────────────────────────────
# Deliberately mixed to fight survivorship bias: explosive names AND laggards/
# boring large caps that did NOT moon over the window.
DEFAULT_BASKET = [
    # high-vol / "explosive"
    "TSLA", "NVDA", "AMD", "COIN", "MSTR", "PLTR", "SMCI", "MARA", "RIVN",
    # mid
    "META", "AMZN", "NFLX", "UBER", "SOFI", "AAPL", "MSFT", "GOOGL",
    # laggards / boring / fallen names (anti-survivorship)
    "INTC", "PYPL", "DIS", "F", "NKE", "PFE", "BA", "T", "KO", "WMT", "VZ",
]


def _fetch_closes(ticker: str, period: str = "2y") -> Optional[List[float]]:
    try:
        import warnings
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.Ticker(ticker).history(period=period, interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return [float(x) for x in df["Close"].dropna().tolist()]
    except Exception:
        return None


def run_backtest(
    tickers: List[str],
    cfg: Dict[str, Any],
    bet_usd: float = 100.0,
    slippage: float = 0.05,
    draw_freq: int = 5,         # trading days between draws (~weekly)
    take_profit: Optional[float] = None,  # e.g. 5.0 = sell at 5x (None = hold to expiry)
    period: str = "2y",
    closes_by_ticker: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Run the three strategies over the same draw dates and return summaries.

    strategies: 'smart' (evidence-based select_best), 'naive' (buy the
    highest-IV-rank / hottest name = the Boyer-Vorkink trap), 'blind' (buy one
    of every eligible candidate each draw = the average-lottery baseline).
    """
    if closes_by_ticker is None:
        closes_by_ticker = {}
        for t in tickers:
            c = _fetch_closes(t, period)
            if c and len(c) > 130:
                closes_by_ticker[t] = c
    usable = {t: c for t, c in closes_by_ticker.items() if c and len(c) > 130}
    if not usable:
        return {"error": "no usable price history", "smart": summarize([]),
                "naive": summarize([]), "blind": summarize([]), "n_draws": 0}

    max_len = max(len(c) for c in usable.values())
    cost_mult = 1.0 + slippage
    trades = {"smart": [], "naive": [], "blind": []}
    n_draws = 0

    start = 130
    for t_idx in range(start, max_len, draw_freq):
        candidates = []
        for t, c in usable.items():
            if t_idx >= len(c):
                continue
            cand = build_candidate(t, c, t_idx, cfg)
            if cand is None:
                continue
            exp_idx = t_idx + cand["expiry_offset"]
            if exp_idx >= len(c):
                continue  # no realized expiry in-sample
            cand["s_exp"] = float(c[exp_idx])
            path = c[t_idx + 1: exp_idx + 1]
            cand["s_fav"] = (max(path) if cand["direction"] == "call" else min(path)) if path else cand["s_exp"]
            candidates.append(cand)
        if not candidates:
            continue
        n_draws += 1

        def _trade_from(cand):
            qty = bet_usd / cand["premium"]
            return simulate_outcome(
                cand["direction"], cand["strike"], cand["premium"], cand["s_exp"],
                qty, cost_mult, s_fav=cand.get("s_fav"), take_profit=take_profit,
            )

        # smart
        smart = select_best(candidates, cfg)
        if smart is not None:
            trades["smart"].append(_trade_from(smart))
        # naive: hottest / most expensive (highest IV rank) — the trap
        naive = max(candidates, key=lambda c: (c["iv_rank"] if c["iv_rank"] is not None else 0))
        trades["naive"].append(_trade_from(naive))
        # blind: one of each eligible candidate
        for cand in candidates:
            trades["blind"].append(_trade_from(cand))

    return {
        "smart": summarize(trades["smart"]),
        "naive": summarize(trades["naive"]),
        "blind": summarize(trades["blind"]),
        "n_draws": n_draws,
        "n_tickers": len(usable),
    }


def calibrate_live(
    tickers: List[str], cfg: Dict[str, Any], provider_name: Optional[str] = None,
    closes_by_ticker: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Measure real VRP + skew from live option chains and fold them into cfg.

    Uses the free yfinance provider by default; pass provider_name='polygon'
    (with a subscription + key) for a paid feed. Returns the calibration dict;
    mutates cfg['vrp'] and cfg['skew_per_sigma'] in place when measurable.
    """
    from src.lottery.data import get_provider, atm_iv_from_chain, calibrate_from_chains

    provider = get_provider(provider_name)
    samples = []
    for t in tickers:
        closes = (closes_by_ticker or {}).get(t) or _fetch_closes(t, "6mo")
        if not closes or len(closes) < 61:
            continue
        chain = provider.get_chain(t, target_dte=int(cfg.get("dte_target", 14)))
        if not chain:
            continue
        atm = atm_iv_from_chain(chain.get("calls") or [], chain["spot"])
        rv_long = _ann_vol(closes[-61:])
        if atm and rv_long and rv_long > 0:
            samples.append({"atm_iv": atm, "realized_vol": rv_long,
                            "spot": chain["spot"], "t_years": chain["t_years"],
                            "calls": chain["calls"]})
    cal = calibrate_from_chains(samples)
    cal["provider"] = provider.name
    if cal["n_samples"] > 0:
        cfg["vrp"] = cal["vrp"]
        cfg["skew_per_sigma"] = cal["skew_per_sigma"]
    return cal


def _fmt_summary(name: str, s: Dict[str, Any]) -> str:
    be = s["breakeven_win_rate"]
    be_str = f"{be:.1%}" if be is not None else "n/a"
    return (
        f"  {name:<7} n={s['n']:<4} win={s['win_rate']:.1%}  "
        f"avgWinX={s['avg_win_multiple']:.1f}  breakeven={be_str}  "
        f"net=${s['net_pnl']:>10,.0f}  ROI={s['roi']:+.1%}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Lottery-ticket sleeve backtest")
    p.add_argument("tickers", nargs="*", default=DEFAULT_BASKET)
    p.add_argument("--bet", type=float, default=100.0)
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--draw-freq", type=int, default=5)
    p.add_argument("--take-profit", type=float, default=None)
    p.add_argument("--period", default="2y")
    p.add_argument("--config", default="config.json")
    p.add_argument("--calibrate", action="store_true",
                   help="measure real VRP+skew from live chains before backtesting")
    p.add_argument("--provider", default=None,
                   help="option-data provider: yfinance (free, default) or polygon (paid)")
    args = p.parse_args(argv)

    cfg = load_lottery_config(args.config)
    tickers = args.tickers or DEFAULT_BASKET
    cal_note = "model-guessed pricing"
    if args.calibrate:
        cal = calibrate_live(tickers, cfg, provider_name=args.provider)
        cal_note = (f"calibrated via {cal['provider']} "
                    f"(n={cal['n_samples']}, VRP={cfg['vrp']:.3f}, "
                    f"skew/σ={cfg['skew_per_sigma']:.3f})")
    res = run_backtest(tickers, cfg, bet_usd=args.bet, slippage=args.slippage,
                       draw_freq=args.draw_freq, take_profit=args.take_profit,
                       period=args.period)
    print()
    print("  LOTTERY SLEEVE BACKTEST  "
          f"(tickers={res.get('n_tickers', 0)}, draws={res.get('n_draws', 0)}, "
          f"bet=${args.bet:.0f}, slippage={args.slippage:.0%}, "
          f"TP={'hold' if args.take_profit is None else str(args.take_profit)+'x'})")
    print(f"  pricing: {cal_note}")
    if res.get("error"):
        print("  ERROR:", res["error"])
        return
    print("  " + "-" * 96)
    print(_fmt_summary("SMART", res["smart"]))
    print(_fmt_summary("naive", res["naive"]))
    print(_fmt_summary("blind", res["blind"]))
    print()


if __name__ == "__main__":
    main()
