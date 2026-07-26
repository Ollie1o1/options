"""Probability Lab CLI: risk-neutral density + view-based structure ranking.

Pure helpers (parse_drift, render_report) are unit-tested; build_context does
the live chain fetch and main() wires it together.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from src import formatting as fmt
from src import ui
from src.probability_lab.rnd import rnd_from_smile
from src.probability_lab.structures import enumerate_structures, rank
from src.probability_lab.view import apply_view

WIDTH = 88


# ----------------------------------------------------------------------------- pure helpers
def parse_drift(s: str) -> float:
    """'+3%' -> 0.03, '-2.5%' -> -0.025, '5' -> 0.05, '0' -> 0.0."""
    s = str(s).strip().rstrip("%")
    if not s:
        return 0.0
    return float(s) / 100.0


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _ev_cell(ev: float) -> str:
    txt = f"{ev:+.0f}"
    return fmt.style(txt, "good" if ev >= 0 else "bad")


def render_report(ctx: dict) -> List[str]:
    c = ctx
    out: List[str] = []
    out.append(ui.rule(WIDTH, "PROBABILITY LAB"))
    conf = c["confidence"]
    tag = ("RND: SVI smile" if conf.get("source") == "svi"
           else "RND: lognormal fallback (thin chain)")
    out.append(
        f"  {fmt.style(c['ticker'], 'label', bold=True)}   spot {c['spot']:.2f}"
        f"   exp {c['expiry']} ({c['dte']}d)   r {c['r'] * 100:.2f}%"
        f"   {fmt.style(tag, 'muted')}"
    )

    # Density sketch (market RND).
    market = c["market"]
    sketch = ui.braille_chart(market.pdf, width=WIDTH - 8, height=4,
                              style_name="accent")
    if sketch:
        out.append("")
        out.extend(sketch)
        out.append(f"  {fmt.style('market-implied density of price at expiry', 'muted')}")

    # Probability table: market vs your view at key levels.
    out.append("")
    out.append(ui.rule(WIDTH, "PROBABILITIES  ·  P(S_T > level)"))
    cols = [
        {"h": "Level", "w": 10, "align": "right"},
        {"h": "%spot", "w": 8, "align": "right"},
        {"h": "Market", "w": 9, "align": "right"},
        {"h": "Your view", "w": 11, "align": "right"},
        {"h": "Edge", "w": 9, "align": "right"},
    ]
    rows = []
    for x in c["levels"]:
        pm = market.prob_above(x)
        pv = c["view"].prob_above(x)
        edge = pv - pm
        edge_cell = fmt.style(f"{edge * 100:+.0f}pt",
                              "good" if edge >= 0 else "bad")
        rows.append([f"{x:.2f}", f"{x / c['spot'] * 100:.0f}%",
                     _fmt_pct(pm), _fmt_pct(pv), edge_cell])
    out.append(ui.table(cols, rows))
    out.append(
        f"  E[S_T]  market {market.mean():.2f}   "
        f"your {c['view'].mean():.2f}   "
        f"(drift {c['drift'] * 100:+.1f}%, vol ×{c['vol_mult']:.2f})"
    )

    # EV-ranked structures.
    out.append("")
    out.append(ui.rule(WIDTH, "STRUCTURES  ·  ranked by EV under your view"))
    scols = [
        {"h": "#", "w": 2, "align": "right"},
        {"h": "Structure", "w": 22, "align": "left"},
        {"h": "Strikes", "w": 12, "align": "left"},
        {"h": "Entry$", "w": 8, "align": "right"},
        {"h": "EV(view)", "w": 10, "align": "right"},
        {"h": "PoP", "w": 6, "align": "right"},
        {"h": "EV(mkt)", "w": 9, "align": "right"},
    ]
    srows = []
    for i, r in enumerate(c["ranked"], 1):
        srows.append([str(i), r["name"], r["strikes"],
                      f"{r['entry'] * 100:.0f}",
                      _ev_cell(r["ev_view"]), _fmt_pct(r["pop_view"]),
                      _ev_cell(r["ev_market"])])
    out.append(ui.table(scols, srows))

    out.append("")
    out.append("  " + fmt.style(
        "RND is model-implied off delayed, SVI-smoothed data; view EV is only as "
        "good as your view.", "muted"))
    out.append("  " + fmt.style(
        "Decision-support, not a gated edge. EV(mkt)≈0 means you're buying "
        "consensus.", "muted"))
    return out


# ----------------------------------------------------------------------------- live path
def _dte(expiration: str) -> int:
    try:
        d = datetime.strptime(expiration, "%Y-%m-%d").date()
        return max((d - date.today()).days, 0)
    except Exception:
        return 0


def _risk_free_rate() -> float:
    try:
        from src.macro_rates import fetch_rates_snapshot
        snap = fetch_rates_snapshot()
        if snap is not None and snap.dgs3mo is not None:
            return float(snap.dgs3mo) / 100.0
    except Exception:
        pass
    return 0.04  # sane default when rates are unavailable


def _build_otm_smile(exp_df: pd.DataFrame, spot: float):
    """OTM smile: puts at/below spot, calls above. Returns (strikes, ivs)."""
    calls = exp_df[exp_df["type"] == "call"]
    puts = exp_df[exp_df["type"] == "put"]
    otm_p = puts[puts["strike"] <= spot]
    otm_c = calls[calls["strike"] > spot]
    combo = pd.concat([otm_p, otm_c], ignore_index=True)
    combo = combo[["strike", "impliedVolatility"]].dropna()
    combo = combo[(combo["impliedVolatility"] > 0) & (combo["strike"] > 0)]
    combo = combo.sort_values("strike")
    return combo["strike"].to_numpy(float), combo["impliedVolatility"].to_numpy(float)


def _pick_expiry(df: pd.DataFrame, requested: Optional[str]) -> str:
    exps = sorted(df["expiration"].dropna().unique())
    if not exps:
        raise RuntimeError("no expirations in chain")
    if requested:
        if requested in exps:
            return requested
        raise RuntimeError(f"expiry {requested} not listed; available: {', '.join(exps[:8])}")
    # Nearest expiry with >= 25 DTE; else the farthest available.
    for e in exps:
        if _dte(e) >= 25:
            return e
    return exps[-1]


def build_context(ticker: str, expiry: Optional[str], drift: float,
                  vol_mult: float) -> dict:
    from src.data_fetching import fetch_options_yfinance
    res = fetch_options_yfinance(ticker.upper(), max_expiries=8)
    df = res["df"]
    if df is None or df.empty:
        raise RuntimeError(f"no options chain for {ticker}")
    spot = float(df["underlying"].iloc[0])
    exp = _pick_expiry(df, expiry)
    exp_df = df[df["expiration"] == exp].copy()
    T = max(_dte(exp), 1) / 365.0
    r = _risk_free_rate()

    strikes, ivs = _build_otm_smile(exp_df, spot)
    if len(strikes) < 5:
        raise RuntimeError(f"too few valid quotes on {exp} to build a density")

    conf: dict = {}
    market = rnd_from_smile(strikes, ivs, T, spot, r, conf)
    view = apply_view(market, drift, vol_mult)

    chain = exp_df[["type", "strike", "bid", "ask"]].copy()
    ranked = rank(enumerate_structures(chain, spot), view, market)

    levels = [round(spot * m, 2) for m in (0.90, 0.95, 1.00, 1.05, 1.10)]
    return {"ticker": ticker.upper(), "spot": spot, "expiry": exp,
            "dte": _dte(exp), "r": r, "confidence": conf, "market": market,
            "view": view, "drift": drift, "vol_mult": vol_mult,
            "ranked": ranked, "levels": levels}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.probability_lab",
        description="Risk-neutral density + view-based option structure ranking.")
    p.add_argument("ticker", help="underlying symbol, e.g. AAPL")
    p.add_argument("--expiry", default=None, metavar="YYYY-MM-DD",
                   help="option expiry (default: nearest listed >= 25 DTE)")
    p.add_argument("--drift", default="0", metavar="PCT",
                   help="your directional view over the horizon, e.g. +3%%")
    p.add_argument("--vol-mult", type=float, default=1.0, metavar="X",
                   help="your vol vs the market's, e.g. 0.9 for calmer")
    args = p.parse_args(argv)

    try:
        ctx = build_context(args.ticker, args.expiry, parse_drift(args.drift),
                            args.vol_mult)
    except Exception as e:
        print(f"  probability_lab: {e}")
        return 1
    print("\n".join(render_report(ctx)))
    return 0


__all__ = ["parse_drift", "render_report", "build_context", "main"]
