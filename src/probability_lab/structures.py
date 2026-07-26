"""Option structures, EV/PoP evaluation under a density, and ranking.

v1 universe: long call, long put, call debit vertical, put debit vertical.
Strikes are drawn from the real listed chain; entry cost from market mids.
All EVs are per single contract, in dollars (100x multiplier).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.probability_lab.rnd import Density

MULT = 100  # US equity option contract multiplier


@dataclass
class Structure:
    name: str
    legs: List[tuple]          # (opt_type, strike, qty); qty>0 long, <0 short
    entry_cost: float          # net debit per share (>0 = you pay)
    strikes_label: str

    def payoff_at(self, S_T):
        S_T = np.asarray(S_T, dtype=float)
        total = np.zeros_like(S_T)
        for opt_type, strike, qty in self.legs:
            intrinsic = (np.maximum(S_T - strike, 0.0) if opt_type == "call"
                         else np.maximum(strike - S_T, 0.0))
            total = total + qty * intrinsic
        return total if total.ndim else float(total)


def evaluate(structure: Structure, density: Density) -> dict:
    exp_payoff = density.expected_payoff(structure.payoff_at)
    ev = (exp_payoff - structure.entry_cost) * MULT
    pop = density.prob_payoff_exceeds(structure.payoff_at, structure.entry_cost)
    return {"ev": ev, "pop": pop}


def _mid(row) -> float:
    b, a = float(row["bid"]), float(row["ask"])
    if b <= 0 or a <= 0:
        return max(b, a)
    return 0.5 * (b + a)


def enumerate_structures(chain: pd.DataFrame, S: float) -> List[Structure]:
    calls = chain[chain["type"] == "call"].sort_values("strike").reset_index(drop=True)
    puts = chain[chain["type"] == "put"].sort_values("strike").reset_index(drop=True)
    out: List[Structure] = []
    if calls.empty or puts.empty:
        return out

    # ATM strike = nearest listed to spot.
    atm_c = calls.iloc[(calls["strike"] - S).abs().argmin()]
    atm_p = puts.iloc[(puts["strike"] - S).abs().argmin()]

    out.append(Structure(f"Long {atm_c['strike']:.0f} call",
                         [("call", float(atm_c["strike"]), 1)],
                         _mid(atm_c), f"{atm_c['strike']:.0f}"))
    out.append(Structure(f"Long {atm_p['strike']:.0f} put",
                         [("put", float(atm_p["strike"]), 1)],
                         _mid(atm_p), f"{atm_p['strike']:.0f}"))

    # Call debit vertical: long ATM, short ~+5% strike.
    higher = calls[calls["strike"] > atm_c["strike"]]
    if not higher.empty:
        tgt = atm_c["strike"] * 1.05
        short_c = higher.iloc[(higher["strike"] - tgt).abs().argmin()]
        out.append(Structure(
            f"{atm_c['strike']:.0f}/{short_c['strike']:.0f} call vert",
            [("call", float(atm_c["strike"]), 1), ("call", float(short_c["strike"]), -1)],
            _mid(atm_c) - _mid(short_c),
            f"{atm_c['strike']:.0f}/{short_c['strike']:.0f}"))

    # Put debit vertical: long ATM, short ~-5% strike.
    lower = puts[puts["strike"] < atm_p["strike"]]
    if not lower.empty:
        tgt = atm_p["strike"] * 0.95
        short_p = lower.iloc[(lower["strike"] - tgt).abs().argmin()]
        out.append(Structure(
            f"{atm_p['strike']:.0f}/{short_p['strike']:.0f} put vert",
            [("put", float(atm_p["strike"]), 1), ("put", float(short_p["strike"]), -1)],
            _mid(atm_p) - _mid(short_p),
            f"{atm_p['strike']:.0f}/{short_p['strike']:.0f}"))
    return out


def rank(structures, view: Density, market: Density) -> List[dict]:
    rows = []
    for s in structures:
        v = evaluate(s, view)
        m = evaluate(s, market)
        rows.append({"name": s.name, "strikes": s.strikes_label,
                     "entry": s.entry_cost, "ev_view": v["ev"],
                     "pop_view": v["pop"], "ev_market": m["ev"]})
    rows.sort(key=lambda r: r["ev_view"], reverse=True)
    return rows


__all__ = ["Structure", "evaluate", "enumerate_structures", "rank", "MULT"]
