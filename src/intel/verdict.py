"""Verdict engine — blend backtest-weighted signals into BUY / WAIT / AVOID.

Pure and deterministic. The composite is a reliability-weighted average of the
directional signals; only signals that earned weight in the backtest move it.
Confidence reflects how much weight actually fired and how much the signals
agree. An imminent earnings event caps confidence and can force WAIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.intel.signals import DIRECTIONAL_NAMES, Signal

# Composite thresholds (tunable; validated against the backtest).
BUY_AT = 0.25
AVOID_AT = -0.60
WAIT_AT = -0.25


@dataclass
class Driver:
    name: str
    glyph: str       # '+', '-', or '~' (context/zero-weight)
    text: str        # "trend up (200d)"
    tag: str         # reliability tag


@dataclass
class Verdict:
    call: str                       # BUY / WAIT / AVOID / NEUTRAL
    confidence: str                 # low / medium / high
    composite: float                # [-1, +1]
    drivers: List[Driver] = field(default_factory=list)
    note: str = ""


def decide(signals: Dict[str, Signal], reliability: Dict[str, Any]) -> Verdict:
    num = den = 0.0
    weighted = []  # (abs_contribution, signal, weight)
    for name in DIRECTIONAL_NAMES:
        sig = signals.get(name)
        if sig is None:
            continue
        w = float(reliability.get(name, {}).get("weight", 0.0) or 0.0)
        if w > 0 and sig.value != 0.0:
            num += sig.value * w
            den += w
        weighted.append((abs(sig.value * w), sig, w))

    composite = (num / den) if den > 0 else 0.0

    # ── Call from composite ──
    if composite >= BUY_AT:
        call = "BUY"
    elif composite <= AVOID_AT:
        call = "AVOID"
    elif composite <= WAIT_AT:
        call = "WAIT"
    else:
        call = "NEUTRAL"

    # ── Earnings gate ──
    note = ""
    earn = signals.get("earnings")
    if earn is not None and earn.value <= -1.0:
        note = "earnings imminent — event risk caps this to WAIT"
        if call == "BUY":
            call = "WAIT"

    # ── Confidence: weight that fired × agreement among weighted signals ──
    agree = _agreement(weighted)
    if den >= 1.0 and agree >= 0.6:
        confidence = "high"
    elif den >= 0.4 and agree >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"
    if note and confidence == "high":
        confidence = "medium"  # never claim high into a binary event

    # Low confidence = little measured edge fired; don't issue a strong
    # directional call. Cap it to a cautious lean so the verdict and the
    # playbook can't contradict each other on near-random readings.
    if confidence == "low":
        if call == "BUY":
            call = "NEUTRAL"
        elif call == "AVOID":
            call = "WAIT"

    # ── Drivers, ranked by |value × weight| (weighted first, then context) ──
    drivers: List[Driver] = []
    for name in DIRECTIONAL_NAMES:
        sig = signals.get(name)
        if sig is None or sig.value == 0.0:
            continue
        w = float(reliability.get(name, {}).get("weight", 0.0) or 0.0)
        tag = reliability.get(name, {}).get("tag", "")
        glyph = "~" if w == 0 else ("+" if sig.value > 0 else "-")
        drivers.append((abs(sig.value * w), w,
                        Driver(name, glyph, f"{name} {sig.label}", tag)))
    drivers.sort(key=lambda d: (d[1] > 0, d[0]), reverse=True)
    ranked = [d[2] for d in drivers]

    return Verdict(call=call, confidence=confidence, composite=round(composite, 3),
                   drivers=ranked, note=note)


def _agreement(weighted) -> float:
    """Fraction of weighted directional signal-mass that points the same way."""
    contribs = [(s.value * w) for (_, s, w) in weighted if w > 0 and s.value != 0]
    if not contribs:
        return 0.0
    pos = sum(c for c in contribs if c > 0)
    neg = -sum(c for c in contribs if c < 0)
    total = pos + neg
    return (max(pos, neg) / total) if total > 0 else 0.0
